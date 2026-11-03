"""EvoSurv joint model and training."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import rna as R


@dataclass
class HParams:
    epochs: int = 500
    patience: int = 50
    min_epochs: int = 20
    ema: float = 0.9             
    lr: float = 1e-3
    weight_decay: float = 1e-3
    enc_lr_mult: float = 0.1    # learning-rate multiplier for the pretrained RNA encoder
    l2sp: float = 1e-2          # penalty on deviation from the pretrained encoder weights
    h_et: int = 16
    dropout: float = 0.3
    mod_drop: float = 0.2       # modality dropout probability
    risk_l2: float = 1e-3       # penalty on branch risk magnitude


class JointSurv(nn.Module):
    """RNA encoder (fine-tuned) and eTime MLP, each with a bias-free linear risk head."""

    def __init__(self, rna_encoder, d_rna, in_et, h_et=16, dropout=0.3):
        super().__init__()
        self.enc_rna = copy.deepcopy(rna_encoder)
        self.enc_et = nn.Sequential(
            nn.Linear(in_et, h_et), nn.LayerNorm(h_et), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h_et, h_et),
        )
        self.head_rna = nn.Linear(d_rna, 1, bias=False)
        self.head_et = nn.Linear(h_et, 1, bias=False)

    def forward(self, xr, xe):
        rr = self.head_rna(self.enc_rna(xr)).squeeze(-1)
        re = self.head_et(self.enc_et(xe)).squeeze(-1)
        return rr, re, rr + re


def build_optimizer(m, hp):
    """Encoder: no weight decay (constrained by L2-SP), reduced learning rate.
    Other weights: weight decay on matrices only."""
    enc, decay, no_decay = [], [], []
    for n, p in m.named_parameters():
        if n.startswith("enc_rna."):
            enc.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "lr": hp.lr, "weight_decay": hp.weight_decay},
        {"params": no_decay, "lr": hp.lr, "weight_decay": 0.0},
        {"params": enc, "lr": hp.lr * hp.enc_lr_mult, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]])


def train_joint(encoder, d_rna, X, A, t, e, idx, hp, seed, dev, offset=None):
    """Train EvoSurv on one split.

    X, A    : RNA and eTime inputs for (train, val, test)
    idx     : (train, val, test) patient indices
    offset  : fixed clinical risk for (train, val, test); None for the model without clinical data
    Final risk = offset + r_RNA + r_eTime. Returns {split: (r_RNA, r_eTime, final risk)} and the best epoch.
    """
    R.set_seed(seed)
    m = JointSurv(encoder, d_rna, A[0].shape[1], h_et=hp.h_et, dropout=hp.dropout).to(dev)
    opt = build_optimizer(m, hp)
    ref = {n: p.detach().clone() for n, p in m.enc_rna.named_parameters()}

    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    xr, xe = [T(x) for x in X], [T(a) for a in A]
    tt, ee = [T(t[i]) for i in idx], [T(e[i]) for i in idx]
    off = [T(o) for o in offset] if offset is not None else [T(np.zeros(len(i))) for i in idx]

    cox = R.cox_ph_loss
    snapshot = lambda: {k: v.detach().clone() for k, v in m.state_dict().items()}
    state = snapshot()
    best, ema, wait, ep_best = float("inf"), None, 0, -1

    for ep in range(hp.epochs):
        m.train()
        rr, re, rf = m(xr[0], xe[0])
        o = off[0]

        # modality dropout: the fused term occasionally uses a single branch
        fused = o + rf
        if hp.mod_drop > 0:
            u = torch.rand(1).item()
            if u < hp.mod_drop / 2:
                fused = o + rr
            elif u < hp.mod_drop:
                fused = o + re

        loss = cox(fused, tt[0], ee[0]) + (cox(o + rr, tt[0], ee[0]) + cox(o + re, tt[0], ee[0]))
        loss = loss + hp.risk_l2 * 0.5 * (rr.pow(2).mean() + re.pow(2).mean())
        loss = loss + hp.l2sp * sum(((p - ref[n]) ** 2).sum() for n, p in m.enc_rna.named_parameters())
        opt.zero_grad()
        loss.backward()
        opt.step()

        m.eval()
        with torch.no_grad():
            vl = cox(off[1] + m(xr[1], xe[1])[2], tt[1], ee[1]).item()
        if not np.isfinite(vl):
            vl = float("inf")
        ema = vl if ema is None else hp.ema * ema + (1 - hp.ema) * vl
        if ep < hp.min_epochs:
            continue
        if ema < best - 1e-5:
            best, wait, ep_best = ema, 0, ep
            state = snapshot()
        else:
            wait += 1
            if wait >= hp.patience:
                break

    m.load_state_dict(state)
    m.eval()
    out = {}
    with torch.no_grad():
        for name, k in (("tr", 0), ("va", 1), ("te", 2)):
            rr, re, rf = m(xr[k], xe[k])
            out[name] = (rr.cpu().numpy(), re.cpu().numpy(), (off[k] + rf).cpu().numpy())
    return out, ep_best
