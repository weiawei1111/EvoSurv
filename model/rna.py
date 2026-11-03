"""RNA branch: preprocessing, autoencoder, Cox utilities and data splits."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def concordance_index(time, event, risk):
    """Harrell's C-index; a higher risk means earlier death."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event).astype(bool)
    r = np.asarray(risk, dtype=float)
    comparable = e[:, None] & (t[:, None] < t[None, :])
    diff = r[:, None] - r[None, :]
    num = ((diff > 0) & comparable).sum() + 0.5 * ((diff == 0) & comparable).sum()
    den = comparable.sum()
    return float(num / den) if den > 0 else float("nan")


def cox_ph_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow ties), averaged over events."""
    event = event.float()
    if event.sum() == 0:
        return risk.sum() * 0.0
    order = torch.argsort(time, descending=True)
    r, t, e = risk[order], time[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    neg_t = -t
    last = torch.searchsorted(neg_t.contiguous(), neg_t.contiguous(), right=True) - 1
    return -((r - log_cum[last]) * e).sum() / e.sum()


class RNAPreprocessor:
    """Fit on training data only. Input is log2(TPM + 1).

    Keeps genes with TPM > min_tpm in at least min_frac of patients, selects the
    top_k genes by median absolute deviation, then z-scores and clips each gene.
    """

    def __init__(self, min_tpm=1.0, min_frac=0.2, top_k=2000, clip=5.0):
        self.min_tpm, self.min_frac, self.top_k, self.clip = min_tpm, min_frac, top_k, clip

    def fit(self, X_df):
        X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0)
        genes = np.asarray(X_df.columns)
        keep = (X > np.log2(self.min_tpm + 1.0)).mean(axis=0) >= self.min_frac
        keep &= X.std(axis=0) > 1e-8
        X, genes = X[:, keep], genes[keep]
        if len(genes) > self.top_k:
            mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
            idx = np.sort(np.argsort(-mad)[: self.top_k])
            X, genes = X[:, idx], genes[idx]
        self.genes_ = genes
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        return self

    def transform(self, X_df):
        X = X_df.reindex(columns=self.genes_).values.astype(np.float64)
        missing = np.isnan(X).all(axis=0)
        X = np.nan_to_num(X, nan=0.0)
        X[:, missing] = self.mean_[missing]
        X = np.clip((X - self.mean_) / self.std_, -self.clip, self.clip)
        return X.astype(np.float32)

    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)


class RNAAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.3):
        super().__init__()
        enc_h = [min(512, input_dim), min(256, input_dim)]
        dec_h = [min(256, input_dim), min(512, input_dim)]
        enc, prev = [], input_dim
        for h in enc_h:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], latent_dim
        for h in dec_h:
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _to_t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


def train_rna_ae(X_tr, X_va, latent_dim=64, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 epochs=300, batch_size=1024, patience=30, seed=0, device=None):
    """Pretrain the autoencoder with reconstruction loss only; early stop on validation loss."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xva = _to_t(X_tr, device), _to_t(X_va, device)
    model = RNAAutoEncoder(Xtr.shape[1], latent_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = Xtr.shape[0]
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            xb = Xtr[idx]
            loss = F.mse_loss(model(xb)[0], xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(Xva)[0], Xva).item()
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def encode_numpy(model, X):
    device = next(model.parameters()).device
    model.eval()
    return model.encoder(_to_t(X, device)).cpu().numpy()


def fit_linear_cox(Z_tr, t_tr, e_tr, l2=1e-2, steps=500, lr=1e-2):
    """L2-penalized linear Cox model on training-standardized inputs."""
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    Z = torch.tensor((Z_tr - mu) / sd, dtype=torch.float32)
    t = torch.tensor(t_tr, dtype=torch.float32)
    e = torch.tensor(e_tr, dtype=torch.float32)
    w = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = cox_ph_loss(Z @ w, t, e) + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = w.detach().numpy()
    return lambda Znew: ((Znew - mu) / sd) @ w


L2_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def fit_linear_cox_val(Z_tr, t_tr, e_tr, Z_va, t_va, e_va, l2_grid=L2_GRID):
    """Linear Cox model with the L2 penalty chosen by validation C-index.

    Returns (predict, chosen l2, validation C-index).
    """
    best = None
    for l2 in l2_grid:
        pred = fit_linear_cox(Z_tr, t_tr, e_tr, l2=l2)
        c = concordance_index(t_va, e_va, pred(Z_va))
        if best is None or c > best[2]:
            best = (pred, l2, c)
    return best


def stratified_split(event, seed, test_frac=0.2, val_frac=0.2):
    """Event-stratified 64/16/20 train/validation/test split."""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(event == cls)[0]
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test_frac))
        rest = idx[n_te:]
        n_va = int(round(len(rest) * val_frac))
        te += list(idx[:n_te])
        va += list(rest[:n_va])
        tr += list(rest[n_va:])
    return np.array(tr), np.array(va), np.array(te)


def load_data(rna_path, surv_path, cohort_path="", id_col="patient_id",
              time_col="OS_time_days", event_col="OS_event"):
    """Expression matrix and survival for patients with both, sorted by patient ID."""
    rna = pd.read_csv(rna_path)
    rid = rna.columns[0] if id_col not in rna.columns else id_col
    rna["_pid"] = rna[rid].astype(str).str[:12]
    rna = rna.drop(columns=[rid]).drop_duplicates("_pid").set_index("_pid")

    surv = pd.read_csv(surv_path)
    surv["_pid"] = surv[id_col].astype(str).str[:12]
    surv = surv[["_pid", time_col, event_col]].dropna().drop_duplicates("_pid")
    surv = surv[surv[time_col] > 0].set_index("_pid")

    common = rna.index.intersection(surv.index)
    if cohort_path:
        cohort = pd.Index(sorted(set(x[:12] for x in open(cohort_path).read().split())))
        common = cohort.intersection(common)
    common = pd.Index(sorted(common))
    rna = rna.loc[common].apply(pd.to_numeric, errors="coerce")
    t = surv.loc[common, time_col].values.astype(float)
    e = surv.loc[common, event_col].values.astype(int)
    return rna, t, e
