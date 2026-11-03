import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import rna as R
import etime as E
from model import HParams, train_joint

CANCERS = ["BLCA", "BRCA", "HNSC", "LIHC", "OV", "UCEC"]
CONFIGS = ["evosurv"]


# ---------------------------------------------------------------- data
def norm_pid(x):
    return str(x).strip()[:12]


def load_etime(cancer, base_dir, features_dir, patient_ids):
    """eTime features in the order of patient_ids; spectra and classes as proportions."""
    p = Path(base_dir) / features_dir / cancer / "all_features.csv"
    d = pd.read_csv(p)
    d["patient_id"] = d["patient_id"].map(norm_pid)
    d = d.drop_duplicates("patient_id").set_index("patient_id").apply(pd.to_numeric, errors="coerce")
    X = E.to_props(d)
    cols = [f for fs in E.GROUPS.values() for f in fs]
    return X.loc[[norm_pid(x) for x in patient_ids], cols]


def stage_to_ord(x):
    """TCGA stage string -> 1-4; stage 0, IS, X and missing -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(AJCC|PATHOLOGIC|PATHOLOGICAL|CLINICAL|TUMOR|FIGO)\b", " ", s)
    s = re.sub(r"\bSTAGE\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.match(r"^(0|IS|X)", s):
        return np.nan
    for pat, v in ((r"^IV", 4), (r"^III", 3), (r"^II", 2), (r"^I", 1),
                   (r"^4", 4), (r"^3", 3), (r"^2", 2), (r"^1", 1)):
        if re.match(pat, s):
            return v
    return np.nan


def load_clinical(cancer, cdr_file):
    """Stage (pathologic, else clinical) and age at diagnosis from TCGA-CDR."""
    d = pd.read_excel(cdr_file)
    d = d[d["type"].astype(str).str.upper() == cancer].copy()
    d["patient_id"] = d["bcr_patient_barcode"].map(norm_pid)
    stage = d["ajcc_pathologic_tumor_stage"].map(stage_to_ord)
    if "clinical_stage" in d.columns:
        stage = stage.where(stage.notna(), d["clinical_stage"].map(stage_to_ord))
    d["stage"] = stage
    age = pd.to_numeric(d["age_at_initial_pathologic_diagnosis"], errors="coerce")
    d["age"] = age.where((age > 0) & (age <= 120))
    d["_ok"] = d["stage"].notna().astype(int) + d["age"].notna().astype(int)
    d = d.sort_values(["patient_id", "_ok"], ascending=[True, False]).drop_duplicates("patient_id")
    return d.set_index("patient_id")[["stage", "age"]]


def load_cohort(c, args):
    base = Path(args.base_dir)
    rna, t, e = R.load_data(base / "rna" / f"{c}_processed_rna.csv",
                            base / "cohort" / "survival" / f"{c}_survival.csv",
                            base / "cohort" / f"{c}_cohort_model_cdr.txt")
    feat = pd.read_csv(base / args.features_dir / c / "all_features.csv", usecols=["patient_id"])
    keep = set(feat["patient_id"].map(norm_pid))
    if args.patient_list_dir:
        f = Path(args.patient_list_dir) / f"{c}_final_patients.txt"
        keep &= set(norm_pid(x) for x in f.read_text().split())

    clin = load_clinical(c, args.cdr_file).reindex([norm_pid(x) for x in rna.index])
    ok = (np.array([norm_pid(x) in keep for x in rna.index])
          & clin["stage"].notna().to_numpy() & clin["age"].notna().to_numpy())
    rna, t, e, clin = rna[ok], t[ok], e[ok], clin[ok]
    X_et = load_etime(c, args.base_dir, args.features_dir, rna.index)
    return rna, X_et, t, e, clin["stage"].to_numpy(int), clin["age"].to_numpy(float)


def clinical_design(stage, age, idx, ref):
    """One-hot stage (reference level dropped) and training-standardized age."""
    tr = idx[0]
    S = np.column_stack([(stage == s).astype(float) for s in (1, 2, 3, 4) if s != ref])
    X = np.column_stack([S, (age - age[tr].mean()) / (age[tr].std() + 1e-8)])
    X = X[:, X[tr].std(axis=0) > 1e-8]          # drop stages absent from the training set
    return [X[i].astype(np.float32) for i in idx]


def fit_offset_linear(X, eta, t, e, idx, steps=300, lr=1e-2):
    """Risk = eta + X w with eta fixed; the L2 penalty is chosen by validation C-index."""
    tr, va, te = idx
    X = [np.asarray(x, np.float32) for x in X]
    mu, sd = X[0].mean(0), X[0].std(0) + 1e-8
    Xs = [torch.tensor((x - mu) / sd, dtype=torch.float32) for x in X]
    O = [torch.tensor(np.asarray(x, np.float32)) for x in eta]
    tt = torch.tensor(t[tr], dtype=torch.float32)
    ee = torch.tensor(e[tr], dtype=torch.float32)
    best_c, best_w = -np.inf, None
    for l2 in R.L2_GRID:
        w = torch.zeros(Xs[0].shape[1], requires_grad=True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            loss = R.cox_ph_loss(O[0] + Xs[0] @ w, tt, ee) + l2 * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            c = R.concordance_index(t[va], e[va], (O[1] + Xs[1] @ w).numpy())
        if c > best_c + 1e-9:
            best_c, best_w = c, w.detach().clone()
    with torch.no_grad():
        return [(O[k] + Xs[k] @ best_w).numpy().astype(float) for k in range(3)]


# ---------------------------------------------------------------- experiment
def run_cancer(c, args, hp, dev):
    rna, X_et, t, e, stage, age = load_cohort(c, args)
    vc = pd.Series(stage).value_counts().sort_index()
    ref = int(vc.idxmax())                      # reference stage: the most frequent one
    print(f"[{c}] patients {len(t)} | events {int(e.sum())} | eTime features {X_et.shape[1]}"
          + (" | eTime permuted" if args.permute_etime else ""))

    rows = []
    for run in range(args.n_runs):
        seed = args.seed + run
        tr, va, te = R.stratified_split(e, seed)
        idx = (tr, va, te)

        # RNA: preprocessing and autoencoder pretraining (reconstruction only)
        pre = R.RNAPreprocessor(top_k=args.rna_top_k)
        Xr = [pre.fit_transform(rna.iloc[tr]), pre.transform(rna.iloc[va]), pre.transform(rna.iloc[te])]
        ae = R.train_rna_ae(Xr[0], Xr[1], epochs=args.ae_epochs, batch_size=args.batch_size,
                            patience=args.ae_patience, seed=seed, device=dev)
        Z = [R.encode_numpy(ae, x) for x in Xr]

        # eTime: training-median imputation, near-constant filter, z-score
        X_run = X_et
        if args.permute_etime:
            perm = np.random.RandomState(args.seed + 90000 + run).permutation(len(X_et))
            X_run = pd.DataFrame(X_et.to_numpy()[perm], index=X_et.index, columns=X_et.columns)
        pp = E.Preprocessor(E.GROUPS).fit(X_run.iloc[tr])
        A = [np.asarray(pp.transform(X_run.iloc[i])[1], dtype=np.float32) for i in idx]

        def linear(F):
            pred, _, _ = R.fit_linear_cox_val(F[0], t[tr], e[tr], F[1], t[va], e[va])
            return [pred(f) for f in F]

        C = clinical_design(stage, age, idx, ref)
        risks = {"clin_only": linear(C), "rna_only": linear(Z), "etime_only": linear(A)}
        r_clin = risks["clin_only"]
        risks["clin_rna"] = fit_offset_linear(Z, r_clin, t, e, idx)
        risks["clin_etime"] = fit_offset_linear(A, r_clin, t, e, idx)
        out, _ = train_joint(ae.encoder, Z[0].shape[1], Xr, A, t, e, idx, hp, seed, dev)
        risks["joint_noclin"] = [out[k][2] for k in ("tr", "va", "te")]
        out, _ = train_joint(ae.encoder, Z[0].shape[1], Xr, A, t, e, idx, hp, seed, dev, offset=r_clin)
        risks["evosurv"] = [out[k][2] for k in ("tr", "va", "te")]

        for cfg in CONFIGS:
            r = risks[cfg]
            rows.append(dict(cancer=c, run=run, seed=seed, config=cfg,
                             n_train=len(tr), n_test=len(te), test_events=int(e[te].sum()),
                             c_val=R.concordance_index(t[va], e[va], r[1]),
                             c_test=R.concordance_index(t[te], e[te], r[2])))
        if (run + 1) % 10 == 0:
            m = pd.DataFrame(rows).groupby("config").c_test.mean().reindex(CONFIGS)
            print(f"  run {run + 1}/{args.n_runs}: " + " | ".join(f"{k} {v:.3f}" for k, v in m.items()))

    out_dir = Path(args.out_root) / c
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)


def summarize(args, cancers):
    files = [Path(args.out_root) / c / "runs.csv" for c in cancers]
    df = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    mean = df.pivot_table(index="cancer", columns="config", values="c_test")[CONFIGS]
    sd = df.pivot_table(index="cancer", columns="config", values="c_test", aggfunc="std")[CONFIGS]
    mean.loc["Mean"] = mean.mean()
    text = ("Test C-index, mean over splits\n" + mean.round(3).to_string()
            + "\n\nTest C-index, SD over splits\n" + sd.round(3).to_string() + "\n")
    (Path(args.out_root) / "summary.txt").write_text(text)
    print("\n" + text)


def parse_args():
    base = os.environ.get("EVOSURV_DATA", "/mnt/storage/survival")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cancers", default=",".join(CANCERS))
    p.add_argument("--base_dir", default=base)
    p.add_argument("--features_dir", default="features_v2")
    p.add_argument("--cdr_file", default=f"{base}/cohort/raw/TCGA-CDR-SupplementalTableS1.xlsx")
    p.add_argument("--patient_list_dir", default=None,
                   help="optional; restricts each cancer to <dir>/<CANCER>_final_patients.txt")
    p.add_argument("--permute_etime", action="store_true",
                   help="shuffle eTime rows across patients in each split (null control)")
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rna_top_k", type=int, default=2000)
    p.add_argument("--ae_epochs", type=int, default=300)
    p.add_argument("--ae_patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--out_root", default="results")
    p.add_argument("--summary_only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cancers = [c.strip().upper() for c in args.cancers.split(",") if c.strip()]
    if not args.summary_only:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for c in cancers:
            run_cancer(c, args, HParams(), dev)
    summarize(args, cancers)
