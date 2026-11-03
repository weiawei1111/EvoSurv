import os
import argparse
import gzip
import re
import traceback
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

DATA_ROOT   = Path(os.environ.get("EVOSURV_DATA", "/mnt/storage/survival"))
OUTPUT_ROOT = DATA_ROOT / "features"

NORMAL_CN  = 2
AMP_THRESH = 2
DEL_THRESH = 2

CANCER_LIST = ["BLCA", "BRCA", "HNSC", "LIHC", "LUAD", "LUSC", "OV", "UCEC"]
CANCER_CONFIGS = {c: {"data_dir": DATA_ROOT / c} for c in CANCER_LIST}

PYCLONE_V2_ROOT = DATA_ROOT / "pyclone_v2"
USE_PYCLONE_V2 = False


def pyc_dir_of(cancer: str, data_dir: Path) -> Path:
    return PYCLONE_V2_ROOT / cancer if USE_PYCLONE_V2 else data_dir

SIFT_NO_SCORE_FILL = 1.0

ANOIKIS_GENE_FILE = Path(os.environ.get("EVOSURV_ANOIKIS", str(Path(__file__).resolve().parent.parent / "data" / "GeneCards_anoikis.csv")))
RNA_MATRIX_TMPL   = str(DATA_ROOT / "rna" / "{cancer}_processed_rna.csv")

NONSYN_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Nonstop_Mutation", "Translation_Start_Site",
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(df: pd.DataFrame, path: Path, tag: str) -> None:
    df.to_csv(path, index=False)
    print(f"  [{tag}] saved {path}  shape={df.shape}")


def load_or_skip(path: Path, tag: str):
    if path.exists():
        df = pd.read_csv(path)
        print(f"  [{tag}] exists, skipped: {path}")
        return df
    return None


def check_relabeled(pyc: pd.DataFrame, where: str) -> None:
    if "clonality_rule" not in pyc.columns:
        raise RuntimeError(f"{where}: missing 'clonality_rule'; clonality labels have not been assigned")


def pick_cnv_files(manifest: pd.DataFrame, tag: str) -> pd.DataFrame:
    m = manifest.copy()
    sid = m["submitter_id"].astype(str)
    m["patient_id"] = sid.str[:12]
    st = sid.str[13:15]
    m["_is_primary"] = (st == "01").astype(int)
    m["_vial"] = sid.str[15:16].fillna("")
    m = m.sort_values(["patient_id", "_is_primary", "_vial", "file_name"],
                      ascending=[True, False, True, True])
    dup = m["patient_id"].duplicated(keep=False)
    if dup.any():
        chosen = m[dup].drop_duplicates("patient_id")
        print(f"  [{tag}] {chosen.shape[0]} patients with multiple CNV files, one selected each")
    return m.drop_duplicates("patient_id").drop(columns=["_is_primary", "_vial"])


def _silent_keys(data_dir: Path) -> set:
    p = data_dir / "processed_mutations.csv"
    if not p.exists():
        return set()
    pm = pd.read_csv(p)
    vc = "variant_classification" if "variant_classification" in pm.columns else "Variant_Classification"
    if vc not in pm.columns:
        return set()
    if "mutation_id" not in pm.columns:
        pm["mutation_id"] = (pm["gene"].astype(str) + ":" + pm["chrom"].astype(str) + ":"
                             + pm["pos"].astype(str))
    s = pm[pm[vc] == "Silent"]
    return set(zip(s["patient_id"].astype(str).str[:12], s["mutation_id"].astype(str)))


# ============================================================
# STEP 1: clonal burden and diversity
# ============================================================
def step1_clonal_features(data_dir: Path, out_dir: Path,
                           force: bool = False) -> pd.DataFrame:
    out_path = out_dir / "step1_clonal_features.csv"
    cached = load_or_skip(out_path, "Step1")
    if not force and cached is not None:
        return cached

    pyclone_file = data_dir / "pyclone_results_all.csv"
    df = pd.read_csv(pyclone_file)
    check_relabeled(df, str(pyclone_file))
    silent = _silent_keys(data_dir)
    keys = list(zip(df["sample_id"].astype(str).str[:12], df["mutation_id"].astype(str)))
    df["_silent"] = [k in silent for k in keys]
    if df["_silent"].any():
        print(f"  [Step1] silent mutations excluded from L_E/L_L: {int(df['_silent'].sum())}")

    records = []
    for sample_id, grp in df.groupby("sample_id"):
        ns = ~grp["_silent"]
        is_clonal    = (grp["clonality_pyclone"] == "clonal") & ns
        is_subclonal = (grp["clonality_pyclone"] == "subclonal") & ns
        L_E = int(is_clonal.sum())
        L_L = int(is_subclonal.sum())
        total = L_E + L_L
        SubFrac = L_L / total if total > 0 else np.nan

        cluster_label = grp.groupby("cluster_id")["clonality_pyclone"].agg(
            lambda s: s.mode().iat[0])
        K = int((cluster_label == "subclonal").sum())

        cluster_counts = grp.groupby("cluster_id").size().values
        probs = cluster_counts / cluster_counts.sum()
        H = float(entropy(probs, base=2))

        records.append({
            "sample_id": sample_id,
            "L_E":       round(float(np.log1p(L_E)), 4),
            "L_L":       round(float(np.log1p(L_L)), 4),
            "SubFrac":   round(SubFrac, 4),
            "K":         K,
            "H":         round(H, 4),
        })

    result = pd.DataFrame(records)
    save(result, out_path, "Step1")
    return result


# ============================================================
# STEP 2: copy-number burden
# ============================================================
def _cnv_single(filepath: Path) -> dict:
    df = pd.read_csv(filepath, sep="\t").dropna(subset=["copy_number"])
    total = len(df)
    if total == 0:
        return dict(FGA=np.nan, AmpFrac=np.nan, DelFrac=np.nan, Aneu=np.nan)
    cn = df["copy_number"]
    amp  = cn > AMP_THRESH
    del_ = cn < DEL_THRESH
    return {
        "FGA":     round(float((amp | del_).sum() / total), 4),
        "AmpFrac": round(float(amp.sum() / total), 4),
        "DelFrac": round(float(del_.sum() / total), 4),
        "Aneu":    round(float((cn - NORMAL_CN).abs().mean()), 4),
    }


def step2_cnv_features(data_dir: Path, out_dir: Path,
                       force: bool = False) -> pd.DataFrame:
    out_path = out_dir / "step2_cnv_features.csv"
    cached = load_or_skip(out_path, "Step2")
    if not force and cached is not None:
        return cached

    manifest = pick_cnv_files(pd.read_csv(data_dir / "cnv_manifest.csv"), "Step2")
    cnv_dir = data_dir / "cnv"

    records, missing = [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="  Step2 CNV", leave=False):
        fp = cnv_dir / row["file_name"]
        if not fp.exists():
            missing.append(row["file_name"])
            continue
        feats = _cnv_single(fp)
        feats["patient_id"] = row["patient_id"]
        records.append(feats)

    if missing:
        print(f"  [Step2] missing CNV files: {len(missing)}")

    result = pd.DataFrame(records)[["patient_id", "FGA", "AmpFrac", "DelFrac", "Aneu"]]
    save(result, out_path, "Step2")
    return result


# ============================================================
# STEP 6: mutation spectra, functional classes, deleteriousness
# ============================================================
SUB_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
VARIANT_GROUPS = ["missense", "nonsense", "splice", "other"]

KEEP_VARIANT_CLASSES = [
    "Missense_Mutation", "Nonsense_Mutation",
    "Splice_Site", "Splice_Region",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
    "Silent",
]


def _raw_sub(ref, alt):
    ref, alt = str(ref).upper(), str(alt).upper()
    if len(ref) != 1 or len(alt) != 1:
        return None
    if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        return None
    if ref in "AG":
        ref, alt = _COMP[ref], _COMP[alt]
    return f"{ref}>{alt}"


def _parse_sift(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"\(([\d\.eE+-]+)\)", str(x))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    try:
        return float(str(x))
    except ValueError:
        return np.nan


def _is_deleterious(x):
    if pd.isna(x):
        return False
    s = str(x).lower()
    if "deleterious" in s:
        return True
    sc = _parse_sift(x)
    return not np.isnan(sc) and sc <= 0.05


def _classify_variant(vc):
    vc = str(vc)
    if vc == "Missense_Mutation":  return "missense"
    if vc == "Nonsense_Mutation":  return "nonsense"
    if "Splice" in vc:             return "splice"
    if vc == "Silent":             return None
    return "other"


def _load_maf_from_gz(maf_dir: Path) -> pd.DataFrame:
    manifest_path = maf_dir / "manifest_full.csv"
    if not manifest_path.exists():
        manifest_path = maf_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"  [MutBio] manifest not found: {maf_dir}")
        return pd.DataFrame()

    manifest = pd.read_csv(manifest_path)
    if "patient_id" not in manifest.columns:
        if "submitter_id" in manifest.columns:
            manifest["patient_id"] = manifest["submitter_id"].astype(str).str[:12]
        else:
            print("  [MutBio] manifest has no patient_id column")
            return pd.DataFrame()

    dfs = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc="  [MutBio] reading MAF", leave=False):
        patient_id = row["patient_id"]
        if not patient_id or patient_id == "nan":
            continue

        fid = row.get("file_id", "")
        fid = "" if pd.isna(fid) else str(fid)
        candidates = [
            maf_dir / fid / row["file_name"],
            maf_dir / row["file_name"],
        ]
        maf_path = next((p for p in candidates if p.exists()), None)
        if maf_path is None:
            continue

        try:
            opener = gzip.open if str(maf_path).endswith(".gz") else open
            with opener(maf_path, "rt") as f:
                lines = [l for l in f if not l.startswith("#")]
            df = pd.read_csv(StringIO("".join(lines)), sep="\t", low_memory=False)
        except Exception:
            continue

        if "Variant_Classification" in df.columns:
            df = df[df["Variant_Classification"].isin(KEEP_VARIANT_CLASSES)]
        if "Variant_Type" in df.columns:
            df = df[df["Variant_Type"] == "SNP"]

        need_cols = ["Hugo_Symbol", "Chromosome", "Start_Position",
                     "Reference_Allele", "Tumor_Seq_Allele2",
                     "Variant_Classification"]
        sift_col = next((c for c in df.columns
                         if c.upper() in ["SIFT", "SIFT_SCORE"]), None)
        if sift_col:
            need_cols.append(sift_col)

        keep = [c for c in need_cols if c in df.columns]
        df = df[keep].copy()
        df["patient_id"] = patient_id

        chrom = df["Chromosome"].astype(str).apply(
            lambda x: x if x.startswith("chr") else "chr" + x)
        df["mutation_id"] = (
            df["Hugo_Symbol"].astype(str) + ":" +
            chrom + ":" +
            df["Start_Position"].astype(str)
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def step_mutbio(data_dir: Path, out_dir: Path,
                force: bool = False, pyc_dir: Path = None) -> pd.DataFrame:
    pyc_dir = pyc_dir or data_dir
    out_path = out_dir / "mutbio_features.csv"
    cached = load_or_skip(out_path, "MutBio")
    if not force and cached is not None:
        return cached

    pyc = pd.read_csv(pyc_dir / "pyclone_results_all.csv")
    check_relabeled(pyc, str(pyc_dir / "pyclone_results_all.csv"))
    pyc["patient_id"]  = pyc["sample_id"].astype(str).str[:12]
    pyc["mutation_id"] = pyc["mutation_id"].astype(str)

    maf = _load_maf_from_gz(data_dir / "maf")
    if maf.empty:
        print("  [MutBio] empty MAF, skipped")
        return pd.DataFrame()
    print(f"  [MutBio] MAF records: {len(maf)}")

    sift_col = next((c for c in maf.columns
                     if c.upper() in ["SIFT", "SIFT_SCORE"]), None)
    if sift_col and sift_col != "SIFT":
        maf = maf.rename(columns={sift_col: "SIFT"})
    if "SIFT" not in maf.columns:
        maf["SIFT"] = np.nan

    maf_small = maf[["patient_id", "mutation_id", "Reference_Allele",
                      "Tumor_Seq_Allele2", "Variant_Classification",
                      "SIFT"]].drop_duplicates(subset=["patient_id", "mutation_id"])

    merged = pyc.merge(maf_small, on=["patient_id", "mutation_id"], how="inner")
    print(f"  [MutBio] merged: {len(merged)} mutations, {merged['patient_id'].nunique()} patients")

    merged["substitution"]  = [_raw_sub(r, a) for r, a in
                                zip(merged["Reference_Allele"], merged["Tumor_Seq_Allele2"])]
    merged["variant_group"] = merged["Variant_Classification"].map(_classify_variant)
    merged["sift_score"]    = merged["SIFT"].map(_parse_sift)
    merged["sift_del"]      = merged["SIFT"].map(_is_deleterious)

    sift_cols = ["mean_clonal_SIFT", "mean_subclonal_SIFT"]
    feat_cols = (
        [f"clonal_{s}"    for s in SUB_TYPES] +
        [f"subclonal_{s}" for s in SUB_TYPES] +
        [f"clonal_{v}"    for v in VARIANT_GROUPS] +
        [f"subclonal_{v}" for v in VARIANT_GROUPS] +
        ["clonal_deleterious_count", "subclonal_deleterious_count"] +
        sift_cols
    )

    records = []
    for patient_id, grp in merged.groupby("patient_id"):
        cl = grp[grp["clonality_pyclone"] == "clonal"]
        sc = grp[grp["clonality_pyclone"] == "subclonal"]
        rec = {"patient_id": patient_id}
        for s in SUB_TYPES:
            rec[f"clonal_{s}"]    = int((cl["substitution"] == s).sum())
            rec[f"subclonal_{s}"] = int((sc["substitution"] == s).sum())
        for v in VARIANT_GROUPS:
            rec[f"clonal_{v}"]    = int((cl["variant_group"] == v).sum())
            rec[f"subclonal_{v}"] = int((sc["variant_group"] == v).sum())
        rec["clonal_deleterious_count"]    = int(cl["sift_del"].sum())
        rec["subclonal_deleterious_count"] = int(sc["sift_del"].sum())
        rec["mean_clonal_SIFT"]    = float(cl["sift_score"].dropna().mean()) \
                                     if cl["sift_score"].notna().any() else np.nan
        rec["mean_subclonal_SIFT"] = float(sc["sift_score"].dropna().mean()) \
                                     if sc["sift_score"].notna().any() else np.nan
        records.append(rec)

    result = pd.DataFrame(records)
    for c in feat_cols:
        if c not in result.columns:
            result[c] = np.nan if c in sift_cols else 0
    result[sift_cols] = result[sift_cols].fillna(SIFT_NO_SCORE_FILL)
    count_cols = [c for c in feat_cols if c not in sift_cols]
    result[count_cols] = np.log1p(result[count_cols].fillna(0)).round(4)
    result = result[["patient_id"] + feat_cols]
    save(result, out_path, "MutBio")
    return result


# ============================================================
# STEP 7: anoikis-related alterations
# ============================================================
def _anoikis_gene_set(cancer: str) -> list:
    g = pd.read_csv(ANOIKIS_GENE_FILE)
    anoikis = set(g["Symbol"].astype(str).str.upper())
    rna_path = Path(RNA_MATRIX_TMPL.format(cancer=cancer))
    rna_genes = set(pd.read_csv(rna_path, nrows=0).columns.astype(str).str.upper())
    genes = sorted(anoikis & rna_genes)
    print(f"  [Step7] anoikis list {len(anoikis)} | expression matrix {len(rna_genes)} | used {len(genes)}")
    return genes


def step7_anoikis(data_dir: Path, out_dir: Path, cancer: str,
                  force: bool = False, pyc_dir: Path = None) -> pd.DataFrame:
    pyc_dir = pyc_dir or data_dir
    out_path = out_dir / "step7_anoikis_features.csv"
    cached = load_or_skip(out_path, "Step7")
    if not force and cached is not None:
        return cached

    genes = _anoikis_gene_set(cancer)
    gset = set(genes)
    pd.DataFrame({"gene": genes}).to_csv(out_dir / "anoikis_genes_used.csv", index=False)

    pm = pd.read_csv(pyc_dir / "processed_mutations.csv")
    vc_col = "variant_classification" if "variant_classification" in pm.columns \
             else "Variant_Classification"
    pm["patient_id"] = pm["patient_id"].astype(str).str[:12]
    pm["gene_u"] = pm["gene"].astype(str).str.upper()
    known = pm["clonality_pyclone"].isin(["clonal", "subclonal"])
    nonsyn = pm[vc_col].isin(NONSYN_CLASSES)
    ns = pm[known & nonsyn]
    has_pyclone = set(pm.loc[known, "patient_id"])

    rows = []
    for pid in sorted(set(pm["patient_id"])):
        if pid not in has_pyclone:
            rows.append(dict(patient_id=pid))
            continue
        g = ns[ns["patient_id"] == pid]
        a = g[g["gene_u"].isin(gset)]
        a_e = int((a["clonality_pyclone"] == "clonal").sum())
        a_l = int((a["clonality_pyclone"] == "subclonal").sum())
        n_all = len(g)
        rows.append(dict(
            patient_id=pid,
            A_frac=round((a_e + a_l) / n_all, 4) if n_all > 0 else 0.0,
            A_SubFrac=round(a_l / (a_e + a_l), 4) if (a_e + a_l) > 0 else np.nan,
            has_anoikis_mut=int((a_e + a_l) > 0),
        ))
    mut = pd.DataFrame(rows)

    manifest = pick_cnv_files(pd.read_csv(data_dir / "cnv_manifest.csv"), "Step7")
    cnv_rows, missing = [], 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="  Step7 CNV", leave=False):
        fp = data_dir / "cnv" / r["file_name"]
        if not fp.exists():
            missing += 1
            continue
        cn = pd.read_csv(fp, sep="\t").dropna(subset=["copy_number"])
        cn = cn[cn["gene_name"].astype(str).str.upper().isin(gset)]
        n = len(cn)
        cnv_rows.append(dict(
            patient_id=r["patient_id"],
            A_amp=round(float((cn["copy_number"] > AMP_THRESH).sum() / n), 4) if n else np.nan,
            A_del=round(float((cn["copy_number"] < DEL_THRESH).sum() / n), 4) if n else np.nan,
        ))
    if missing:
        print(f"  [Step7] missing CNV files: {missing}")
    cnv = pd.DataFrame(cnv_rows, columns=["patient_id", "A_amp", "A_del"])

    result = mut.merge(cnv, on="patient_id", how="outer")
    cols = ["patient_id", "A_frac", "A_SubFrac", "has_anoikis_mut", "A_amp", "A_del"]
    for c in cols:
        if c not in result.columns:
            result[c] = np.nan
    result = result[cols]
    print(f"  [Step7] patients with anoikis mutations {result['has_anoikis_mut'].eq(1).mean():.1%}")
    save(result, out_path, "Step7")
    return result


def merge_all_features(dfs: list, out_dir: Path) -> pd.DataFrame:
    normed = []
    for df in dfs:
        if df is None or df.empty:
            continue
        if "sample_id" in df.columns and "patient_id" not in df.columns:
            df = df.rename(columns={"sample_id": "patient_id"})
            df["patient_id"] = df["patient_id"].str[:12]
        normed.append(df)

    if not normed:
        print("  [MERGE] no features to merge")
        return pd.DataFrame()

    base = normed[0]
    for other in normed[1:]:
        overlap = [c for c in other.columns if c != "patient_id" and c in base.columns]
        if overlap:
            other = other.drop(columns=overlap)
        base = base.merge(other, on="patient_id", how="outer")

    dup = base["patient_id"].duplicated()
    if dup.any():
        raise RuntimeError(f"duplicated patients after merge: {sorted(base.loc[dup, 'patient_id'].unique())[:10]}")

    save(base, out_dir / "all_features_full.csv", "MERGE-full")

    if "L_E" in base.columns:
        keep = base["L_E"].notna()
        dropped = base.loc[~keep, "patient_id"]
        if len(dropped):
            print(f"  [MERGE] excluded {len(dropped)} patients without PyClone results; kept {int(keep.sum())}")
            pd.DataFrame({"patient_id": dropped}).to_csv(out_dir / "excluded_no_pyclone.csv", index=False)
        base = base[keep].reset_index(drop=True)
    save(base, out_dir / "all_features.csv", "MERGE")
    return base


def run_cancer(cancer: str, steps: list, force: bool) -> None:
    print(f"\n{'=' * 60}\n  {cancer}\n{'=' * 60}")

    data_dir = Path(CANCER_CONFIGS[cancer]["data_dir"])
    out_dir  = ensure_dir(OUTPUT_ROOT / cancer)
    if not data_dir.exists():
        print(f"  data directory not found: {data_dir}, skipped")
        return

    pyc_dir = pyc_dir_of(cancer, data_dir)
    jobs = [
        (1, "step1",   lambda: step1_clonal_features(pyc_dir, out_dir, force)),
        (2, "step2",   lambda: step2_cnv_features(data_dir, out_dir, force)),
        (6, "mutbio",  lambda: step_mutbio(data_dir, out_dir, force, pyc_dir=pyc_dir)),
        (7, "anoikis", lambda: step7_anoikis(data_dir, out_dir, cancer, force, pyc_dir=pyc_dir)),
    ]
    results = {}
    for step, name, fn in jobs:
        if step in steps:
            try:
                results[name] = fn()
            except Exception as e:
                print(f"  {name} failed: {e}")
                traceback.print_exc()

    if "merge" in steps:
        merge_all_features([v for v in results.values()
                            if v is not None and not v.empty], out_dir)
    print(f"\n  {cancer} done: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="eTime feature extraction")
    p.add_argument("--cancer", nargs="+", default=None)
    p.add_argument("--steps", nargs="+", default=None,
                   help="1 clonal burden, 2 CNV, 6 spectra/classes/SIFT, 7 anoikis, merge")
    p.add_argument("--force", action="store_true", help="recompute existing outputs")
    p.add_argument("--pyclone_v2", action="store_true",
                   help="use PyClone results in pyclone_v2/ and write to features_v2/")
    return p.parse_args()


def main():
    global USE_PYCLONE_V2, OUTPUT_ROOT
    args = parse_args()
    if args.pyclone_v2:
        USE_PYCLONE_V2 = True
        OUTPUT_ROOT = DATA_ROOT / "features_v2"
    cancers = args.cancer if args.cancer else CANCER_LIST
    steps = ([int(s) if s.isdigit() else s for s in args.steps] if args.steps
             else [1, 2, 6, 7, "merge"])

    print(f"cancers: {cancers}\nsteps: {steps}\nforce: {args.force}\noutput: {OUTPUT_ROOT}")
    for cancer in cancers:
        if cancer not in CANCER_CONFIGS:
            print(f"unknown cancer: {cancer}, skipped")
            continue
        try:
            run_cancer(cancer, steps, args.force)
        except Exception as e:
            print(f"\n{cancer} failed: {e}")
            traceback.print_exc()
    print(f"\nall done: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()