import numpy as np

SUB6 = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
CLS4 = ["missense", "nonsense", "splice", "other"]

GROUPS = {
    "clonal_burden":      ["L_E", "L_L", "SubFrac", "K", "H"],
    "cnv_burden":         ["FGA", "AmpFrac", "DelFrac", "Aneu"],
    "clonal_spectrum":    [f"clonal_{s}" for s in SUB6],
    "subclonal_spectrum": [f"subclonal_{s}" for s in SUB6],
    "clonal_class":       [f"clonal_{c}" for c in CLS4],
    "subclonal_class":    [f"subclonal_{c}" for c in CLS4],
    "deleteriousness":    ["clonal_deleterious_count", "subclonal_deleterious_count",
                           "mean_clonal_SIFT", "mean_subclonal_SIFT"],
    "anoikis":            ["A_frac", "A_SubFrac", "has_anoikis_mut", "A_amp", "A_del"],
}


def to_props(X):

    X = X.copy()
    cnt = lambda cols: np.expm1(X[cols])
    for per in ("clonal", "subclonal"):
        sp = cnt([f"{per}_{s}" for s in SUB6])
        tot = sp.sum(1).replace(0, np.nan)
        for s in SUB6:
            X[f"{per}_{s}"] = sp[f"{per}_{s}"] / tot
        cl = cnt([f"{per}_{c}" for c in CLS4])
        ctot = cl.sum(1).replace(0, np.nan)
        for c in CLS4:
            X[f"{per}_{c}"] = cl[f"{per}_{c}"] / ctot
        X[f"{per}_deleterious_count"] = np.expm1(X[f"{per}_deleterious_count"]) / tot
    return X


class Preprocessor:

    def __init__(self, groups, const_frac=0.99):
        self.groups, self.const_frac = groups, const_frac

    def fit(self, X):
        self.median_ = X.median()
        Xf = X.fillna(self.median_)
        self.all_nan_ = [col for col in X.columns if X[col].isna().all()]
        keep = []
        for col in X.columns:
            if col in self.all_nan_:
                continue
            v = Xf[col]
            vc = v.value_counts(normalize=True)
            top = vc.iloc[0] if len(vc) else 1.0
            if v.std() > 1e-8 and top < self.const_frac:
                keep.append(col)
        self.cols_ = keep
        self.mean_, self.std_ = Xf[keep].mean(), Xf[keep].std() + 1e-8
        self.groups_ = {g: [f for f in fs if f in keep] for g, fs in self.groups.items()}
        self.groups_ = {g: fs for g, fs in self.groups_.items() if fs}
        return self

    def transform(self, X):
        Z = ((X.fillna(self.median_)[self.cols_] - self.mean_) / self.std_).astype(np.float32)
        return {g: Z[fs].values for g, fs in self.groups_.items()}, Z.values
