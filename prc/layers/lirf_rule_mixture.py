import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from prc.paths import DATA_ROOT, MODELS, OOF, RANKING
TRAINING_ALL = DATA_ROOT / 'processed' / 'training_all_2025.parquet'
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
FEATS = ['te', 'lm', 'b2', 'b4', 'b6', 'sp4', 'noite']


def cell_frame(dep: pd.DataFrame) -> pd.DataFrame:
    c = dep.copy(); c['m'] = (c.MVT_TIME_UTC_mvt - c.SCHED_TIME_UTC_mvt).dt.total_seconds()
    c = c[(c.ADEP_mvt == 'LIRF') & c.AOBT_3_flt.isna() & (c.m >= 3600) & (c.m < 14 * 3600)].copy()
    mh = c.m / 3600
    c['pref'] = c.FLIGHT_mvt.astype(str).str[:3]; c['sp4'] = (c.STAND_mvt.astype(str).str[:1] == '4').astype(int)
    c['noite'] = c.MVT_TIME_UTC_mvt.dt.hour.le(5).astype(int); c['lm'] = np.log(c.m.clip(lower=1))
    c['b2'] = ((mh >= 2) & (mh < 4)).astype(int); c['b4'] = ((mh >= 4) & (mh < 6)).astype(int); c['b6'] = (mh >= 6).astype(int)
    c['mon'] = c.SCHED_TIME_UTC_mvt.dt.month
    return c


def te_map(c, rows, k=8.0):
    g = c.eco[rows].mean(); s = c[rows].groupby('pref').eco.agg(['sum', 'count']); return ((s['sum'] + k * g) / (s['count'] + k)), g


def design(c, te_col):
    X = c[['lm', 'b2', 'b4', 'b6', 'sp4', 'noite']].copy(); X['te'] = np.log(te_col / (1 - te_col)); return X[FEATS]


def fit_predict(c, train_mask, test_c, save=None):
    te = pd.Series(np.nan, index=c.index)
    for j in c.mon[train_mask].unique():
        mp, g = te_map(c, train_mask & (c.mon != j)); sel = train_mask & (c.mon == j); te[sel] = c.pref[sel].map(mp).fillna(g)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(design(c, te)[train_mask], c.eco[train_mask])
    if save is not None:
        import joblib
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, save)
    mp, g = te_map(c, train_mask); tt = test_c.pref.map(mp).fillna(g)
    return pd.Series(lr.predict_proba(design(test_c, tt))[:, 1], index=test_c.index)


def load_train():
    t = pd.read_parquet(TRAINING_ALL)
    dep = t[(t.PHASE_mvt == 'DEP') & t.TAXITIME_SEC_mvt.notna()]
    c = cell_frame(dep); y = c.TAXITIME_SEC_mvt.astype(float); c['eco'] = ((y - c.m).abs() <= 120).astype(int); c['y'] = y
    return c


def mu_of(c, mask): return c[mask & (c.eco == 0) & (c.y <= 3600)].y.mean()


def validate(months=(1, 7)):
    c = load_train(); h = pd.read_parquet(HOLDOUT); idx = h.index.intersection(c.index)
    H = h.loc[idx].copy(); H['m'] = c.m.loc[idx]; H['mon'] = c.mon.loc[idx]; H['y'] = H.TAXITIME_SEC_mvt.astype(float); H['eco'] = c.eco.loc[idx]
    H['p'] = np.nan
    for mth in months:
        tr = (c.mon != mth); te_rows = idx[c.loc[idx, 'mon'] == mth]; H.loc[te_rows, 'p'] = fit_predict(c, tr, c.loc[te_rows])
        H.loc[te_rows, 'mu'] = mu_of(c, tr)
    H['mix'] = H.p * H.m + (1 - H.p) * H.mu
    e0 = (h.pred - h.TAXITIME_SEC_mvt.astype(float)) ** 2; N = len(h); S = e0.sum(); mons = h.MVT_TIME_UTC_mvt.dt.month; days = h.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    print(f'campeão {np.sqrt(S/N):.3f} | célula no holdout {len(H)} linhas | AUC jan {roc_auc_score(H.eco[H.mon==1], H.p[H.mon==1]):.3f} jul {roc_auc_score(H.eco[H.mon==7], H.p[H.mon==7]):.3f}')
    for w in (0.5, 0.75, 1.0):
        v = w * H.mix + (1 - w) * H.pred; d = (v - H.y) ** 2 - (H.pred - H.y) ** 2
        m_ = {k: np.sqrt((e0[mons == k].sum() + d[H.mon == k].sum()) / (mons == k).sum()) for k in months}
        g = d.groupby(days.loc[H.index]).sum().to_numpy(); rs = np.random.default_rng(0); bs = [np.sqrt((S + g[rs.integers(0, len(g), len(g))].sum()) / N) for _ in range(3000)]
        print(f'w={w:4}: RMSE {np.sqrt((S + d.sum())/N):.2f} (' + ', '.join(f'mes {k} {m_[k]:.1f}' for k in months) + f') IC90 por dia [{np.percentile(bs,5):.2f}, {np.percentile(bs,95):.2f}]')


def build(base_path, v36_path, out_path, w, rank_path=None):
    c = load_train(); allm = pd.Series(True, index=c.index)
    rk = pd.read_parquet(rank_path or RANKING); rk = rk[rk.PHASE_mvt == 'DEP']; rc = cell_frame(rk); rc['MVT_ID_mvt'] = rc.MVT_ID_mvt.astype('int64')
    p = fit_predict(c, allm, rc, save=MODELS / 'layers' / 'lirf_rule.joblib'); mu = mu_of(c, allm); mix = pd.Series((p * rc.m + (1 - p) * mu).to_numpy(), index=rc.MVT_ID_mvt.to_numpy())
    base = pd.read_parquet(base_path); v36 = pd.read_parquet(v36_path).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    out = base.copy(); k = out.MVT_ID_mvt.astype('int64').isin(mix.index)
    ids = out.loc[k, 'MVT_ID_mvt'].astype('int64')
    out.loc[k, 'TAXITIME_SEC_mvt'] = (w * mix.reindex(ids).to_numpy() + (1 - w) * v36.reindex(ids.astype(float)).to_numpy()).clip(min=0)
    assert out.TAXITIME_SEC_mvt.notna().all() and out.MVT_ID_mvt.is_unique and len(out) == len(base)
    out.to_parquet(out_path, index=False); print(f'v42: {int(k.sum())} linhas alteradas | w={w} | mu {mu:.0f} s | p médio {p.mean():.2f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default=None); ap.add_argument('--base', default=None); ap.add_argument('--v36', default=None)
    ap.add_argument('--w', type=float, default=0.75); ap.add_argument('--rank', default=None)
    a = ap.parse_args()
    if a.build:
        build(a.base, a.v36, a.build, a.w, rank_path=a.rank)
    else: validate()
