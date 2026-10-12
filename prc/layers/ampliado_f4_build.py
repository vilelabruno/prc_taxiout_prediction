import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, json
import numpy as np, pandas as pd
from prc.paths import FEATURES, OOF, RANKING, SUBMITTING
from own_flight_features import TRAINING_ALL
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
from adsb_corrector_champion import frame_base
from adsb_corrector_callsign import add_blocks, obs_mask
import pos_features as pf
import ampliado_f4_corrector as C

S4 = ['EDDF', 'EDDM', 'EHAM', 'LSZH']
PARAMS = dict(num_leaves=128, min_child_samples=20, n_estimators=800, learning_rate=0.03)
LEGACY_OFF = ['EGLL', 'LEMD', 'LIRF', 'LFPG']
B_SHRINK = {'EGLL': (0.5, (1, 7)), 'LEMD': (0.5, (7,))}
B_BLEND = {'LEBL': (0.5, (1, 7))}
NB_MONTHS = {1: [12, 2], 7: [6, 8]}; W_EXT = 0.5


def _i64(d):
    d = d.copy(); d['MVT_ID_mvt'] = d.MVT_ID_mvt.astype('int64'); return d


def _add_new(d, work, yr):
    idx = d.index
    rt = _i64(pd.read_parquet(work / f'rt_{yr}.parquet')); lg = _i64(pd.read_parquet(work / f'leg_rt_{yr}.parquet'))
    d = _i64(d).merge(rt, on='MVT_ID_mvt', how='left').merge(lg[['MVT_ID_mvt'] + C.LGF + C.LRT], on='MVT_ID_mvt', how='left'); d.index = idx
    return C.derived(d)


def _legacy_off(d, f):
    for a in LEGACY_OFF:
        m = (f.ADEP_mvt == a).to_numpy()
        for c in C.ADSB + C.POS_FULL:
            d.loc[m, c] = np.nan
    return d


def build(pts_dir, work_dir, out_path, pos2_path, v36_path, params=PARAMS, neighbors=True, label='final'):
    pts_dir, work = Path(pts_dir), Path(work_dir)
    cols = ['MVT_ID_mvt', 'ADES_mvt', 'STAND_mvt']
    a25 = pd.read_parquet(pts_dir / 'e2_arr_2025.parquet').merge(pd.read_parquet(TRAINING_ALL, columns=cols), on='MVT_ID_mvt')
    a26 = pd.read_parquet(pts_dir / f'e2_arr_{label}.parquet').merge(pd.read_parquet(RANKING, columns=cols), on='MVT_ID_mvt')
    t25, t26 = pf.stand_tables(a25, a26, 'cons')
    h = pd.read_parquet(HOLDOUT); idx = h.index
    t = pd.read_parquet(TRAINING_ALL, columns=['MVT_ID_mvt', 'CALLSIGN_flt']); h = _i64(h); t = _i64(t)
    h = h.merge(t, on='MVT_ID_mvt', how='left'); h.index = idx
    own = _i64(pd.read_parquet(FEATURES / 'adsb_own_cs_2025.parquet')); h = h.merge(own, on='MVT_ID_mvt', how='left'); h.index = idx; h['adsb_cov'] = h.adsb_cov.fillna(0)
    h = pf.add_pos_features(h, pd.read_parquet(pts_dir / 'e2_pts_2025.parquet'), t25)
    cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
    h = frame_base(h); h = add_blocks(h); y = h.TAXITIME_SEC_mvt.astype(float).to_numpy(); mk = obs_mask(h); h = _add_new(h, work, '2025')
    rk = pd.read_parquet(RANKING); rk = _i64(rk[rk.PHASE_mvt == 'DEP'].reset_index(drop=True))
    o = _i64(pd.read_parquet(FEATURES / f'adsb_own_cs_{label}.parquet')); rk = rk.merge(o, on='MVT_ID_mvt', how='left'); rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk = pf.add_pos_features(rk, pd.read_parquet(pts_dir / f'e2_pts_{label}.parquet'), t26)
    v36 = _i64(pd.read_parquet(v36_path))
    rk['pred'] = rk.MVT_ID_mvt.map(v36.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt).astype(float)
    rk = frame_base(rk, cats); rk = add_blocks(rk); rk = _add_new(rk, work, label)
    drift = pd.read_csv(work / 'drift_all.csv'); new_keep = C.keep_stable(drift)
    pk = json.load(open(work / 'pos_keep.json')); pk.pop('LEBL', None)
    dtr = _legacy_off(C.prep(h, pk, new_keep), h); dap = _legacy_off(C.prep(rk, pk, new_keep), rk)
    F = C.feature_list()
    mr = obs_mask(rk); cov = (rk.adsb_cov == 1).to_numpy(); apt = rk.ADEP_mvt.to_numpy(); mes = rk.mes.to_numpy(); covh = (h.adsb_cov == 1).to_numpy(); apth = h.ADEP_mvt.to_numpy()
    cA = np.zeros(len(rk)); cB = np.zeros(len(rk)); elA = mr & cov & np.isin(apt, S4) & np.isin(mes, (1, 7))
    elB = np.zeros(len(rk), bool); shr = np.zeros(len(rk)); blend = np.zeros(len(rk), bool)
    for a, (s, ms) in B_SHRINK.items():
        for m in ms:
            e = mr & cov & (apt == a) & (mes == m); elB |= e; shr[e] = s
    for a, (s, ms) in B_BLEND.items():
        for m in ms:
            e = mr & cov & (apt == a) & (mes == m); elB |= e; shr[e] = s; blend |= e
    for m in (1, 7):
        am = elB & (mes == m)
        cB[am] = C.fit_apply(dtr, y, F, (h.mes == m).to_numpy() & mk, dap, am, params=params)[am]
        am = elA & (mes == m); trA = (h.mes == m).to_numpy() & mk & covh & np.isin(apth, S4)
        if neighbors:
            parts = [pd.read_parquet(work / f'nbfeat_m{n}.parquet') for n in NB_MONTHS[m]]
            ext = C.prep(pd.concat(parts, ignore_index=True), pk, new_keep)
            ext, dropped = C.drift_zero_neighbors(ext, dap[(dap.mes == m) & (dap.adsb_cov == 1) & mr], F, S4)
            me = obs_mask(ext) & (ext.adsb_cov == 1).to_numpy() & ext.ADEP_mvt.isin(S4).to_numpy()
            print('mes', m, 'vizinhos', NB_MONTHS[m], 'linhas', int(me.sum()), 'features zeradas por deriva', {a: len(b) for a, b in dropped.items()}, flush=True)
            ext = ext[me]; cA[am] = C.fit_apply_ext(dtr, y, F, trA, ext, ext.TAXITIME_SEC_mvt.astype(float).to_numpy(), W_EXT, dap, am, params=params)[am]
        else:
            cA[am] = C.fit_apply(dtr, y, F, trA, dap, am, params=params)[am]
    p2 = _i64(pd.read_parquet(pos2_path)); tpl = pd.read_parquet(SUBMITTING)
    assert (p2.MVT_ID_mvt.values == tpl.MVT_ID_mvt.values).all()
    rr = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt, 'pred': rk.pred, 'cA': cA, 'cB': cB, 'elA': elA, 'elB': elB, 'shr': shr, 'bl': blend}); mm = p2.merge(rr, on='MVT_ID_mvt', how='left'); assert len(mm) == len(p2)
    cur = mm.TAXITIME_SEC_mvt.to_numpy(float); pr = mm.pred.to_numpy(float); eA = mm.elA.fillna(False).to_numpy(bool); eB = mm.elB.fillna(False).to_numpy(bool); bl = mm.bl.fillna(False).to_numpy(bool)
    new = cur.copy()
    new = np.where(eA, np.maximum(pr + mm.cA.to_numpy(), 0.0), new)
    new = np.where(eB & ~bl, np.maximum(pr + mm.shr.to_numpy() * mm.cB.to_numpy(), 0.0), new)
    new = np.where(eB & bl, (1 - mm.shr.to_numpy()) * cur + mm.shr.to_numpy() * np.maximum(pr + mm.cB.to_numpy(), 0.0), new)
    out = tpl[['MVT_ID_mvt']].copy(); out['TAXITIME_SEC_mvt'] = new.astype(tpl.TAXITIME_SEC_mvt.dtype)
    assert out.TAXITIME_SEC_mvt.notna().all() and (out.TAXITIME_SEC_mvt >= 0).all() and out.MVT_ID_mvt.is_unique
    out.to_parquet(out_path, index=False); rr.to_parquet(Path(out_path).with_suffix('.corr.parquet'), index=False)
    print('linhas alteradas', int((np.abs(out.TAXITIME_SEC_mvt.to_numpy() - cur) > 1e-9).sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pts-dir', default=str(FEATURES)); ap.add_argument('--work-dir', default=str(FEATURES)); ap.add_argument('--out', required=True)
    ap.add_argument('--pos2', required=True); ap.add_argument('--v36', required=True); ap.add_argument('--label', default='final'); ap.add_argument('--no-neighbors', action='store_true')
    a = ap.parse_args(); build(a.pts_dir, a.work_dir, a.out, a.pos2, a.v36, neighbors=not a.no_neighbors, label=a.label)
