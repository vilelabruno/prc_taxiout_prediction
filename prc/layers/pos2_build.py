import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adsb'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse, tempfile, time
import numpy as np, pandas as pd
from prc.paths import DATA_ROOT, FEATURES, MODELS, OOF, RANKING, SUBMITTING
from own_flight_features import TRAINING_ALL
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
from adsb_corrector_champion import frame_base
from adsb_corrector_callsign import add_blocks, obs_mask
import ltfm_delay_state_corrector as L
import adsb_position_corrector as PC

COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'EOBT_1_flt', 'AOBT_3_flt', 'MVT_TIME_UTC_mvt']
SHRINK, SEEDS = 0.5, (3, 11, 29)


def _state(df):
    d = L.delay_state_features(df, center=True); d['MVT_ID_mvt'] = d.MVT_ID_mvt.astype('int64')
    d = d[['MVT_ID_mvt'] + L.STATE_ALL]
    return d.rename(columns={c: c + '_c' for c in L.STATE_ALL})


def ltfm_layer(cp, base_path, months=(1, 7)):
    tr = pd.read_parquet(TRAINING_ALL, columns=COLS); tr = tr[tr.PHASE_mvt == 'DEP']
    rk = pd.read_parquet(RANKING, columns=COLS); rk = rk[rk.PHASE_mvt == 'DEP']
    s25, s26 = _state(tr), _state(rk)
    h = pd.read_parquet(HOLDOUT); idx = h.index
    cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64'); h = h.merge(s25, on='MVT_ID_mvt', how='left'); h.index = idx
    h = add_blocks(frame_base(h, cats)); y = h.TAXITIME_SEC_mvt.astype(float).to_numpy(); mk = obs_mask(h)
    rk = pd.read_parquet(RANKING); rk = rk[rk.PHASE_mvt == 'DEP'].copy(); rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    rk = rk.reset_index(drop=True).merge(s26, on='MVT_ID_mvt', how='left')
    fin = pd.read_parquet(base_path); fin['MVT_ID_mvt'] = fin.MVT_ID_mvt.astype('int64')
    rk = rk.merge(fin.rename(columns={'TAXITIME_SEC_mvt': 'pred'}), on='MVT_ID_mvt', how='left')
    rk = add_blocks(frame_base(rk, cats)).reset_index(drop=True)
    F = L.BASE9 + [c + '_c' for c in L.STATE3]
    mr = obs_mask(rk); lt = (rk.ADEP_mvt == 'LTFM').to_numpy()
    snow = L.cold_snow_flag(rk[lt], DATA_ROOT, 2026).reindex(rk.index, fill_value=False).to_numpy()
    c = np.zeros(len(rk))
    for mes in months:
        trm = ((h.ADEP_mvt == 'LTFM') & (h.mes == mes)).to_numpy() & mk; ap = lt & (rk.mes == mes).to_numpy() & mr
        c[ap] = L.fit_apply(h, rk, F, trm, ap, y, seeds=SEEDS, shrink=1.0, save=MODELS / 'layers' / f'ltfm_m{mes}')[ap]
    el = lt & mr & ~snow
    m = cp.assign(k=cp.MVT_ID_mvt.astype('int64')).merge(
        pd.DataFrame({'k': rk.MVT_ID_mvt.to_numpy(), 'pred': rk.pred.to_numpy(), 'c': c, 'el': el}), on='k', how='left')
    el_m = m.el.fillna(False).to_numpy(bool)
    assert np.allclose(m.TAXITIME_SEC_mvt[el_m], m.pred[el_m]), 'base do LTFM no candidato difere da v36/candidate_final'
    new = np.where(el_m, np.maximum(m.TAXITIME_SEC_mvt + SHRINK * m.c, 0.0), m.TAXITIME_SEC_mvt)
    return new, el_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--v36', required=True)
    ap.add_argument('--label', default='final')
    ap.add_argument('--pts-dir', default=str(FEATURES))
    ap.add_argument('--drift-csv', default=str(FEATURES / 'drift_pos_auc.csv'))
    a = ap.parse_args(); t0 = time.time()
    tmp = Path(tempfile.mkdtemp()) / 'cand_pos.parquet'
    PC.build(str(tmp), a.pts_dir, PC.APPROVED, a.drift_csv, a.base, a.v36, a.label)
    cp = pd.read_parquet(tmp); tpl = pd.read_parquet(SUBMITTING)
    assert (cp.MVT_ID_mvt.values == tpl.MVT_ID_mvt.values).all()
    new, el = ltfm_layer(cp, a.base)
    out = cp[['MVT_ID_mvt']].copy(); out['TAXITIME_SEC_mvt'] = new.astype(tpl.TAXITIME_SEC_mvt.dtype)
    assert out.TAXITIME_SEC_mvt.notna().all() and (out.TAXITIME_SEC_mvt >= 0).all() and out.MVT_ID_mvt.is_unique and len(out) == len(tpl)
    out.to_parquet(a.out, index=False)
    print('linhas', len(out), 'LTFM elegiveis', int(el.sum()), 'alteradas pelo LTFM', int((np.abs(out.TAXITIME_SEC_mvt.to_numpy() - cp.TAXITIME_SEC_mvt.to_numpy()) > 1e-9).sum()),
          'segundos', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
