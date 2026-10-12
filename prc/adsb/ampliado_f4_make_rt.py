import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'layers')); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import FEATURES, OOF, RANKING
from adsb_corrector_champion import frame_base
from adsb_corrector_callsign import add_blocks
import pos_features as pf
import ampliado_f4_stand_routes as S
import ampliado_f4_leg_features as LG
import own_flight_features as O
from own_flight_features import TRAINING_ALL

D = FEATURES
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
COLS = ['MVT_ID_mvt', 'ADES_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt']


def frames(fin):
    a25 = pd.read_parquet(D / 'e2_arr_2025.parquet').merge(pd.read_parquet(TRAINING_ALL, columns=COLS), on='MVT_ID_mvt')
    rank_path = Path(fin['rank'])
    a26 = pd.read_parquet(Path(fin['arr'])).merge(pd.read_parquet(rank_path, columns=COLS), on='MVT_ID_mvt')
    t25, t26 = pf.stand_tables(a25, a26, 'cons')
    h = pd.read_parquet(HOLDOUT); idx = h.index
    t = pd.read_parquet(TRAINING_ALL, columns=['MVT_ID_mvt', 'CALLSIGN_flt'])
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64'); t['MVT_ID_mvt'] = t.MVT_ID_mvt.astype('int64')
    h = h.merge(t, on='MVT_ID_mvt', how='left'); h.index = idx
    own = pd.read_parquet(D / 'adsb_own_cs_2025.parquet'); own['MVT_ID_mvt'] = own.MVT_ID_mvt.astype('int64')
    h = h.merge(own, on='MVT_ID_mvt', how='left'); h.index = idx; h['adsb_cov'] = h.adsb_cov.fillna(0)
    h = pf.add_pos_features(h, pd.read_parquet(D / 'e2_pts_2025.parquet'), t25)
    h = add_blocks(frame_base(h))
    rk = pd.read_parquet(rank_path); rk = rk[rk.PHASE_mvt == 'DEP']
    if fin.get('months'):
        rk = rk[rk.MVT_TIME_UTC_mvt.dt.month.isin([int(m) for m in str(fin['months']).split(',')])]
    rk = rk.reset_index(drop=True); rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    o = pd.read_parquet(Path(fin['own'])); o['MVT_ID_mvt'] = o.MVT_ID_mvt.astype('int64')
    rk = rk.merge(o, on='MVT_ID_mvt', how='left'); rk['adsb_cov'] = rk.adsb_cov.fillna(0)
    rk = pf.add_pos_features(rk, pd.read_parquet(Path(fin['pts'])), t26)
    return h, rk


def _prep(d):
    c = d.CALLSIGN_flt.astype('string').str.strip().replace('', pd.NA); f = d.FLIGHT_mvt.astype('string').str.strip()
    return pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt.to_numpy(), 'ADEP_mvt': d.ADEP_mvt.to_numpy(), 'STAND_mvt': d.STAND_mvt.astype(str).to_numpy(), 'key': c.fillna(f).to_numpy(),
                         'mvt': (d.MVT_TIME_UTC_mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy(),
                         'day': d.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d').to_numpy(), 'sla': d.sla.to_numpy(), 'slo': d.slo.to_numpy(),
                         'ds0': d.ds0.to_numpy(), 'gs0': d.gs0.to_numpy()})


def make_refs(h, rk, out, nproc=6, base=None):
    a, b = _prep(h), _prep(rk)
    a = a[(a.ds0 < S.REF_DS) & (a.gs0 < 3)]; b = b[(b.ds0 < S.REF_DS) & (b.gs0 < 3)]
    if base is None:
        S.build_refs(pd.concat([a, b]).drop(columns=['ds0', 'gs0']), out, nproc); return
    out = Path(out); novas = out.with_name(out.stem + '_novas.parquet')
    n = S.build_refs(b.drop(columns=['ds0', 'gs0']), novas, nproc)
    r0 = pd.read_parquet(base); dup = set(r0.ref) & set(n.ref)
    if dup: print('refs: ids repetidos entre a base e as novas:', len(dup), '(a versao nova substitui a da base)'); r0 = r0[~r0.ref.isin(dup)]
    pd.concat([r0, n]).reset_index(drop=True).to_parquet(out, index=False); print('refs total', len(r0) + len(n), 'voos', r0.ref.nunique() + n.ref.nunique())


def _router(h, rk, refs):
    st = pd.concat([h[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']], rk[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']]]).dropna().drop_duplicates(['ADEP_mvt', 'STAND_mvt']).rename(columns={'ADEP_mvt': 'apt', 'STAND_mvt': 'stand'})
    st['stand'] = st.stand.astype(str)
    return S.Router(refs, st)


def make_rt(d, R_):
    out = np.full((len(d), 8), np.nan); cache = {}
    def q(apt, stand, lat, lon, own):
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return (np.nan, 0, np.nan, np.nan)
        k = (apt, stand, round(lat / 1.3e-4), round(lon / 1.8e-4))
        if k in cache and not (own in cache[k][1]):
            return cache[k][0]
        r = R_.query(apt, stand, lat, lon, own); cache[k] = (r, {own}); return r
    A = d.ADEP_mvt.to_numpy(); ST = d.STAND_mvt.astype(str).to_numpy(); L0 = d.lat0.to_numpy(float); O0 = d.lon0.to_numpy(float)
    LF = d.latfm.to_numpy(float); OF = d.lonfm.to_numpy(float); ID = d.MVT_ID_mvt.to_numpy(); cov = (d.adsb_cov == 1).to_numpy()
    for i in range(len(d)):
        if not cov[i]:
            continue
        out[i, :4] = q(A[i], ST[i], L0[i], O0[i], ID[i]); out[i, 4:8] = q(A[i], ST[i], LF[i], OF[i], ID[i])
    f = pd.DataFrame(out, columns=['rt0', 'rt0_n', 'rt0_iqr', 'rt0_lvl', 'rtfm', 'rtfm_n', 'rtfm_iqr', 'rtfm_lvl'], index=d.index)
    f['MVT_ID_mvt'] = d.MVT_ID_mvt.to_numpy()
    f['tx0'] = d.adsb_first_off.to_numpy(float) + f.rt0; f['txfm'] = d.adsb_first_moving_off.to_numpy(float) + f.rtfm
    return f


def make_legrt(d, lg, R_):
    x = d[['MVT_ID_mvt', 'ADEP_mvt', 'STAND_mvt']].copy(); x['MVT_ID_mvt'] = x.MVT_ID_mvt.astype('int64'); lg = lg.copy(); lg['MVT_ID_mvt'] = lg.MVT_ID_mvt.astype('int64')
    x = x.merge(lg, on='MVT_ID_mvt', how='inner')
    A = x.ADEP_mvt.to_numpy(); ST = x.STAND_mvt.astype(str).to_numpy(); L0 = x.lg_lat0.to_numpy(float); O0 = x.lg_lon0.to_numpy(float)
    LF = x.lg_latfm.to_numpy(float); OF = x.lg_lonfm.to_numpy(float); ID = x.MVT_ID_mvt.to_numpy(); cache = {}; out = np.full((len(x), 8), np.nan)
    def q(apt, stand, lat, lon):
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return (np.nan, 0, np.nan, np.nan)
        k = (apt, stand, round(lat / 1.3e-4), round(lon / 1.8e-4))
        if k in cache:
            return cache[k]
        r = R_.query(apt, stand, lat, lon, None); cache[k] = r; return r
    for i in range(len(x)):
        out[i, :4] = q(A[i], ST[i], L0[i], O0[i]); out[i, 4:] = q(A[i], ST[i], LF[i], OF[i])
    f = pd.DataFrame(out, columns=['lrt0', 'lrt0_n', 'lrt0_iqr', 'lrt0_lvl', 'lrtfm', 'lrtfm_n', 'lrtfm_iqr', 'lrtfm_lvl']); f['MVT_ID_mvt'] = ID
    f['ltx0'] = x.lg_first_off.to_numpy(float) + f.lrt0; f['ltxfm'] = x.lg_first_moving_off.to_numpy(float) + f.lrtfm
    x2 = pd.concat([x.reset_index(drop=True).drop(columns=['ADEP_mvt', 'STAND_mvt', 'MVT_ID_mvt']), f], axis=1); x2['MVT_ID_mvt'] = ID
    return x2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(FEATURES)); ap.add_argument('--steps', default='refs,rt,lg,legrt'); ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--rank', default=str(RANKING)); ap.add_argument('--months', default=None); ap.add_argument('--label', default='final')
    ap.add_argument('--own', default=None); ap.add_argument('--pts', default=None); ap.add_argument('--arr', default=None)
    ap.add_argument('--refs-base', default=None); ap.add_argument('--lg', default=None); ap.add_argument('--with-2025', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); steps = a.steps.split(','); t0 = time.time()
    fin = dict(rank=a.rank, months=a.months, own=a.own or FEATURES / f'adsb_own_cs_{a.label}.parquet', pts=a.pts or FEATURES / f'e2_pts_{a.label}.parquet',
               arr=a.arr or FEATURES / f'e2_arr_{a.label}.parquet')
    h, rk = frames(fin); print('frames', len(h), len(rk), round(time.time() - t0, 1), 's', flush=True)
    sets = ((a.label, rk),)
    if a.with_2025:
        sets = (('2025', h),) + sets
    if 'refs' in steps:
        make_refs(h, rk, out / 'refs.parquet', a.nproc, a.refs_base); print('refs', round(time.time() - t0, 1), 's', flush=True)
    refs = pd.read_parquet(out / 'refs.parquet'); R_ = _router(h, rk, refs)
    if 'rt' in steps:
        for nm, d in sets:
            make_rt(d, R_).to_parquet(out / f'rt_{nm}.parquet'); print('rt', nm, round(time.time() - t0, 1), 's', flush=True)
    if 'lg' in steps and not a.lg:
        cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt']
        LG.run(a.label, O.read_rank(a.rank, cols, 'DEP', a.months), str(out / f'lg_{a.label}.parquet'), a.nproc)
        if a.with_2025:
            t = pd.read_parquet(TRAINING_ALL, columns=cols); t = t[(t.PHASE_mvt == 'DEP') & t.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])]
            LG.run('2025', t, str(out / 'lg_2025.parquet'), a.nproc)
        print('lg', round(time.time() - t0, 1), 's', flush=True)
    if 'legrt' in steps:
        for nm, d in sets:
            lg_path = Path(a.lg) if (a.lg and nm == a.label) else out / f'lg_{nm}.parquet'
            make_legrt(d, pd.read_parquet(lg_path), R_).to_parquet(out / f'leg_rt_{nm}.parquet'); print('leg_rt', nm, round(time.time() - t0, 1), 's', flush=True)
    print('total', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
