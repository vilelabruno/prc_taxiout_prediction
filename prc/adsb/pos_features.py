import numpy as np, pandas as pd

POS_FULL = ['ds0', 'dsfm', 'dsl', 'gs0', 'gsfm', 'tfm_off', 'kin'] + [f for k in (60, 120, 300, 600, 900) for f in (f'dl_{k}', f'gs_{k}')]
POS_LEAN = ['ds0', 'dsfm', 'dsl', 'gs0', 'gsfm', 'tfm_off', 'kin']
MIN_N = 3
MOVED_M = 30.0


def hav(la1, lo1, la2, lo2):
    q = np.pi / 180
    a = np.sin((la2 - la1) * q / 2) ** 2 + np.cos(la1 * q) * np.cos(la2 * q) * np.sin((lo2 - lo1) * q / 2) ** 2
    return 12742000 * np.arcsin(np.sqrt(a))


def _tab(a):
    a = a[a.ags < 3]
    return a.groupby(['ADES_mvt', 'STAND_mvt']).agg(sla=('alat', 'median'), slo=('alon', 'median'), n=('alat', 'size')).reset_index()


def stand_tables(arr25, arr26, mode='year', min_n=MIN_N):
    t25, t26 = _tab(arr25), _tab(arr26)
    if mode == 'pooled':
        t = _tab(pd.concat([arr25, arr26]))
        t = t[t.n >= 5].rename(columns={'ADES_mvt': 'ADEP_mvt'})[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']]
        return t, t
    m = t25.merge(t26, on=['ADES_mvt', 'STAND_mvt'], how='outer', suffixes=('_a', '_b'))
    m['d'] = hav(m.sla_a, m.slo_a, m.sla_b, m.slo_b)
    if mode == 'cons':
        tp = _tab(pd.concat([arr25, arr26]))
        tp = tp[tp.n >= min_n].merge(m[['ADES_mvt', 'STAND_mvt', 'd', 'n_a', 'n_b']], on=['ADES_mvt', 'STAND_mvt'], how='left')
        moved = (tp.n_a.fillna(0) >= min_n) & (tp.n_b.fillna(0) >= min_n) & (tp.d > MOVED_M)
        pool = tp[~moved].rename(columns={'ADES_mvt': 'ADEP_mvt'})[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']]
        mv = m[(m.n_a.fillna(0) >= min_n) & (m.n_b.fillna(0) >= min_n) & (m.d > MOVED_M)]
        oa = mv.rename(columns={'ADES_mvt': 'ADEP_mvt', 'sla_a': 'sla', 'slo_a': 'slo'})[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']]
        ob = mv.rename(columns={'ADES_mvt': 'ADEP_mvt', 'sla_b': 'sla', 'slo_b': 'slo'})[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']]
        return pd.concat([pool, oa]).reset_index(drop=True), pd.concat([pool, ob]).reset_index(drop=True)

    def build(me, other):
        n_me, n_ot = m[f'n_{me}'].fillna(0), m[f'n_{other}'].fillna(0)
        own_ok = n_me >= min_n
        other_ok = n_ot >= min_n
        agree = m.d.isna() | (m.d <= MOVED_M)
        use_other = (~own_ok) & other_ok & (n_me == 0) & (mode == 'yearfb')
        lat = np.where(own_ok, m[f'sla_{me}'], np.where(use_other, m[f'sla_{other}'], np.nan))
        lon = np.where(own_ok, m[f'slo_{me}'], np.where(use_other, m[f'slo_{other}'], np.nan))
        out = pd.DataFrame({'ADEP_mvt': m.ADES_mvt, 'STAND_mvt': m.STAND_mvt, 'sla': lat, 'slo': lon}).dropna()
        return out
    return build('a', 'b'), build('b', 'a')


def add_pos_features(d, pts, stands):
    idx = d.index
    d = d.merge(pts.assign(MVT_ID_mvt=pts.MVT_ID_mvt.astype('int64')), on='MVT_ID_mvt', how='left')
    d = d.merge(stands[['ADEP_mvt', 'STAND_mvt', 'sla', 'slo']], on=['ADEP_mvt', 'STAND_mvt'], how='left')
    d.index = idx
    d['ds0'] = hav(d.lat0, d.lon0, d.sla, d.slo)
    d['dsfm'] = hav(d.latfm, d.lonfm, d.sla, d.slo)
    d['dsl'] = hav(d.latl, d.lonl, d.sla, d.slo)
    for k in (60, 120, 300, 600, 900):
        d[f'dl_{k}'] = hav(d[f'lat_{k}'], d[f'lon_{k}'], d.latl, d.lonl)
    d['tfm_off'] = d.tfm - d.t0
    d['kin'] = d.ds0 / np.maximum(d.gs0.fillna(0) * 0.514, 1.5)
    return d
