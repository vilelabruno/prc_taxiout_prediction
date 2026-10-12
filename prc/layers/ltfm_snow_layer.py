import pandas as pd


def _metar(root, year):
    e = pd.read_parquet(f'{root}/processed/metar_events_{year}.parquet')
    e = e[e.ADEP_mvt == 'LTFM'].copy()
    e['t'] = e.metar_time_utc
    e['sn'] = e.wxcodes.fillna('').str.contains('SN').astype(float)
    e['tc'] = (e.tmpf - 32) / 1.8
    e['sknt'] = e.sknt
    e = e.sort_values('t').set_index('t')
    e['sn_r6'] = e.sn.rolling('6h', closed='both').max()
    return e[['tc', 'sknt', 'sn', 'sn_r6']].reset_index()
