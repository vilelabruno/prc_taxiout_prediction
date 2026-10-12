import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from own_flight_features import load_day, EP, TRAINING_ALL
from prc.paths import FEATURES, RANKING
def process(args):
    day,df=args
    s=load_day(day); out=[]
    if s is None: return out
    cs=s.dropna(subset=['callsign']).groupby(['apt','callsign']).icao24.unique().to_dict()
    by={k:v for k,v in s.groupby('icao24')}
    for r in df.itertuples():
        ics=cs.get((r.ADES_mvt,r.key))
        if ics is None: continue
        best=None
        for ic in ics:
            g=by[ic]; w=g[(g.t>=r.mvt)&(g.t<=r.mvt+2400)&(g.apt==r.ADES_mvt)&(g.alt<=0)]
            if len(w)>=3 and (best is None or len(w)>len(best)): best=w
        if best is None: continue
        t=best.t.to_numpy(); gs=best.gs.to_numpy(float)
        last=best.iloc[-3:]
        out.append((r.MVT_ID_mvt,float(last.lat.median()),float(last.lon.median()),float(np.nanmedian(last.gs)),float(t[-1]-r.mvt)))
    return out
def run(label,df,out_path=None,nproc=12):
    df=df.copy(); df['mvt']=(df.MVT_TIME_UTC_mvt-EP).dt.total_seconds()
    c=df.CALLSIGN_flt.astype('string').str.strip().replace('',pd.NA); f=df.FLIGHT_mvt.astype('string').str.strip()
    df['key']=c.fillna(f); df['day']=df.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    tasks=[(d,g[['MVT_ID_mvt','ADES_mvt','key','mvt']]) for d,g in df.groupby('day')]
    rows=[]
    with Pool(nproc) as p:
        for res in p.imap_unordered(process,tasks): rows+=res
    out_path=Path(out_path) if out_path else FEATURES/f'e2_arr_{label}.parquet'
    out_path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['MVT_ID_mvt','alat','alon','ags','adt']).to_parquet(out_path,index=False); print(label,len(rows))
if __name__=='__main__':
    import argparse
    from own_flight_features import read_rank, rank_args
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('which',nargs='?',default='all',choices=['all','2025','final'])
    rank_args(ap); ap.add_argument('--nproc',type=int,default=12); a=ap.parse_args()
    cols=['MVT_ID_mvt','PHASE_mvt','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','MVT_TIME_UTC_mvt','STAND_mvt']
    if a.which in ('all','2025'):
        tr=pd.read_parquet(TRAINING_ALL,columns=cols); tr=tr[(tr.PHASE_mvt=='ARR')&tr.MVT_TIME_UTC_mvt.dt.month.isin([1,7])]; run('2025',tr,nproc=a.nproc)
    if a.which in ('all','final'):
        run(a.label,read_rank(a.rank or RANKING,cols,'ARR',a.months),a.out,a.nproc)
