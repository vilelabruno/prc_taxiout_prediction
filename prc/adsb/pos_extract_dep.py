import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from own_flight_features import load_day, EP, WIN_BEFORE, WIN_AFTER, TRAINING_ALL
from prc.paths import FEATURES, RANKING
def interp_at(t,v,tq):
    return np.interp(tq,t,v,left=np.nan,right=np.nan)
def summ(w,mvt):
    t=w.t.to_numpy(); gs=w.gs.to_numpy(float); lat=w.lat.to_numpy(); lon=w.lon.to_numpy(); alt=w.alt.to_numpy(float)
    g=np.nan_to_num(gs,nan=-1.0); mov=g>=2.0
    o={'t0':t[0],'lat0':lat[0],'lon0':lon[0],'gs0':gs[0],'n':len(t)}
    if mov.any():
        i=np.flatnonzero(mov)[0]; o.update(tfm=t[i],latfm=lat[i],lonfm=lon[i],gsfm=gs[i])
    ground=alt<=0
    j=np.flatnonzero(ground)
    if len(j): o.update(tl=t[j[-1]],latl=lat[j[-1]],lonl=lon[j[-1]],gsl=gs[j[-1]])
    for k in (60,120,300,600,900):
        tq=mvt-k
        if t[0]<=tq<=t[-1]:
            o[f'gs_{k}']=float(np.interp(tq,t,np.nan_to_num(gs,nan=0))); o[f'lat_{k}']=float(np.interp(tq,t,lat)); o[f'lon_{k}']=float(np.interp(tq,t,lon))
    return o
def process(args):
    day,df=args
    s=load_day(day); out=[]
    if s is None: return out
    cs=s.dropna(subset=['callsign']).groupby(['apt','callsign']).icao24.unique().to_dict()
    by={k:v for k,v in s.groupby('icao24')}
    for r in df.itertuples():
        ics=cs.get((r.ADEP_mvt,r.key))
        if ics is None: continue
        best=None
        for ic in ics:
            g=by[ic]; w=g[(g.t>=r.mvt-WIN_BEFORE)&(g.t<=r.mvt+WIN_AFTER)&(g.apt==r.ADEP_mvt)]
            if len(w)>=2 and (best is None or len(w)>len(best)): best=w
        if best is not None:
            o=summ(best,r.mvt); o['MVT_ID_mvt']=r.MVT_ID_mvt; out.append(o)
    return out
E2_COLS=['t0','lat0','lon0','gs0','n','tfm','latfm','lonfm','gsfm','tl','latl','lonl','gsl']+[f'{v}_{k}' for k in (60,120,300,600,900) for v in ('gs','lat','lon')]+['MVT_ID_mvt']
def run(label,df,out_path=None,nproc=12,canon=False):
    df=df.copy(); df['mvt']=(df.MVT_TIME_UTC_mvt-EP).dt.total_seconds()
    c=df.CALLSIGN_flt.astype('string').str.strip().replace('',pd.NA); f=df.FLIGHT_mvt.astype('string').str.strip()
    df['key']=c.fillna(f); df['day']=df.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
    tasks=[(d,g[['MVT_ID_mvt','ADEP_mvt','key','mvt']]) for d,g in df.groupby('day')]
    rows=[]
    with Pool(nproc) as p:
        for k,res in enumerate(p.imap_unordered(process,tasks)):
            rows+=res
            if k%20==0: print(label,k,len(tasks),flush=True)
    out_path=Path(out_path) if out_path else FEATURES/f'e2_pts_{label}.parquet'
    out_path.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    if canon:
        df=df.reindex(columns=E2_COLS)
    df.to_parquet(out_path,index=False); print(label,len(rows))
if __name__=='__main__':
    import argparse
    from own_flight_features import read_rank, rank_args
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('which',nargs='?',default='final',choices=['2025','final'])
    rank_args(ap); ap.add_argument('--nproc',type=int,default=12); a=ap.parse_args()
    cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','FLIGHT_mvt','CALLSIGN_flt','MVT_TIME_UTC_mvt']
    if a.which=='2025':
        tr=pd.read_parquet(TRAINING_ALL,columns=cols); tr=tr[(tr.PHASE_mvt=='DEP')&tr.MVT_TIME_UTC_mvt.dt.month.isin([1,7])]; run('2025',tr,a.out,a.nproc,canon=True)
    else:
        run(a.label,read_rank(a.rank or RANKING,cols,'DEP',a.months),a.out,a.nproc,canon=True)
