import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prc.paths import ARTIFACTS, DATA_ROOT, FEATURES, MODELS, OOF, RANKING

TRAINING_ALL = DATA_ROOT / 'processed' / 'training_all_2025.parquet'
HOLDOUT = OOF / 'champion_h045_holdout.parquet'
LAYER_MODELS = MODELS / 'layers'
import lirf_rule_mixture as LR
import lirf_normal_flight_copy as FC
import lirf_sem_match as SM
import ltfm_delay_state_corrector as L
from adsb_corrector_champion import frame_base
from adsb_corrector_callsign import add_blocks, obs_mask

RANK_COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AOBT_3_flt',
             'LOBT_flt', 'EOBT_1_flt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'BLOCK_TIME_UTC_mvt']
TRAIN_COLS = RANK_COLS + ['TAXITIME_SEC_mvt']
LTFM_F = L.BASE9 + [c + '_c' for c in L.STATE3]
ORDER = ['lirf_rule', 'flight_copy', 'ltfm', 'sem_match']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_rank(path):
    rk = pd.read_parquet(path, columns=RANK_COLS)
    rk.loc[rk.PHASE_mvt == 'DEP', 'BLOCK_TIME_UTC_mvt'] = pd.NaT
    rk['MVT_ID_mvt'] = rk.MVT_ID_mvt.astype('int64')
    return rk


def load_train():
    return pd.read_parquet(TRAINING_ALL, columns=TRAIN_COLS)


def align_pred(pred_path, rk):
    p = pd.read_parquet(pred_path, columns=['MVT_ID_mvt', 'TAXITIME_SEC_mvt'])
    ids = p.MVT_ID_mvt.astype('int64').to_numpy(); base = p.TAXITIME_SEC_mvt.to_numpy(dtype='float64')
    assert len(np.unique(ids)) == len(ids), 'ids repetidos no pred'
    assert np.isfinite(base).all(), 'pred com NaN/inf'
    dep = rk[rk.PHASE_mvt == 'DEP'].set_index('MVT_ID_mvt')
    missing = ~np.isin(ids, dep.index.to_numpy())
    assert not missing.any(), f'{int(missing.sum())} ids do pred nao sao partidas do ranking'
    dep = dep.loc[ids].reset_index()
    dep['mes'] = dep.MVT_TIME_UTC_mvt.dt.month
    return ids, base, dep


def layer_lirf_rule(t, dep, base, w=0.75, lo_h=1.0, hi_h=14.0):
    dep25 = t[(t.PHASE_mvt == 'DEP') & t.TAXITIME_SEC_mvt.notna()]
    c = LR.cell_frame(dep25)
    c = c[(c.m >= lo_h * 3600) & (c.m < hi_h * 3600)].copy()
    y = c.TAXITIME_SEC_mvt.astype(float); c['eco'] = ((y - c.m).abs() <= 120).astype(int); c['y'] = y
    allm = pd.Series(True, index=c.index)
    rc = LR.cell_frame(dep); rc = rc[(rc.m >= lo_h * 3600) & (rc.m < hi_h * 3600)]
    if len(rc) == 0:
        return np.zeros(0, int), np.zeros(0), dict(cell_rows=0)
    p = LR.fit_predict(c, allm, rc, save=LAYER_MODELS / 'lirf_rule.joblib'); mu = LR.mu_of(c, allm)
    mix = p.to_numpy() * rc.m.to_numpy() + (1 - p.to_numpy()) * mu
    pos = rc.index.to_numpy()
    new = np.clip(w * mix + (1 - w) * base[pos], 0, None)
    return pos, new, dict(cell_rows=int(len(pos)), mu=float(mu), p_mean=float(p.mean()), w=w, lo_h=lo_h, hi_h=hi_h)


def layer_flight_copy(t, dep, cur, months, windows, w=0.75, thr=0.9, nmin=8, kmin=2):
    x = FC.label_frame(t)
    fs = {m: FC.flag_set(x, windows[m], thr, nmin, kmin) for m in months if m in windows}
    missing = [m for m in months if m not in windows]
    rows = dep[dep.ADEP_mvt == 'LIRF']
    b = pd.Series(cur[rows.index.to_numpy()], index=rows.index)
    nv, ch = FC.apply_layer(rows, b, fs, w)
    pos = rows.index[ch.to_numpy()].to_numpy()
    return pos, nv[ch].to_numpy(), dict(flights={int(m): len(fs[m]) for m in fs}, months_without_window=missing, w=w, thr=thr, nmin=nmin, kmin=kmin,
                                        windows={int(m): windows[m] for m in fs})


def _state(df):
    d = L.delay_state_features(df, center=True); d['MVT_ID_mvt'] = d.MVT_ID_mvt.astype('int64')
    return d[['MVT_ID_mvt'] + L.STATE_ALL].rename(columns={c: c + '_c' for c in L.STATE_ALL})


def ltfm_train_frame(t, h, s25, cats, mes, mode, proxy_path, drop_snow_train=True):
    if mode == 'h045':
        hh = h[(h.ADEP_mvt == 'LTFM') & (h.MVT_TIME_UTC_mvt.dt.month == mes)].copy()
        aux = t.loc[hh.index, ['EOBT_1_flt']]; hh = hh.join(aux)
    else:
        hh = t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LTFM') & (t.MVT_TIME_UTC_mvt.dt.month == mes) & t.TAXITIME_SEC_mvt.notna()].copy()
        px = pd.read_parquet(proxy_path, columns=['MVT_ID_mvt', 'proxy']); px['MVT_ID_mvt'] = px.MVT_ID_mvt.astype('int64')
        hh['MVT_ID_mvt'] = hh.MVT_ID_mvt.astype('int64')
        hh = hh.merge(px.rename(columns={'proxy': 'pred'}), on='MVT_ID_mvt', how='inner')
    hh['MVT_ID_mvt'] = hh.MVT_ID_mvt.astype('int64')
    idx = hh.index; hh = hh.merge(s25, on='MVT_ID_mvt', how='left'); hh.index = idx
    hh = add_blocks(frame_base(hh, cats))
    snow = L.cold_snow_flag(hh, DATA_ROOT, 2025).to_numpy() if drop_snow_train else np.zeros(len(hh), bool)
    hh['snow'] = snow
    return hh


def layer_ltfm(t, dep, cur, months, rank_all, mode_by_month, proxy_path, shrink=0.5, seeds=(3, 11, 29), n_jobs=4, drop_snow_train=True,
               year=None):
    h = pd.read_parquet(HOLDOUT)
    cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique())
    h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64')
    tr_ltfm = t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LTFM')]
    s25 = _state(tr_ltfm)
    rk_ltfm_all = rank_all[(rank_all.PHASE_mvt == 'DEP') & (rank_all.ADEP_mvt == 'LTFM')]
    s26 = _state(rk_ltfm_all)
    year = year or int(rank_all.MVT_TIME_UTC_mvt.dt.year.mode().iloc[0])
    rk = dep.copy(); rk['pred'] = cur
    rk = rk.merge(s26, on='MVT_ID_mvt', how='left'); rk.index = dep.index
    rk = add_blocks(frame_base(rk, cats))
    mr = obs_mask(rk); lt = (rk.ADEP_mvt == 'LTFM').to_numpy()
    snow = L.cold_snow_flag(rk[lt], DATA_ROOT, year).reindex(rk.index, fill_value=False).to_numpy()
    c = np.zeros(len(rk)); info = {}
    for mes in months:
        mode = mode_by_month[mes]
        ap = lt & (rk.mes == mes).to_numpy() & mr
        if mode == 'skip' or not ap.any():
            info[int(mes)] = dict(mode=mode, applied=0); continue
        trf = ltfm_train_frame(t, h, s25, cats, mes, mode, proxy_path, drop_snow_train)
        mk = obs_mask(trf) & ~trf.snow.to_numpy()
        y = trf.TAXITIME_SEC_mvt.astype(float).to_numpy()
        c[ap] = L.fit_apply(trf, rk, LTFM_F, mk, ap, y, seeds=seeds, shrink=1.0, params={'n_jobs': n_jobs}, save=LAYER_MODELS / f'ltfm_m{mes}')[ap]
        info[int(mes)] = dict(mode=mode, train_rows=int(mk.sum()), train_rows_snow_dropped=int((obs_mask(trf) & trf.snow.to_numpy()).sum()),
                              train_pred_rmse=float(np.sqrt(((trf.pred.to_numpy() - y)[mk] ** 2).mean())),
                              applied=int(ap.sum()), snow_excluded=int((ap & snow).sum()))
    el = lt & mr & ~snow & np.isin(rk.mes.to_numpy(), list(months)) & np.array([mode_by_month.get(m) != 'skip' for m in rk.mes.to_numpy()])
    pos = np.flatnonzero(el)
    new = np.maximum(cur[pos] + shrink * c[pos], 0.0)
    return pos, new, dict(shrink=shrink, seeds=list(seeds), n_jobs=n_jobs, drop_snow_train=drop_snow_train, by_month=info,
                          eligible=int(el.sum()), snow_rows_ltfm=int((lt & snow).sum()))


def layer_sem_match(t, rank_all, ids, cur, dose=0.75, hi_apply_h=4.0):
    SM.APPLY_HI_S = hi_apply_h * 3600
    pos2 = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': cur})
    out, info = SM.build_layer(pos2, t, rank_all, dose_mix=dose, save_dir=LAYER_MODELS / 'sem_match')
    new = out.TAXITIME_SEC_mvt.to_numpy(dtype='float64')
    pos = np.flatnonzero(new != cur)
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()}
    return pos, new[pos], dict(info, dose=dose, hi_apply_h=hi_apply_h)


def apply_change(cur, pos, new, min_change=0.0):
    if min_change > 0:
        keep = np.abs(new - cur[pos]) >= min_change; pos, new = pos[keep], new[keep]
    changed = new != cur[pos]
    pos, new = pos[changed], new[changed]
    before = cur[pos].copy(); cur = cur.copy(); cur[pos] = new
    return cur, pos, before, new


def summarize(rec, dep, months):
    out = {}
    for cam in ORDER:
        r = rec[rec.camada == cam]
        if len(r) == 0:
            out[cam] = {}; continue
        g = r.assign(d=r.valor_depois - r.valor_antes).groupby(['apt', 'mes']).d
        out[cam] = {f'{a}-{m}': dict(n_changed=int(len(v)), rms=float(np.sqrt((v ** 2).mean())), mean=float(v.mean()),
                                     p50_abs=float(v.abs().median()), max_abs=float(v.abs().max())) for (a, m), v in g}
    return out


def run(a):
    t0 = time.time(); times = {}
    rk = load_rank(a.rank); t = load_train()
    ids, base, dep = align_pred(a.pred, rk)
    months = tuple(int(m) for m in (a.months or sorted(dep.mes.unique())))
    sel_m = np.isin(dep.mes.to_numpy(), months)
    log(f'pred {len(ids)} linhas; meses no pred {sorted(dep.mes.unique().tolist())}; meses processados {months}; ranking {a.rank}')
    cur = base.copy(); recs = []; report = dict(args=vars(a), months=list(months), layers={})

    def record(cam, pos, before, after):
        recs.append(pd.DataFrame({'MVT_ID_mvt': ids[pos], 'apt': dep.ADEP_mvt.to_numpy()[pos], 'mes': dep.mes.to_numpy()[pos].astype('int16'),
                                  'camada': cam, 'valor_antes': before, 'valor_depois': after}))
        d = after - before
        report['layers'][cam]['n_changed'] = int(len(pos)); report['layers'][cam]['rms_change'] = float(np.sqrt((d ** 2).mean())) if len(d) else 0.0
        log(f'  {cam}: {len(pos)} linhas alteradas, RMS {report["layers"][cam]["rms_change"]:.2f} s')

    tt = time.time(); pos, new, info = layer_lirf_rule(t, dep, base, a.w_lirf, a.lo_h, a.hi_h)
    k = sel_m[pos]; pos, new = pos[k], new[k]
    report['layers']['lirf_rule'] = info; cur, pos, bef, aft = apply_change(cur, pos, new); record('lirf_rule', pos, bef, aft); times['lirf_rule'] = time.time() - tt
    tt = time.time(); windows = {int(k): v for k, v in (json.loads(a.fc_windows) if a.fc_windows else FC.WINDOWS_2026).items()}
    pos, new, info = layer_flight_copy(t, dep, cur, months, windows, a.fc_w, a.fc_thr, a.fc_nmin, a.fc_kmin)
    report['layers']['flight_copy'] = info; cur, pos, bef, aft = apply_change(cur, pos, new, a.fc_min_change); record('flight_copy', pos, bef, aft); times['flight_copy'] = time.time() - tt
    tt = time.time()
    mode_by_month = {m: (a.ltfm if a.ltfm != 'auto' else ('h045' if m in (1, 7) else 'proxy')) for m in months}
    for m in (a.ltfm_skip_months or []):
        mode_by_month[int(m)] = 'skip'
    lt = (dep.ADEP_mvt == 'LTFM').to_numpy()
    assert np.array_equal(cur[lt], base[lt]), 'paridade: a base do LTFM ja foi alterada por outra camada'
    pos, new, info = layer_ltfm(t, dep, cur, months, rk, mode_by_month, a.proxy, a.ltfm_shrink, tuple(a.ltfm_seeds), a.threads, not a.ltfm_keep_snow_train)
    report['layers']['ltfm'] = info; cur, pos, bef, aft = apply_change(cur, pos, new); record('ltfm', pos, bef, aft); times['ltfm'] = time.time() - tt
    tt = time.time(); pos, new, info = layer_sem_match(t, rk, ids, cur, a.sm_dose, a.sm_hi_h)
    k = sel_m[pos]; pos, new = pos[k], new[k]
    report['layers']['sem_match'] = info; cur, pos, bef, aft = apply_change(cur, pos, new, a.sm_min_change); record('sem_match', pos, bef, aft); times['sem_match'] = time.time() - tt

    rec = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=['MVT_ID_mvt', 'apt', 'mes', 'camada', 'valor_antes', 'valor_depois'])
    rec['camada'] = rec.camada.astype(str)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists() or a.overwrite, f'{out} ja existe (use --overwrite)'
    rec.to_parquet(out, index=False)
    pred_out = Path(a.pred_out) if a.pred_out else out.with_name(out.stem + '_pred.parquet')
    assert not pred_out.exists() or a.overwrite, f'{pred_out} ja existe'
    assert np.isfinite(cur).all() and (cur >= 0).all()
    pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': cur}).to_parquet(pred_out, index=False)
    d = cur - base
    report['summary'] = summarize(rec, dep, months)
    report['total'] = dict(rows=int(len(ids)), n_changed=int((d != 0).sum()), rms_change_all=float(np.sqrt((d ** 2).mean())),
                           rms_change_changed=float(np.sqrt((d[d != 0] ** 2).mean())) if (d != 0).any() else 0.0, min_pred=float(cur.min()), mean_pred=float(cur.mean()))
    report['seconds'] = {k: round(v, 1) for k, v in times.items()} | {'total': round(time.time() - t0, 1)}
    report['outputs'] = dict(long=str(out), pred=str(pred_out))
    rep = Path(a.report) if a.report else out.with_suffix('.json')
    rep.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    log(f'gravado {out} ({len(rec)} registros), {pred_out}, {rep}; total {report["total"]}; segundos {report["seconds"]}')
    return report


def eval_ltfm(a):
    t = load_train(); h = pd.read_parquet(HOLDOUT)
    cats = sorted(h.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})')[0].dropna().unique()); h['MVT_ID_mvt'] = h.MVT_ID_mvt.astype('int64')
    s25 = _state(t[(t.PHASE_mvt == 'DEP') & (t.ADEP_mvt == 'LTFM')])
    px = pd.read_parquet(a.proxy, columns=['MVT_ID_mvt', 'mes', 'proxy', 'y']); px['MVT_ID_mvt'] = px.MVT_ID_mvt.astype('int64')
    months = tuple(int(m) for m in (a.months or (2, 6)))
    res = dict(months=list(months), shrink=a.ltfm_shrink, seeds=list(a.ltfm_seeds), drop_snow_train=not a.ltfm_keep_snow_train, cv={}, transfer={})
    frames = {}
    for mes in sorted(set(months) | {1, 7}):
        mode = 'h045' if mes in (1, 7) else 'proxy'
        f = ltfm_train_frame(t, h, s25, cats, mes, mode, a.proxy, not a.ltfm_keep_snow_train)
        f['y'] = f.TAXITIME_SEC_mvt.astype(float); f['mk'] = obs_mask(f) & ~f.snow.to_numpy()
        frames[mes] = (mode, f)
        if mode == 'h045':
            hm = h[h.MVT_TIME_UTC_mvt.dt.month == mes]; sse = float(((hm.pred - hm.TAXITIME_SEC_mvt.astype(float)) ** 2).sum()); n_all = len(hm)
        else:
            pm = px[px.mes == mes]; sse = float(((pm.proxy - pm.y) ** 2).sum()); n_all = len(pm)
        y = f.y.to_numpy(); mk = f.mk.to_numpy(); pred = f.pred.to_numpy()
        oof = np.zeros(len(f))
        for b in np.unique(f.bloco[mk]):
            tr = mk & (f.bloco != b).to_numpy(); te = mk & (f.bloco == b).to_numpy()
            oof[te] = L.fit_apply(f, f, LTFM_F, tr, te, y, seeds=tuple(a.ltfm_seeds), shrink=1.0, params={'n_jobs': a.threads})[te]
        new = np.maximum(pred + a.ltfm_shrink * oof, 0)
        e0 = (pred[mk] - y[mk]) ** 2; e1 = (new[mk] - y[mk]) ** 2
        days = f.dia.to_numpy()[mk]; dd = pd.Series(e1 - e0).groupby(days).sum()
        rs = np.random.default_rng(0); g = dd.to_numpy()
        bs = np.array([g[rs.integers(0, len(g), len(g))].sum() for _ in range(3000)])
        d_tot = lambda s: float(np.sqrt((sse + s) / n_all) - np.sqrt(sse / n_all))
        res['cv'][int(mes)] = dict(mode=mode, n_train_mask=int(mk.sum()), n_snow_dropped=int((obs_mask(f) & f.snow.to_numpy()).sum()),
                                   rmse_ltfm_before=float(np.sqrt(e0.mean())), rmse_ltfm_after=float(np.sqrt(e1.mean())),
                                   delta_rmse_ltfm=float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                                   delta_rmse_month_scale=d_tot(float((e1 - e0).sum())), month_rows=n_all, month_rmse_before=float(np.sqrt(sse / n_all)),
                                   ic90_month_scale=[d_tot(float(np.percentile(bs, 5))), d_tot(float(np.percentile(bs, 95)))],
                                   days_improved=f'{int((dd < 0).sum())}/{len(dd)}', s_eff=float((oof[mk] * (y[mk] - pred[mk])).sum() / (oof[mk] ** 2).sum()),
                                   corr_rms=float(np.sqrt((oof[mk] ** 2).mean())))
        log(f'CV mes {mes} ({mode}): LTFM {res["cv"][mes]["rmse_ltfm_before"]:.2f} -> {res["cv"][mes]["rmse_ltfm_after"]:.2f}; escala do mes {res["cv"][mes]["delta_rmse_month_scale"]:+.3f} '
            f'IC90 {res["cv"][mes]["ic90_month_scale"]}; dias {res["cv"][mes]["days_improved"]}; s_eff {res["cv"][mes]["s_eff"]:.2f}')
    pairs = [(m, 1 if m <= 4 else 7) for m in months if m not in (1, 7)]
    for src, dst in pairs + [(d, s) for s, d in pairs]:
        ms, fs = frames[src]; md, fd = frames[dst]
        c = L.fit_apply(fs, fd, LTFM_F, fs.mk.to_numpy(), fd.mk.to_numpy(), fs.y.to_numpy(), seeds=tuple(a.ltfm_seeds), shrink=1.0, params={'n_jobs': a.threads})
        mk = fd.mk.to_numpy(); y = fd.y.to_numpy(); pred = fd.pred.to_numpy(); new = np.maximum(pred + a.ltfm_shrink * c, 0)
        e0 = (pred[mk] - y[mk]) ** 2; e1 = (new[mk] - y[mk]) ** 2
        res['transfer'][f'{src}->{dst}'] = dict(train_mode=ms, apply_mode=md, n_apply=int(mk.sum()), rmse_ltfm_before=float(np.sqrt(e0.mean())),
                                                rmse_ltfm_after=float(np.sqrt(e1.mean())), delta_rmse_ltfm=float(np.sqrt(e1.mean()) - np.sqrt(e0.mean())),
                                                s_eff=float((c[mk] * (y[mk] - pred[mk])).sum() / (c[mk] ** 2).sum()), corr_rms=float(np.sqrt((c[mk] ** 2).mean())))
        log(f'transfer {src}->{dst}: LTFM {res["transfer"][f"{src}->{dst}"]["rmse_ltfm_before"]:.2f} -> {res["transfer"][f"{src}->{dst}"]["rmse_ltfm_after"]:.2f} s_eff {res["transfer"][f"{src}->{dst}"]["s_eff"]:.2f}')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8'); log(f'gravado {out}')
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', help='parquet (MVT_ID_mvt, TAXITIME_SEC_mvt) com a BASE de 2026 (v36 do final); define as linhas tratadas')
    ap.add_argument('--out', default=str(ARTIFACTS / 'layers' / 'camadas_nao_adsb.parquet'), help='saida longa (MVT_ID_mvt, apt, mes, camada, valor_antes, valor_depois) ou json no --eval-ltfm')
    ap.add_argument('--pred-out', default=None, help='vetor final (default: <out>_pred.parquet)')
    ap.add_argument('--report', default=None, help='json de relatorio (default: <out>.json)')
    ap.add_argument('--rank', default=str(RANKING))
    ap.add_argument('--months', type=int, nargs='*', default=None, help='meses-alvo (default: os meses presentes no pred)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    g = ap.add_argument_group('lirf_rule'); g.add_argument('--w-lirf', type=float, default=0.75); g.add_argument('--lo-h', type=float, default=1.0); g.add_argument('--hi-h', type=float, default=14.0)
    g = ap.add_argument_group('flight_copy'); g.add_argument('--fc-w', type=float, default=0.75); g.add_argument('--fc-thr', type=float, default=0.9)
    g.add_argument('--fc-nmin', type=int, default=8); g.add_argument('--fc-kmin', type=int, default=2); g.add_argument('--fc-min-change', type=float, default=0.5)
    g.add_argument('--fc-windows', default=None, help='JSON {mes: [meses de 2025]}; default WINDOWS_2026 do modulo')
    g = ap.add_argument_group('ltfm'); g.add_argument('--ltfm', choices=['auto', 'h045', 'proxy', 'skip'], default='auto', help='auto: h045 em jan/jul, proxy nos demais')
    g.add_argument('--ltfm-skip-months', type=int, nargs='*', default=None, help='meses em que a camada LTFM NAO e aplicada (ex.: 2, se a decisao for pular fev)')
    g.add_argument('--ltfm-shrink', type=float, default=0.5); g.add_argument('--ltfm-seeds', type=int, nargs='+', default=[3, 11, 29])
    g.add_argument('--ltfm-keep-snow-train', action='store_true', help='NAO tirar as linhas de neve fria do treino de 2025 (default: tira; no-op em jan/jul)')
    g.add_argument('--proxy', default=str(FEATURES / 'proxy2_pred.parquet'))
    g = ap.add_argument_group('sem_match'); g.add_argument('--sm-dose', type=float, default=0.75); g.add_argument('--sm-hi-h', type=float, default=4.0); g.add_argument('--sm-min-change', type=float, default=0.5)
    ap.add_argument('--eval-ltfm', action='store_true', help='so avalia o LTFM (opcao a) em 2025 e grava json em --out')
    return ap.parse_args(argv)


if __name__ == '__main__':
    a = parse()
    if a.eval_ltfm:
        eval_ltfm(a)
    else:
        assert a.pred, '--pred obrigatorio'
        run(a)
