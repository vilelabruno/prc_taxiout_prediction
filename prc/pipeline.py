import os
import subprocess
import sys
from prc.paths import ARTIFACTS, FEATURES, REPO, WORK

PY = sys.executable
COMPONENTS = ['light70', 'light123', 'light153', 'specialist_single', 'specialist_five', 'specialist_arrival',
              'airport_experts', 'cat70', 'cat93', 'cat153', 'cat183']
V36 = str(ARTIFACTS / 'v36_final.parquet')
V35 = str(ARTIFACTS / 'baseb' / 'base_b_v35.parquet')


def m(mod, *args):
    return [PY, '-m', mod, *map(str, args)]


def p(path, *args):
    return [PY, path, *map(str, args)]


def stage_features(a):
    s = 'prc/adsb/'
    return [
        p(s + 'own_flight_features.py', 'all', '--nproc', a.nproc),
        p(s + 'own_flight_features_callsign.py', 'all', '--nproc', a.nproc),
        p(s + 'pos_extract_dep.py', 'all', '--nproc', a.nproc),
        p(s + 'pos_extract_arr_stands.py', 'all', '--nproc', a.nproc),
        p(s + 'lebl_adsb_offblock.py', 'all', '--nproc', a.nproc),
        p(s + 'ampliado_f4_leg_features.py', 'final', FEATURES / 'lg_final.parquet', '--nproc', a.nproc),
        p(s + 'otimista_features.py', 'final', FEATURES / 'ox_final.parquet', '--nproc', a.nproc),
        p(s + 'ox_features.py', 'final', FEATURES / 'ox_LIRF_final.parquet', '--nproc', a.nproc),
        p(s + 'ampliado_f4_make_rt.py', '--with-2025', '--nproc', a.nproc),
    ]


def stage_basea(a):
    c = [m('prc.basea.full_chain', 'prepare')]
    c += [m('prc.basea.full_chain', 'fit', '--name', n, '--threads', a.threads) for n in COMPONENTS]
    c += [m('prc.basea.assemble')]
    return c


def stage_baseb(a):
    return [m('prc.baseb', 'build', 'prc/baseb/configs/h045_h039_mlp_refit.yaml'), m('prc.baseb', 'v35', '--write')]


def stage_blend(a):
    return [m('prc.blend')]


def stage_layers(a):
    return [
        p('prc/layers/camadas_nao_adsb_final.py', '--pred', V36, '--out', ARTIFACTS / 'layers' / 'camadas_nao_adsb.parquet'),
        p('prc/layers/lebl_chain.py', '--base', V36, '--out', ARTIFACTS / 'layers' / 'lebl_chain.parquet'),
    ]


def stage_otimista(a):
    return [
        m('prc.otimista.aplicar_final', '--pred', V36, '--out', WORK / 'camadaD_final.parquet', '--months', '2,6', '--jobs', a.jobs),
        m('prc.otimista.hbase_prep'),
        m('prc.otimista.hbase_final', '--pred', V36, '--months', '2,6', '--jobs', a.jobs),
    ]


def stage_v53(a):
    return [m('prc.layers.compose_layers', '--e6', ARTIFACTS / 'layers' / 'camadas_nao_adsb_pred.parquet', '--camada-d', WORK / 'camadaD_final.parquet')]


def stage_stack(a):
    return [m('prc.stack.apply_perm', '--hbase-dir', WORK / 'hbase' / 'final')]


def stage_stack_full(a):
    dev = a.device
    return [
        m('prc.stack.nets.seq', 'prc/stack/nets/configs.json', '--device', dev),
        m('prc.stack.mlp.build_unified'),
        m('prc.stack.mlp.train', f'device={dev}'),
        m('prc.stack.build', '--mlp-npz', ARTIFACTS / 'oof' / 'mlp_member' / 'final.npz'),
        m('prc.stack.perm', '--mlp'),
        m('prc.stack.mlp_dep', '--device', dev),
        m('prc.stack.candidate', '--out', ARTIFACTS / 'stack_pred.parquet'),
    ]


def stage_espera(a):
    return [
        m('prc.espera_rot.dataset'),
        m('prc.espera_rot.train', '--threads', a.threads),
        m('prc.espera_rot.lirf_dataset'),
        m('prc.espera_rot.lirf_train'),
        m('prc.espera_rot.predict', '--pred', ARTIFACTS / 'pre_espera_final.parquet', '--out', ARTIFACTS / 'espera_final.parquet', '--months', '1,2,6,7'),
        m('prc.espera_rot.lirf_predict', '--pred', ARTIFACTS / 'espera_final.parquet', '--out', ARTIFACTS / 'lirf_final.parquet', '--months', '1,2,6,7'),
        m('prc.layers.lirf_night_rule', '--pred', ARTIFACTS / 'lirf_final.parquet', '--out', ARTIFACTS / 'night_final.parquet', '--months', '2,6', '--dose', '0.5'),
        m('prc.layers.lirf_eco_rule', '--pred', ARTIFACTS / 'night_final.parquet', '--out', ARTIFACTS / 'submission_final.parquet', '--months', '1,2,6,7', '--lo-h', '6', '--hi-h', '14', '--dose', '0.5'),
    ]


def stage_compose(a):
    return [
        m('prc.stack.apply_stack', '--dir', ARTIFACTS / 'reference' / 'stack_old', '--legacy-months', '1,7', '--out', ARTIFACTS / 'reference' / 'stack_old_pred.parquet'),
        m('prc.compose', '--stack', ARTIFACTS / 'stack_full_pred.parquet', '--ref-final', ARTIFACTS / 'reference' / 'v78_janjul.parquet',
          '--jj-new', ARTIFACTS / 'stack_full_pred.parquet', '--jj-old', ARTIFACTS / 'reference' / 'stack_old_pred.parquet',
          '--jj-add', ARTIFACTS / 'stack_inputs' / 'd_h19.parquet', ARTIFACTS / 'stack_inputs' / 'd_pool.parquet', ARTIFACTS / 'stack_inputs' / 'd_lirf_fisico.parquet', ARTIFACTS / 'stack_inputs' / 'd_row_adjustments_removed.parquet',
          '--jj-set', ARTIFACTS / 'stack_inputs' / 'set_model_values_on_adjusted_rows.parquet', '--out', ARTIFACTS / 'pre_espera_final.parquet'),
    ]


def cv_basea(a):
    return [m('prc.basea.residual_chain_stack_adsb', '--upto', 'v12', '--threads', a.threads)]


def cv_baseb(a):
    return [m('prc.baseb', 'holdout', 'prc/baseb/configs/h045_champion_integration.yaml')]


def cv_layers(a):
    return [p('prc/layers/camadas_nao_adsb_final.py', '--pred', V36, '--eval-ltfm', '--out', ARTIFACTS / 'layers' / 'camadas_nao_adsb_eval.parquet')]


def cv_otimista(a):
    return [m('prc.otimista.lmo_final', '--pred', V36, '--out', WORK / 'lmo_final.csv', '--months', '2,6', '--jobs', a.jobs)]


def cv_espera(a):
    return [m('prc.espera_rot.cv', '--threads', a.threads), m('prc.espera_rot.lirf_cv')]


def cv_stack(a):
    return [m('prc.stack.perm', '--dir', ARTIFACTS / 'reference' / 'stack_old', '--no-nets')]


TRAIN = {'features': stage_features, 'basea': stage_basea, 'baseb': stage_baseb, 'blend': stage_blend, 'layers': stage_layers,
         'otimista': stage_otimista, 'v53': stage_v53, 'stack': stage_stack, 'espera': stage_espera, 'stack_full': stage_stack_full}
CV = {'basea': cv_basea, 'baseb': cv_baseb, 'layers': cv_layers, 'otimista': cv_otimista, 'stack': cv_stack, 'espera': cv_espera}
PREDICT = {'blend': stage_blend, 'compose': stage_compose, 'espera': stage_espera}


def run(table, names, args):
    for name in names:
        for cmd in table[name](args):
            print(' '.join(cmd), flush=True)
            if not args.dry_run:
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get('PYTHONPATH')])))
                subprocess.run(cmd, check=True, cwd=REPO, env=env)
