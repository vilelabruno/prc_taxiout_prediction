PB = dict(num_leaves=256, min_child_samples=20, n_estimators=1200, learning_rate=0.03, colsample_bytree=0.7)
PM = dict(num_leaves=64, min_child_samples=30, n_estimators=500, learning_rate=0.04)
ALLM = [12, 2, 3, 4, 5, 6, 8, 9, 10, 11]
POOL = ['EDDF', 'EDDM', 'EHAM', 'LSZH', 'LEBL', 'EGLL', 'LEMD']
WDIST = {0: 1.0, 1: .5, 2: .4, 3: .3, 4: .25, 5: .2, 6: .2}


def dist(M, T):
    d = abs(M - T)
    return min(d, 12 - d)


def wmap(T, months=None):
    return {mm: WDIST[dist(mm, T)] for mm in (ALLM if months is None else months)}


def allm(T):
    return [mm for mm in range(1, 13) if mm != T]
