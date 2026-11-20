#!/usr/bin/env python3
"""Fit layer 1 of Larsen et al. 2015 -- the MAP model of Larsen, Field & Gilbert
2012 -- with PySR, on the L4 2003-2008 16S V6 series and Copernicus NEMO-ERSEM
predictors. Derived from 08_L4_2009-2010_replication/train_map_pysr.py.

Procedure
  Stage 0  build_training_table(): y = the 24 MAP Orders as % (V6 fractions x 100,
           each date sums to 100); X = Copernicus variables at L4 for the same date,
           plus size-fraction proportions fpico, fnano, flarge of total chl.
  Stage 1  "non-ANN": per Order, PySR with + - * / only, squared error, maxsize
           cap, fixed seed. Choose from the Pareto front by date-blocked CV error
           (see cv_select), ties -> simpler.
  Stage 2  "MAP": refit allowing the other Orders (observed) as candidate inputs.
           Break any cycle by reverting a node in it to its Stage-1 equation.
  Output   Equation tables in the format of
           03_MAP_model_NatMethods2012/MAP_equations_Larsen2012_SuppTable1.tsv,
           evaluated by apply_map.py.

Changes vs 08_.../train_map_pysr.py
  - Stage 0 rewritten for the V6 + Copernicus tables (the old one read Qiita 1240).
  - Held-out r is genuine: nested date-blocked 5-fold CV. Each outer fold reruns
    Stage 1, Stage 2 and cycle breaking WITHOUT its test dates; nonANN_r / MAP_r in
    the equation tables are out-of-fold correlations (the old r_all was in-sample).
  - Pareto selection loops over all 5 folds: each front equation's constants are
    refitted on 4 folds and scored on the 5th; lowest mean CV MSE wins.
  - model_selection='best' removed; it was overridden by our selection anyway.
  - Parents are found by whole-name match (the old regex only saw 3-char names).
  - MAP held-out predictions are end-to-end (parents predicted, via apply_map),
    and also reported teacher-forced (parents observed) for diagnosis.
  - Prochlorales is all zero in this table; it is fixed at 0 and never a parent.

Commands (run from Larsen2015_ISMEJ_data/):
  python3 09_MAP_V6_2003-2008_PySR/train_map_v6.py stage0       # tables + checks, no PySR
  python3 09_MAP_V6_2003-2008_PySR/train_map_v6.py calibrate    # time one unit
  python3 09_MAP_V6_2003-2008_PySR/train_map_v6.py run --workers 6
  python3 09_MAP_V6_2003-2008_PySR/train_map_v6.py assemble     # metrics from finished units
`run` is resumable: finished units (one config x one fold) are skipped.
"""
import argparse, itertools, json, os, re, sys, tempfile, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # Larsen2015_ISMEJ_data/
REPO = os.path.dirname(ROOT)
V6 = os.path.join(REPO, 'V6_tables', 'V6_24orders_by_date_relabund.tsv')
COP = os.path.join(REPO, 'Copernicus', 'L4_copernicus_by_date_push1.tsv')
PUB_DIR = os.path.join(ROOT, '03_MAP_model_NatMethods2012')
PUB = os.path.join(PUB_DIR, 'MAP_equations_Larsen2012_SuppTable1.tsv')
sys.path.insert(0, PUB_DIR)
import apply_map  # noqa: E402

MAP_ORDERS = {  # paper's 24 Orders -> abbreviation (from Nat Methods Supp. Table 1)
 'Actinomycetales':'act','Alteromonadales':'alt','Bacillales':'bac','Burkholderiales':'bur',
 'Cyanobacteria':'cya','Desulfobacterales':'des','Enterobacteriales':'ent','Flavobacteriales':'fla',
 'Legionellales':'leg','Methylophilales':'met','Oceanospirillales':'oce','Opitutales':'opi',
 'Planctomycetales':'pla','Prochlorales':'pro','Pseudomonadales':'pse','Rhizobiales':'rhi',
 'Rhodobacterales':'rho','Rhodospirillales':'rhs','Rickettsiales':'ric','Sphingobacteriales':'spb',
 'Sphingomonadales':'sph','Verrucomicrobiales':'ver','Vibrionales':'vib','Xanthomonadales':'xan'}
ABBREVS = list(MAP_ORDERS.values())
DOMINANT = ['Rickettsiales', 'Flavobacteriales', 'Rhodobacterales', 'Oceanospirillales', 'Alteromonadales']

# Predictor sets. Copernicus codes: MO2 O2, MNT nitrate, MPO phosphate (mmol m-3),
# MCH chl (mg m-3), MPY phytoplankton carbon (mmol C m-3), MPP net primary production
# (mg C m-3 d-1); fpico = MPI/MCH, flarge = (MDI+MDN)/MCH. fnano = MNA/MCH is built but
# not used: the three fractions sum to 1 (+/-0.07%), so any one is redundant given two.
PREDICTOR_SETS = {
 'A': ['MO2', 'MNT', 'MPO', 'MCH'],                                   # direct matches only
 'B': ['MO2', 'MNT', 'MPO', 'MCH', 'MPY'],                            # + bulk carbon proxy for POC
 'C': ['MO2', 'MNT', 'MPO', 'MCH', 'fpico', 'flarge'],                # + size structure
 'D': ['MO2', 'MNT', 'MPO', 'MCH', 'fpico', 'flarge', 'MPP'],         # + PP as labile-DOC proxy
 'T': ['MO2', 'MNT', 'MPO', 'MCH', 'MTM'],    # NON-REPLICATION: temperature is not one of the 13
 'E': ['MO2', 'MNT', 'MPO', 'MCH', 'NH4', 'SIL', 'PAR', 'BAC'],   # Phase 2: needs EXTRA_PREDICTORS (STATUS.md)
}
# Phase 2: TSVs with a `date` column and any of NH4, SIL, PAR, BAC for the 72 dates.
# They are joined in Stage 0; nothing else in the pipeline changes.
EXTRA_PREDICTORS = []
NEVER_PREDICT = {'MSP', 'MPH', 'MSL', 'MLD', 'MKD', 'MDI', 'MDN', 'MNA', 'MPI'}  # MSP = validation target only
assert all(not (set(v) & NEVER_PREDICT) for v in PREDICTOR_SETS.values())

WINDOWS = {
 'w24': dict(years=(2007, 2008), drop=[], label='2007-2008, 24 dates: the paper\'s training window (replication)'),
 'w72': dict(years=(2003, 2008), drop=[], label='2003-2008, all 72 dates (contribution, not replication)'),
 'w72noVib': dict(years=(2003, 2008), drop=['2003-06-09', '2003-08-04'],
                  label='2003-2008 minus the two Vibrionales outlier dates (70 dates)'),
}
S4_DATES = ['2008-03-17', '2008-06-23', '2008-09-22', '2008-12-08']   # never filtered out
DEFAULT = dict(sets='ABCD', windows=['w24', 'w72', 'w72noVib'], seeds=[0, 1, 2])
RUNS = os.path.join(HERE, 'runs')


# ----------------------------------------------------------------------------- Stage 0
def build_training_table(window='w72'):
    """Stage 0. Returns (X, Y) indexed by date string.
    Y: the 24 MAP Orders as % of the 24-Order community (rows sum to 100).
    X: all 16 Copernicus columns plus fpico, fnano, flarge; predictor sets pick from it."""
    y = pd.read_csv(V6, sep='\t', index_col=0).T * 100
    X = pd.read_csv(COP, sep='\t', index_col='date')
    y.index = pd.to_datetime(y.index); X.index = pd.to_datetime(X.index)
    if not y.index.equals(X.index):
        raise ValueError('V6 and Copernicus tables are not on the same date index')
    if X.isna().any().any() or y.isna().any().any():
        raise ValueError('missing values in the input tables')
    if not np.allclose(y.sum(axis=1), 100, atol=1e-6):
        raise ValueError('V6 rows do not sum to 1.0 (x100)')
    y = y[list(MAP_ORDERS)]
    X = X.assign(fpico=X.MPI / X.MCH, fnano=X.MNA / X.MCH, flarge=(X.MDI + X.MDN) / X.MCH)
    for path in EXTRA_PREDICTORS:
        extra = pd.read_csv(path, sep='\t', index_col='date', parse_dates=True)
        X = X.join(extra.reindex(X.index))
    w = WINDOWS[window]
    keep = (y.index.year >= w['years'][0]) & (y.index.year <= w['years'][1]) & ~y.index.isin(pd.to_datetime(w['drop']))
    X, y = X[keep], y[keep]
    if X.isna().any().any():
        gaps = X.isna().sum(); raise ValueError(f'missing predictor values in window {window}: {gaps[gaps > 0].to_dict()}')
    if y.index.year.max() >= 2008:
        lost = [d for d in S4_DATES if pd.Timestamp(d) not in y.index]
        if lost: raise ValueError(f'S4 PRMT dates filtered out: {lost}')
    X.index = y.index = y.index.strftime('%Y-%m-%d')
    return X, y


def dead_orders(Y):
    """Orders that are 0 on every date of the window: fixed at 0, never parents."""
    return [o for o in MAP_ORDERS if (Y[o] == 0).all()]


# ----------------------------------------------------------------------------- PySR + selection
def pysr_model(args, seed, workdir):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=args.niterations, binary_operators=['+', '-', '*', '/'], unary_operators=[],
        maxsize=args.maxsize, populations=args.populations, population_size=args.population_size,
        elementwise_loss='L2DistLoss()',    # squared error, as Eureqa's default
        parsimony=args.parsimony, constraints={'/': (-1, 9)},   # keep denominators simple-ish
        random_state=seed, deterministic=True, parallelism='serial',
        progress=False, verbosity=0, output_directory=workdir)


def date_blocks(index, k=5):
    """Assign dates to k folds by cycling through time, so every fold spans the year."""
    order = np.argsort(pd.to_datetime(index)); folds = np.empty(len(index), int)
    folds[order] = np.arange(len(index)) % k
    return folds


def cv_mse(expr, feats, Xf, yf, folds):
    """Mean squared error of one Pareto-front expression under k-fold CV, with its
    constants refitted on the training folds (structure fixed). inf if unusable."""
    import sympy
    from scipy.optimize import least_squares
    by_name = {s.name: s for s in expr.free_symbols}
    syms = [by_name.get(f, sympy.Symbol(f)) for f in feats]
    consts = sorted(expr.atoms(sympy.Float), key=float)
    ps = sympy.symbols(f'p0:{len(consts)}') if consts else ()
    f = sympy.lambdify(syms + list(ps), expr.xreplace(dict(zip(consts, ps))), 'numpy')
    x0 = np.array([float(c) for c in consts])

    def pred(rows, p):
        with np.errstate(all='ignore'):
            v = f(*[Xf[rows, j] for j in range(Xf.shape[1])], *p)
        return np.broadcast_to(np.asarray(v, dtype=float), (int(rows.sum()),))

    sse, n = 0.0, 0
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        p = x0
        if len(x0):
            try:
                r = least_squares(lambda q: pred(tr, q) - yf[tr], x0, max_nfev=100 * (len(x0) + 1))
                if np.all(np.isfinite(r.x)): p = r.x
            except Exception:
                pass                                   # singular start: score PySR's own constants
        res = pred(va, p) - yf[va]
        if not np.all(np.isfinite(res)): return float('inf')
        sse += float(res @ res); n += int(va.sum())
    return sse / n


def cv_select(model, feats, Xf, yf, folds, tie_tol):
    """Among Pareto-front equations, pick the lowest CV MSE over ALL folds; any within
    tie_tol (relative) of the best count as tied and the simplest of those wins."""
    cands = []
    for i, row in model.equations_.iterrows():
        try: m = cv_mse(row['sympy_format'], feats, Xf, yf, folds)
        except Exception: m = float('inf')
        # sympy's printing, not PySR's Julia string: Julia prints small Float32 constants as 1.0f-5
        cands.append(dict(complexity=int(row['complexity']), equation=str(row['sympy_format']),
                          train_loss=float(row['loss']), cv_mse=m))
    ok = [c for c in cands if np.isfinite(c['cv_mse'])]
    if not ok:   # nothing on the front survives CV: fall back to the training mean
        return dict(complexity=1, equation=repr(float(np.mean(yf))), train_loss=float(np.var(yf)),
                    cv_mse=float('nan'), fallback='mean'), cands
    best = min(c['cv_mse'] for c in ok)
    tied = [c for c in ok if c['cv_mse'] <= best * (1 + tie_tol) + 1e-12]
    return min(tied, key=lambda c: (c['complexity'], c['cv_mse'])), cands


def fit_stage(X, Y, feats_for, folds, args, seed, stage, dead, workdir, log):
    res = {}
    for order, ab in MAP_ORDERS.items():
        if order in dead:
            res[ab] = dict(equation='0.0', complexity=1, cv_mse=0.0, parents=[], note='all-zero Order, fixed at 0', front=[])
            continue
        feats = feats_for(order)
        Xf = X[feats].to_numpy(dtype=float); yf = Y[order].to_numpy(dtype=float)
        m = pysr_model(args, seed, workdir)
        m.fit(Xf, yf, variable_names=feats)
        choice, front = cv_select(m, feats, Xf, yf, folds, args.tie_tol)
        res[ab] = dict(choice, parents=sorted(apply_map.names_in(choice['equation']) & (set(feats) | set(ABBREVS))), front=front)
        log(f'[{stage}] {order:20s} cx={choice["complexity"]:2d} cvMSE={choice["cv_mse"]:9.3f}  {choice["equation"][:90]}')
    return res


REVERT_RULES = ('dfs', 'mincost')
MAP_FORMS = {'MAP': 'dfs', 'MAP_mincost': 'mincost'}     # prediction label -> cycle-breaking rule
STAGE2_KEY = {'dfs': 'stage2', 'mincost': 'stage2_mincost'}


def break_cycles(res2, res1, rule='dfs', yvar=None):
    """Replace stage-2 equations by stage-1 ones until the Order graph is acyclic.
    rule 'dfs' (as briefed; as in 08_): revert the node at which the DFS closes the cycle.
    rule 'mincost': revert the node in the cycle whose Stage-1 equation loses the least
    CV skill, (cvMSE_stage1 - cvMSE_stage2) / var(y). Returns (new res2, [{node, cycle}])."""
    res2 = {k: dict(v) for k, v in res2.items()}
    fired = []
    while True:
        parents = {ab: [p for p in r['parents'] if p in res2] for ab, r in res2.items()}
        state, cyc = {}, None
        def dfs(n, path):
            nonlocal cyc
            state[n] = 1
            for p in parents[n]:
                if state.get(p) == 1: cyc = (p, path[path.index(p):] + [p]); return True
                if state.get(p) is None and dfs(p, path + [p]): return True
            state[n] = 2; return False
        for n in res2:
            if state.get(n) is None and dfs(n, [n]): break
        if not cyc: return res2, fired
        node, path = cyc
        if rule == 'mincost':
            def cost(n):
                c = (res1[n]['cv_mse'] - res2[n]['cv_mse']) / yvar[n]
                return c if np.isfinite(c) else float('inf')
            node = min(path[:-1], key=cost)
        fired.append(dict(node=node, cycle=' -> '.join(path)))
        res2[node] = dict(res1[node], reverted_from_stage2=True)


def equation_table(res1, res2, r1=None, r2=None):
    """DataFrame in the published Supp. Table 1 format, indexed by abbrev."""
    rows = []
    for order, ab in MAP_ORDERS.items():
        rows.append(dict(taxon=order, abbrev=ab, nonANN_equation=res1[ab]['equation'],
                         MAP_equation=res2[ab]['equation'],
                         MAP_parents=','.join(p for p in res2[ab]['parents'] if p in ABBREVS),
                         nonANN_r=_fmt(r1, order), MAP_r=_fmt(r2, order)))
    return pd.DataFrame(rows).set_index('abbrev')


def _fmt(r, order):
    if r is None: return 'NA'
    v = r.get(order, float('nan'))
    return 'const' if v == 'const' else ('NA' if not np.isfinite(v) else f'{v:.3f}')


def write_equations(eqt, path):
    eqt.reset_index()[['taxon', 'abbrev', 'nonANN_equation', 'MAP_equation', 'MAP_parents', 'nonANN_r', 'MAP_r']] \
       .to_csv(path, sep='\t', index=False)


# ----------------------------------------------------------------------------- work units
def config_name(set_name, window, seed):
    return f'{set_name}_{window}_s{seed}'


def unit_path(set_name, window, seed, fold, runs):
    return os.path.join(runs, config_name(set_name, window, seed), f'fold{fold}.json')


def run_unit(set_name, window, seed, fold, args):
    """One config x one fold ('full' = all dates, used for the published equation table).
    Outer folds 0-4 also store predictions for their held-out dates."""
    path = unit_path(set_name, window, seed, fold, args.runs)
    if os.path.exists(path): return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X, Y = build_training_table(window)
    dead = dead_orders(Y)
    outer = date_blocks(Y.index, 5)
    tr = np.ones(len(Y), bool) if fold == 'full' else outer != int(fold)
    inner = date_blocks(Y.index[tr], 5)
    feats = PREDICTOR_SETS[set_name]
    live = [MAP_ORDERS[o] for o in MAP_ORDERS if o not in dead]
    XY = X.join(Y.rename(columns=MAP_ORDERS))
    lines = []
    def log(s): lines.append(s)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as wd:
        res1 = fit_stage(X[tr], Y[tr], lambda o: feats, inner, args, seed, 'stage1', dead, wd, log)
        res2_raw = fit_stage(XY[tr], Y[tr], lambda o: feats + [a for a in live if a != MAP_ORDERS[o]],
                             inner, args, seed, 'stage2', dead, wd, log)
    yvar = {MAP_ORDERS[o]: float(Y.loc[tr, o].var()) for o in MAP_ORDERS}
    broken = {rule: break_cycles(res2_raw, res1, rule, yvar) for rule in REVERT_RULES}
    slim = {ab: {k: v for k, v in r.items() if k != 'front'} for ab, r in res2_raw.items()}
    out = dict(set=set_name, window=window, seed=seed, fold=str(fold), predictors=feats, dead_orders=dead,
               train_dates=list(Y.index[tr]), test_dates=list(Y.index[~tr]), stage1=res1,
               stage2=broken['dfs'][0], cycles_reverted=broken['dfs'][1],
               stage2_mincost=broken['mincost'][0], cycles_reverted_mincost=broken['mincost'][1],
               stage2_before_cycle_breaking=slim, seconds=round(time.time() - t0, 1), log=lines,
               pysr=dict(niterations=args.niterations, maxsize=args.maxsize, populations=args.populations,
                         population_size=args.population_size, parsimony=args.parsimony, tie_tol=args.tie_tol))
    if fold != 'full':
        te = ~tr
        out['pred'] = {'nonANN': apply_map.evaluate(equation_table(res1, res1), X[te], 'nonANN').to_dict(orient='index'),
                       'mean_profile': {d: Y[tr].mean().to_dict() for d in Y.index[te]}}
        for form, rule in MAP_FORMS.items():
            eqt = equation_table(res1, broken[rule][0])
            out['pred'][form] = apply_map.evaluate(eqt, X[te], 'MAP').to_dict(orient='index')
            out['pred'][f'{form}_teacher_forced'] = apply_map.evaluate(eqt, X[te], 'MAP', observed=Y[te]).to_dict(orient='index')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(out, f, indent=1, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, float) and not np.isfinite(o): return None
    return str(o)


def _worker(task):
    set_name, window, seed, fold, args = task
    os.environ.setdefault('PYTHON_JULIACALL_THREADS', '1')
    t0 = time.time()
    try:
        run_unit(set_name, window, seed, fold, args)
        return task[:4], round(time.time() - t0, 1), None
    except Exception as e:
        return task[:4], round(time.time() - t0, 1), f'{type(e).__name__}: {e}'


def tasks_for(args):
    folds = [0, 1, 2, 3, 4, 'full']     # window outermost: the w24 replication finishes first
    for w, seed, s, k in itertools.product(args.windows, args.seeds, args.sets, folds):
        if not os.path.exists(unit_path(s, w, seed, k, args.runs)):
            yield (s, w, seed, k, args)


def cmd_run(args):
    import concurrent.futures as cf, multiprocessing as mp
    todo = list(tasks_for(args))
    print(f'{len(todo)} units to run ({args.workers} workers); outputs in {args.runs}', flush=True)
    logf = open(os.path.join(args.runs, 'progress.log'), 'a')
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as ex:
        for i, (key, secs, err) in enumerate(ex.map(_worker, todo), 1):
            msg = f'{time.strftime("%H:%M:%S")} [{i}/{len(todo)}] {config_name(*key[:3])} fold {key[3]}: ' \
                  + (f'done in {secs}s' if err is None else f'FAILED after {secs}s: {err}')
            print(msg, flush=True); logf.write(msg + '\n'); logf.flush()


def cmd_calibrate(args):
    """Time one full unit (46 PySR fits) and extrapolate to the configured grid."""
    calib = os.path.join(args.runs, '_calibration')
    args_c = argparse.Namespace(**{**vars(args), 'runs': calib})
    t0 = time.time(); run_unit(args.sets[0], 'w72', 0, 'full', args_c); dt = time.time() - t0
    n = len(args.sets) * len(args.windows) * len(args.seeds) * 6
    print(f'one unit (46 PySR fits, 72 dates, set {args.sets[0]}, incl. Julia start): {dt/60:.1f} min')
    print(f'grid: {n} units -> ~{n * dt / 3600 / args.workers:.1f} h on {args.workers} workers (upper bound; '
          f'24-date units are faster)')


# ----------------------------------------------------------------------------- metrics
def bray_curtis(P, O):
    """Per-sample Bray-Curtis similarity x100 between rows of P and O (each sums to 100)."""
    P = np.asarray(P, float); O = np.asarray(O, float)
    den = (P + O).sum(axis=1)
    return 100 * (1 - np.abs(P - O).sum(axis=1) / np.where(den > 0, den, np.nan))


def permutation_p(P, O, n_perm=10000, seed=0, taxa=True):
    """Paper's test: permute predicted samples across days (and taxa within each sample)
    n_perm times; p = (1 + #null mean BC >= observed) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(P, float); O = np.asarray(O, float); n, T = P.shape
    obs = np.nanmean(bray_curtis(P, O)); hits = 0
    for start in range(0, n_perm, 500):
        b = min(500, n_perm - start)
        Pp = P[np.array([rng.permutation(n) for _ in range(b)])]          # b x n x T, days shuffled
        if taxa:
            idx = rng.permuted(np.broadcast_to(np.arange(T), (b, n, T)), axis=2)
            Pp = np.take_along_axis(Pp, idx, axis=2)
        den = (Pp + O).sum(axis=2)
        null = np.nanmean(100 * (1 - np.abs(Pp - O).sum(axis=2) / den), axis=1)
        hits += int((null >= obs).sum())
    return (1 + hits) / (1 + n_perm)


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0: return float('nan')
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def load_config(set_name, window, seed, runs):
    units = {}
    for k in [0, 1, 2, 3, 4, 'full']:
        p = unit_path(set_name, window, seed, k, runs)
        if not os.path.exists(p): return None
        units[str(k)] = json.load(open(p))
    return units


def assemble_config(set_name, window, seed, runs, n_perm):
    U = load_config(set_name, window, seed, runs)
    if U is None: return None
    X, Y = build_training_table(window)
    cfg = config_name(set_name, window, seed)
    maps = list(MAP_FORMS)                                   # 'MAP' (dfs rule, as briefed), 'MAP_mincost'
    scored = ['nonANN'] + maps
    forms = scored + [f'{m}_teacher_forced' for m in maps] + ['mean_profile']
    oof = {}
    for form in forms:
        rows = {}
        for k in '01234': rows.update(U[k]['pred'][form])
        oof[form] = pd.DataFrame.from_dict(rows, orient='index').reindex(index=Y.index, columns=list(MAP_ORDERS))
    full = U['full']
    stage = {'nonANN': full['stage1'], **{m: full[STAGE2_KEY[MAP_FORMS[m]]] for m in maps}}
    eqs = {m: equation_table(full['stage1'], stage[m]) for m in maps}
    ins = {'nonANN': apply_map.evaluate(eqs['MAP'], X, 'nonANN'), **{m: apply_map.evaluate(eqs[m], X, 'MAP') for m in maps}}
    const = {f: {o: stage[f][ab]['complexity'] == 1 for o, ab in MAP_ORDERS.items()} for f in scored}
    r = {f: {o: pearson(oof[f][o], Y[o]) for o in MAP_ORDERS} for f in forms if f != 'mean_profile'}
    po = pd.DataFrame([dict(config=cfg, set=set_name, window=window, seed=seed, taxon=o, abbrev=ab,
                            mean_pct=float(Y[o].mean()),
                            **{f'r_heldout_{f}': r[f][o] for f in r},
                            **{f'complexity_full_{f}': stage[f][ab]['complexity'] for f in scored},
                            **{f'constant_full_{f}': const[f][o] for f in scored},
                            **{f'absdiff_{f}': float(np.abs(apply_map.normalize(oof[f])[o] - Y[o]).sum()) for f in scored})
                       for o, ab in MAP_ORDERS.items()])
    for f in scored:
        po[f'error_share_{f}'] = po[f'absdiff_{f}'] / po[f'absdiff_{f}'].sum()
    summ = dict(config=cfg, set=set_name, window=window, seed=seed, n_dates=len(Y),
                predictors=','.join(PREDICTOR_SETS[set_name]), dead_orders=','.join(full['dead_orders']))
    for f in forms:
        P = apply_map.normalize(oof[f]); bc = bray_curtis(P, Y)
        summ[f'BC_heldout_{f}'] = float(np.nanmean(bc)); summ[f'BC_heldout_{f}_sd'] = float(np.nanstd(bc, ddof=1))
        if f in scored:
            summ[f'p_day_taxa_{f}'] = permutation_p(P, Y, n_perm, seed=seed)
            summ[f'p_day_only_{f}'] = permutation_p(P, Y, n_perm, seed=seed, taxa=False)
            summ[f'nonfinite_heldout_{f}'] = int((~np.isfinite(oof[f].to_numpy())).sum())
    for f in scored:
        bc = bray_curtis(apply_map.normalize(ins[f]), Y)
        summ[f'BC_insample_{f}'] = float(np.nanmean(bc)); summ[f'BC_insample_{f}_sd'] = float(np.nanstd(bc, ddof=1))
        summ[f'error_share_dominant5_{f}'] = float(po.loc[po.taxon.isin(DOMINANT), f'error_share_{f}'].sum())
        summ[f'n_constant_{f}'] = int(sum(const[f].values()))
    summ['abundance_share_dominant5'] = float(Y[DOMINANT].sum(axis=1).mean() / 100)
    out_eqs, edges, events = {}, [], []
    for m in maps:
        rule = MAP_FORMS[m]; ckey = 'cycles_reverted' + ('' if rule == 'dfs' else f'_{rule}')
        summ[f'{m}_cycles_reverted_full'] = ';'.join(c['node'] for c in full[ckey])
        summ[f'{m}_n_cycles_reverted_all_folds'] = sum(len(U[k][ckey]) for k in U)
        dag = dag_comparison(eqs[m], set_name)
        summ.update({f'{m}_{k}': v for k, v in dag.items() if k != 'edges'})
        edges.append(dag['edges'].assign(config=cfg, rule=rule))
        events += [dict(config=cfg, rule=rule, fold=k, node=c['node'], cycle=c['cycle']) for k in U for c in U[k][ckey]]
        eq = equation_table(full['stage1'], stage[m], r['nonANN'], r[m])
        for o, ab in MAP_ORDERS.items():   # 'const' rather than NA where the full-data equation is a constant
            if const['nonANN'][o]: eq.loc[ab, 'nonANN_r'] = 'const'
            if const[m][o]: eq.loc[ab, 'MAP_r'] = 'const'
        out_eqs[m] = eq
    oof_long = pd.concat({f: oof[f] for f in scored}, names=['form', 'date']).reset_index()
    return dict(summary=summ, per_order=po, equations=out_eqs, edges=pd.concat(edges),
                events=events, oof=oof_long.assign(config=cfg))


PUB_ENV = {'MO2': 'DO2', 'MNT': 'NIT', 'MPO': 'SRP', 'MCH': 'CHL',     # direct matches
           'NH4': 'NH4', 'SIL': 'SIL', 'PAR': 'PAR', 'BAC': 'BAC'}       # Phase 2, if supplied


def dag_comparison(eqt, set_name):
    """Compare the recovered Order->Order graph (full-data fit) with the published one."""
    pub = apply_map.load_equations(PUB)
    pub_edges = {(c, p) for c in pub.index for p in apply_map.parents(pub.loc[c, 'MAP_equation'], pub.index)}
    ours = {(c, p) for c in eqt.index for p in apply_map.parents(eqt.loc[c, 'MAP_equation'], eqt.index)}
    recoverable = {e for e in pub_edges if 'pro' not in e}            # pro is all-zero here
    hit = ours & pub_edges
    n_nodes = len(ABBREVS) - 1
    pub_roots = {c for c in pub.index if not apply_map.parents(pub.loc[c, 'MAP_equation'], pub.index)}
    our_roots = {c for c in eqt.index if not apply_map.parents(eqt.loc[c, 'MAP_equation'], eqt.index)}
    env_agree = env_total = 0
    available = {PUB_ENV[v] for v in PREDICTOR_SETS[set_name] if v in PUB_ENV}
    for c in eqt.index:   # environmental parents that have a direct counterpart in this set
        pe = apply_map.names_in(pub.loc[c, 'MAP_equation']) & available
        oe = {PUB_ENV[v] for v in apply_map.names_in(eqt.loc[c, 'MAP_equation']) if v in PUB_ENV}
        env_agree += len(pe & oe); env_total += len(pe)
    rows = [dict(child=c, parent=p, in_published=(c, p) in pub_edges, in_recovered=(c, p) in ours)
            for c, p in sorted(ours | pub_edges)]
    return dict(n_edges_recovered=len(ours), n_edges_published=len(pub_edges), n_edges_shared=len(hit),
                edge_precision=len(hit) / len(ours) if ours else float('nan'),
                edge_recall_recoverable=len(hit & recoverable) / len(recoverable),
                edge_precision_chance=len(recoverable) / (n_nodes * (n_nodes - 1)),
                roots_recovered=','.join(sorted(our_roots)), roots_shared_with_published=len(our_roots & pub_roots),
                env_parent_direct_matches=f'{env_agree}/{env_total}', edges=pd.DataFrame(rows))


def cmd_assemble(args):
    out = args.results; os.makedirs(os.path.join(out, 'equations'), exist_ok=True)
    S, P, E, EV, O = [], [], [], [], []
    for s, w, seed in itertools.product(args.sets, args.windows, args.seeds):
        a = assemble_config(s, w, seed, args.runs, args.n_perm)
        if a is None:
            print(f'skip {config_name(s, w, seed)}: not all 6 units finished'); continue
        for m, eq in a['equations'].items():   # MAP = dfs rule (as briefed); MAP_mincost alongside
            suffix = '' if MAP_FORMS[m] == 'dfs' else f'_{MAP_FORMS[m]}'
            write_equations(eq, os.path.join(out, 'equations', f'MAP_equations_{config_name(s, w, seed)}{suffix}.tsv'))
        S.append(a['summary']); P.append(a['per_order']); E.append(a['edges']); EV += a['events']; O.append(a['oof'])
        sm = a['summary']
        print(f"{config_name(s, w, seed):18s} BC held-out: MAP {sm['BC_heldout_MAP']:.1f}  MAP_mincost "
              f"{sm['BC_heldout_MAP_mincost']:.1f}  nonANN {sm['BC_heldout_nonANN']:.1f}  mean-profile "
              f"{sm['BC_heldout_mean_profile']:.1f} | in-sample MAP {sm['BC_insample_MAP']:.1f}", flush=True)
    if not S: return
    summ = pd.DataFrame(S); summ.to_csv(os.path.join(out, 'metrics_by_config.tsv'), sep='\t', index=False)
    pd.concat(P).to_csv(os.path.join(out, 'metrics_per_order.tsv'), sep='\t', index=False)
    pd.concat(E).to_csv(os.path.join(out, 'dag_edges.tsv'), sep='\t', index=False)
    pd.DataFrame(EV, columns=['config', 'rule', 'fold', 'node', 'cycle']).to_csv(
        os.path.join(out, 'cycles_reverted.tsv'), sep='\t', index=False)
    pd.concat(O).to_csv(os.path.join(out, 'heldout_predictions_raw.tsv'), sep='\t', index=False)
    num = [c for c in summ.select_dtypes('number').columns if c not in ('seed', 'n_dates')]
    spread = summ.groupby(['set', 'window'])[num].agg(['mean', 'std', 'min', 'max'])
    spread.columns = [f'{a}__{b}' for a, b in spread.columns]
    spread.to_csv(os.path.join(out, 'metrics_seed_spread.tsv'), sep='\t')
    print('wrote', out)


# ----------------------------------------------------------------------------- stage0 command
def cmd_stage0(args):
    out = os.path.join(HERE, 'training_tables'); os.makedirs(out, exist_ok=True)
    for w in WINDOWS:
        X, Y = build_training_table(w)
        cols = sorted(set(sum((PREDICTOR_SETS[s] for s in 'ABCDT'), [])), key=lambda c: (c.startswith('f'), c))
        X[cols].join(Y).to_csv(os.path.join(out, f'training_table_{w}.tsv'), sep='\t')
        print(f'{w:9s} {len(Y):2d} dates {Y.index[0]}..{Y.index[-1]}  dead Orders: {dead_orders(Y)}  '
              f'S4 dates kept: {all(d in Y.index for d in S4_DATES) if Y.index[-1] >= "2008" else "n/a"}')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser(description='MAP layer-1 replication with PySR (V6 2003-2008)')
    ap.add_argument('command', choices=['stage0', 'calibrate', 'run', 'assemble'])
    ap.add_argument('--sets', default=DEFAULT['sets'], help='predictor sets, e.g. ABCD (add T for the non-replication temperature set)')
    ap.add_argument('--windows', nargs='+', default=DEFAULT['windows'], choices=list(WINDOWS))
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT['seeds'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--niterations', type=int, default=60); ap.add_argument('--maxsize', type=int, default=25)
    ap.add_argument('--populations', type=int, default=15); ap.add_argument('--population-size', type=int, default=33)
    ap.add_argument('--parsimony', type=float, default=0.001)
    ap.add_argument('--tie-tol', type=float, default=0.01, help='relative CV-MSE margin treated as a tie')
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--quick', action='store_true', help='smoke test: 5 iterations, maxsize 12, runs_quick/')
    ap.add_argument('--runs', default=RUNS)
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()
    args.sets = list(args.sets)
    if args.quick:
        args.niterations, args.maxsize = 5, 12
        args.runs, args.results = os.path.join(HERE, 'runs_quick'), os.path.join(HERE, 'results_quick')
    os.makedirs(args.runs, exist_ok=True)
    dict(stage0=cmd_stage0, calibrate=cmd_calibrate, run=cmd_run, assemble=cmd_assemble)[args.command](args)


if __name__ == '__main__':
    main()
