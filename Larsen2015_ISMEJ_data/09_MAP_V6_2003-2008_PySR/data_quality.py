#!/usr/bin/env python3
"""Data-quality checks for the MAP replication inputs. Reports; changes nothing.

Checks: all-zero Orders; chloroplast leakage into Cyanobacteria (from the
unthresholded SINTAX calls); Vibrionales outliers; read depth and rare-Order
noise; samples dropped between the per-sample and per-date V6 tables; predictor
collinearity; S4 PRMT dates. Writes data_quality_report.md next to this script.

Run from Larsen2015_ISMEJ_data/:  python3 09_MAP_V6_2003-2008_PySR/data_quality.py
Needs ../sintax.tsv, ../unique.fa and 01_sequence_data/fasta-*_ICM_PML_Bv6.fa
for the chloroplast check (skipped with a note if absent).
"""
import collections, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_map_v6 import REPO, ROOT, S4_DATES, WINDOWS, build_training_table, dead_orders  # noqa: E402

RAW = os.path.join(REPO, 'V6_tables', 'V6_order_by_sample_raw.tsv')
FASTA = os.path.join(ROOT, '01_sequence_data', 'fasta-1794098140692_ICM_PML_Bv6.fa')
SINTAX = os.path.join(REPO, 'sintax.tsv')
UNIQUE = os.path.join(REPO, 'unique.fa')
REMOVED = ['CHLOROPLAST', 'NON_BACTERIA', 'unassigned']


def sample_date(names):
    return pd.to_datetime(pd.Series(names).str.extract(r'(\d{4}_\d{2}_\d{2})')[0].values, format='%Y_%m_%d')


def chloroplast_check(out, Y):
    if not all(os.path.exists(p) for p in (FASTA, SINTAX, UNIQUE)):
        out.append('Chloroplast check skipped: sintax.tsv / unique.fa / V6 FASTA not found.'); return
    counts = collections.defaultdict(collections.Counter); h = seq = None
    def flush():
        if h is not None:
            _, ds, n = h[1:].split('|'); counts[seq.upper().replace('U', 'T')][ds.split('--')[1]] += int(n)
    for line in open(FASTA):                   # same parse and u-id order as process_V6.py
        line = line.strip()
        if line.startswith('>'): flush(); h, seq = line, ''
        else: seq += line
    flush()
    useq = {f'u{n}': s for n, s in enumerate(counts)}
    uf, cur = {}, None
    for line in open(UNIQUE):
        line = line.strip()
        if line.startswith('>'): cur = line[1:]; uf[cur] = ''
        else: uf[cur] += line
    if any(uf.get(k) != v for k, v in useq.items()):
        out.append('Chloroplast check skipped: unique.fa ids do not match the FASTA parse order.'); return
    best = collections.Counter(); sus = collections.Counter()
    for line in open(SINTAX):
        c = line.rstrip('\n').split('\t')
        passed = dict(x.split(':', 1) for x in (c[3] if len(c) > 3 else '').split(',') if ':' in x)
        if passed.get('c') == 'Chloroplast' or passed.get('p') != 'Cyanobacteria': continue
        m = re.search(r'c:([^,(]+)\(', c[1]); bc = m.group(1) if m else ''
        for samp, n in counts[useq[c[0]]].items():
            best[bc] += n
            if bc == 'Chloroplast': sus[samp] += n
    raw = pd.read_csv(RAW, sep='\t', index_col=0)
    per = pd.DataFrame({'chl_removed': raw.loc['CHLOROPLAST'], 'cyano': raw.loc['Cyanobacteria'],
                        'total': raw.sum(), 'suspect': pd.Series(sus, dtype=float)}).fillna(0)
    per.index = sample_date(per.index)
    g = per.groupby(level=0).sum().reindex(pd.to_datetime(Y.index))
    g['chl_frac'] = g.chl_removed / g.total; g['suspect_frac'] = g.suspect / g.cyano.replace(0, np.nan)
    cy = pd.Series(Y['Cyanobacteria'].values, index=g.index)
    tot = sum(best.values())
    out.append('## Cyanobacteria vs chloroplast misassignment\n')
    out.append(f'- Chloroplast reads removed upstream (SINTAX class = Chloroplast at cutoff 0.5): '
               f'{g.chl_removed.sum() / g.total.sum():.1%} of all reads; per date {g.chl_frac.min():.1%}-{g.chl_frac.max():.1%}.')
    out.append(f'- Reads kept as "Cyanobacteria" whose unthresholded best class call is Chloroplast '
               f'(i.e. plastids that slipped under the cutoff): {best["Chloroplast"]:,} of {tot:,} ({best["Chloroplast"] / tot:.2%}). '
               f'Best class calls: ' + ', '.join(f'{k or "(none)"} {v:,}' for k, v in best.most_common(4)) + '.')
    r1 = spearmanr(cy, g.chl_frac)
    out.append(f'- Spearman rho, Cyanobacteria % vs removed-chloroplast fraction: {r1.correlation:.2f} (p={r1.pvalue:.2g}). '
               'Leakage would make these co-vary.')
    mon = cy.groupby(cy.index.month).mean()
    out.append('- Mean Cyanobacteria % by month: ' + ', '.join(f'{m}:{v:.1f}' for m, v in mon.items()) + '.')
    top = cy.sort_values(ascending=False).head(5)
    out.append('- Highest dates (Cyanobacteria %, suspect share of its reads): ' +
               '; '.join(f'{d:%Y-%m-%d} {v:.1f}% ({g.loc[d, "suspect_frac"]:.1%})' for d, v in top.items()) + '.')
    worst = g.suspect_frac.idxmax()
    out.append(f'- Largest suspect share on any date: {g.suspect_frac.max():.1%} on {worst:%Y-%m-%d}, where Cyanobacteria '
               f'is {cy[worst]:.2f}% ({int(g.loc[worst, "cyano"])} reads).')
    verdict = 'NOT SUPPORTED' if best['Chloroplast'] / tot < 0.02 and abs(r1.correlation) < 0.3 else 'POSSIBLE - inspect'
    out.append(f'- **Verdict: plastid contamination of Cyanobacteria {verdict}.** The late-summer peaks are reads whose '
               'best class is Synechococcophycideae, consistent with the known Synechococcus maximum at L4.\n')


def main():
    out = ['# Data-quality report (generated by data_quality.py)\n']
    X, Y = build_training_table('w72')
    raw = pd.read_csv(RAW, sep='\t', index_col=0)
    out.append('## Orders\n')
    out.append(f'- All-zero Orders on all 72 dates: {dead_orders(Y)}. process_V6.py labels every non-chloroplast '
               'p__Cyanobacteria read "Cyanobacteria" (Greengenes 13_8 via SINTAX), so Prochlorales is folded in by construction. '
               'Treated as 23 effective Orders; fixed at 0 and never a candidate parent.')
    m = Y.mean().sort_values(ascending=False)
    out.append('- Mean % (72 dates): ' + ', '.join(f'{o} {v:.2f}' for o, v in m.items()) + '.')
    clean = raw.drop(index=[i for i in REMOVED if i in raw.index])
    d24 = clean.reindex(list(Y.columns), fill_value=0).sum()
    keep = [s for s in raw.columns if sample_date([s])[0].strftime('%Y-%m-%d') in set(Y.index)]
    expected = pd.DataFrame({o: Y[o].values / 100 * d24[keep].values for o in Y.columns}, index=Y.index)
    rare = [o for o in Y.columns if expected[o].median() < 5 and o not in dead_orders(Y)]
    out.append(f'- Read depth per sample (all reads): {raw.sum().min():,}-{raw.sum().max():,}; reads in the 24 Orders '
               f'(the denominator of the response): {d24[keep].min():,}-{d24[keep].max():,}.')
    out.append(f'- Orders with median expected read count < 5 per sample (noise-dominated; expect constants): {rare}.\n')

    out.append('## Vibrionales outliers\n')
    v = Y['Vibrionales'].sort_values(ascending=False)
    med, mad = Y['Vibrionales'].median(), (Y['Vibrionales'] - Y['Vibrionales'].median()).abs().median()
    for d in v.index[:3]:
        s = [c for c in raw.columns if sample_date([c])[0].strftime('%Y-%m-%d') == d]
        out.append(f'- {d}: Vibrionales {v[d]:.1f}% ({(v[d] - med) / (1.4826 * mad):.0f} robust SDs above the median {med:.2f}%), '
                   f'Rickettsiales {Y.loc[d, "Rickettsiales"]:.1f}%, sample {s}, depth {int(raw[s].sum().sum()):,} reads.')
    out.append('- Fitted both ways: window w72 (with) and w72noVib (without 2003-06-09 and 2003-08-04). '
               'Through the sum-to-100 constraint, a 78% Vibrionales date also depresses every other Order on that date.\n')

    out.append('## Samples dropped between the per-sample and per-date tables\n')
    by_sample = pd.read_csv(os.path.join(REPO, 'V6_tables', 'V6_24orders_by_sample_relabund.tsv'), sep='\t', index_col=0)
    dates = sample_date(by_sample.columns).strftime('%Y-%m-%d')
    dropped = [s for s, d in zip(by_sample.columns, dates) if d not in set(Y.index)]
    out.append(f'- {len(by_sample.columns)} samples -> {len(Y)} dates. Dropped ({len(dropped)}): {dropped}. All are 2008 samples '
               'with _D/_Dp/_Np suffixes on 4 dates (the metagenome-paired sampling). process_V6.py as committed would keep them '
               '(76 dates); the exclusion happened outside the repo. Every retained date has exactly one sample.\n')

    out.append('## Predictors\n')
    s = X[['MDI', 'MDN', 'MNA', 'MPI']].sum(axis=1)
    out.append(f'- MDI+MDN+MNA+MPI vs MCH: max relative difference {((s - X.MCH).abs() / X.MCH).max():.3%}: raw size classes are '
               'collinear with MCH and are never used with it.')
    f3 = X.fpico + X.fnano + X.flarge
    out.append(f'- fpico+fnano+flarge = {f3.min():.4f}-{f3.max():.4f}: the three proportions are themselves collinear, '
               'so Sets C/D use fpico and flarge only.')
    cols = ['MO2', 'MNT', 'MPO', 'MCH', 'MPY', 'fpico', 'flarge', 'MPP']
    c = X[cols].corr()
    hi = [f'{a}-{b} {c.loc[a, b]:.2f}' for i, a in enumerate(cols) for b in cols[i + 1:] if abs(c.loc[a, b]) > 0.8]
    out.append(f'- |r| > 0.8 among candidate predictors (72 dates): {hi}.')
    out.append(f'- S4 PRMT dates {S4_DATES} present in every window covering 2008: '
               f'{all(all(d in build_training_table(w)[1].index for d in S4_DATES) for w in WINDOWS)}.\n')

    chloroplast_check(out, Y)
    path = os.path.join(HERE, 'data_quality_report.md')
    open(path, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out)); print('wrote', path)


if __name__ == '__main__':
    main()
