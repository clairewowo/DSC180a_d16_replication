#!/usr/bin/env python3
"""Evaluate a MAP equation table on a table of inputs.

Equation table: TSV in the format of MAP_equations_Larsen2012_SuppTable1.tsv
(taxon, abbrev, nonANN_equation, MAP_equation, MAP_parents, nonANN_r, MAP_r).
Equations use only + - * / ( ), numbers, the input column names, and (MAP form)
the 3-letter abbreviations of other Orders.

Inputs: TSV with one row per sample/date and one column per root variable the
equations use (published table: DO2 SRP NIT NH4 SIL CHL PAR SPC MPC LPC LDC SDC
BAC; retrained tables: whatever predictor names they were fitted with).

MAP form: Orders are evaluated in a topological order of the parent graph, and an
Order's parents are the PREDICTED values of its parent Orders (pass `observed` to
evaluate teacher-forced instead). nonANN form: inputs only.

Normalisation (the paper: a sample's community values sum to 100): non-finite
values -> 0, negatives -> 0, then each row rescaled to sum to 100. Rows that are
all zero stay zero. --raw skips this.

usage: python3 apply_map.py EQUATIONS.tsv INPUTS.tsv OUT.tsv [--form MAP|nonANN] [--raw]
"""
import argparse, re, sys
import numpy as np, pandas as pd

NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
ALLOWED = re.compile(r'^[\sA-Za-z0-9_.+\-*/()eE]*$')


def load_equations(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False).set_index('abbrev')


def names_in(eq):
    """Variable names in an equation (number exponents like 1e-5 are not names)."""
    eq = re.sub(r'(?<![A-Za-z_])\d+\.?\d*[eE][+-]?\d+', ' ', eq)
    return set(NAME.findall(eq))


def parents(eq, abbrevs):
    return sorted(names_in(eq) & set(abbrevs))


def topo_order(eqs, form='MAP'):
    col = f'{form}_equation'
    graph = {ab: parents(eqs.loc[ab, col], eqs.index) for ab in eqs.index}
    order, state = [], {}
    def visit(n, path):
        if state.get(n) == 1:
            raise ValueError('MAP graph has a cycle: ' + ' -> '.join(path + [n]))
        if state.get(n) == 2: return
        state[n] = 1
        for p in graph[n]: visit(p, path + [n])
        state[n] = 2; order.append(n)
    for n in sorted(graph): visit(n, [])
    return order


def _compile(eq):
    if not ALLOWED.match(eq):
        raise ValueError(f'unexpected characters in equation: {eq!r}')
    return compile(eq, '<equation>', 'eval')


def evaluate(eqs, X, form='MAP', observed=None):
    """Raw (unnormalised) predictions, DataFrame indexed like X, columns = taxon names.
    observed: optional DataFrame of observed Order values (columns = taxon names) used
    as parent values instead of predictions (teacher forcing)."""
    col = f'{form}_equation'
    taxon = eqs['taxon'].to_dict()
    env = {c: X[c].to_numpy(dtype=float) for c in X.columns}
    pred = {}
    order = topo_order(eqs, form) if form == 'MAP' else list(eqs.index)
    with np.errstate(all='ignore'):
        for ab in order:
            eq = eqs.loc[ab, col]
            ns = names_in(eq)
            missing = ns - set(env) - set(eqs.index)
            if missing:
                raise KeyError(f'{ab}: equation uses {sorted(missing)}, not in inputs')
            scope = {n: env[n] for n in ns if n in env}
            for p in ns & set(eqs.index):
                scope[p] = observed[taxon[p]].to_numpy(dtype=float) if observed is not None else pred[p]
            v = eval(_compile(eq), {'__builtins__': {}}, scope)
            pred[ab] = np.broadcast_to(np.asarray(v, dtype=float), (len(X),)).copy()
    return pd.DataFrame({taxon[ab]: pred[ab] for ab in eqs.index}, index=X.index)


def normalize(P):
    """Non-finite -> 0, negatives -> 0, rows rescaled to sum to 100."""
    A = P.to_numpy(dtype=float).copy()
    A[~np.isfinite(A)] = 0.0
    A[A < 0] = 0.0
    s = A.sum(axis=1, keepdims=True)
    A = np.divide(100 * A, s, out=np.zeros_like(A), where=s > 0)
    return pd.DataFrame(A, index=P.index, columns=P.columns)


def apply_map(eqs, X, form='MAP', raw=False, observed=None):
    P = evaluate(eqs, X, form, observed)
    return P if raw else normalize(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('equations'); ap.add_argument('inputs'); ap.add_argument('out')
    ap.add_argument('--form', choices=['MAP', 'nonANN'], default='MAP')
    ap.add_argument('--raw', action='store_true', help='skip clipping/renormalisation to 100')
    a = ap.parse_args()
    eqs = load_equations(a.equations)
    X = pd.read_csv(a.inputs, sep='\t', index_col=0)
    print('evaluation order:', ' '.join(topo_order(eqs, a.form)) if a.form == 'MAP' else '(independent)', file=sys.stderr)
    apply_map(eqs, X, a.form, a.raw).to_csv(a.out, sep='\t')
    print('wrote', a.out, file=sys.stderr)


if __name__ == '__main__':
    main()
