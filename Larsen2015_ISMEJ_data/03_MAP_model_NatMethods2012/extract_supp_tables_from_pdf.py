#!/usr/bin/env python3
"""Rebuild Larsen, Field & Gilbert 2012 (Nat Methods) Supplementary Tables 1 and 2
as TSVs from the text layer of Larsen2012_NatMethods_Supplement_MAP_equations.pdf.

The PDF is a Word/Quartz export with a simple (MacRoman) text encoding, so the
equations can be read exactly from the Tj/TJ operators; no OCR is involved.
Table layout (all pages): taxon label x~77, abbreviation x~173,
non-ANN equation 208 <= x < 366, MAP equation x >= 366.

Outputs (next to this script):
  MAP_equations_Larsen2012_SuppTable1.tsv  taxon, abbrev, nonANN_equation,
      MAP_equation, MAP_parents, nonANN_r, MAP_r   (r not published -> NA)
  MAP_parents_Larsen2012_SuppTable2.tsv    taxon, abbrev, insitu_parameters,
      SIM_parameters   (as printed; Table 2 has typos, e.g. 'sgm')

Every equation is checked to parse with sympy and to use only the 13 SIM codes
and the 24 Order abbreviations; the MAP graph is checked to be acyclic.
Run:  python3 03_MAP_model_NatMethods2012/extract_supp_tables_from_pdf.py
"""
import os, re, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, 'Larsen2012_NatMethods_Supplement_MAP_equations.pdf')
ABBREV = ['act', 'alt', 'bac', 'bur', 'cya', 'des', 'ent', 'fla', 'leg', 'met', 'oce', 'opi',
          'pla', 'pro', 'pse', 'rhi', 'rho', 'rhs', 'ric', 'spb', 'sph', 'ver', 'vib', 'xan']
SIM = ['DO2', 'SRP', 'NIT', 'NH4', 'SIL', 'CHL', 'PAR', 'SPC', 'MPC', 'LPC', 'LDC', 'SDC', 'BAC']


def unescape(b):
    out = bytearray(); i = 0
    while i < len(b):
        c = b[i]
        if c == 0x5C and i + 1 < len(b):
            n = b[i + 1]
            if 0x30 <= n <= 0x37:
                j = i + 1
                while j < len(b) and j < i + 4 and 0x30 <= b[j] <= 0x37: j += 1
                out.append(int(b[i + 1:j], 8) & 0xFF); i = j; continue
            out.append({0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}.get(n, n)); i += 2; continue
        out.append(c); i += 1
    return bytes(out).decode('mac_roman')


TOKEN = re.compile(rb'\[((?:\\.|[^\]])*?)\]\s*TJ|\(((?:\\.|[^\\)])*)\)\s*Tj|'
                   rb'(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+Tm|'
                   rb'(-?[\d.]+)\s+(-?[\d.]+)\s+T[dD]|T\*', re.S)


def page_lines(stream):
    """[(y, [(x, text), ...]), ...] top to bottom, for one page content stream."""
    chunks = []; x0 = y = sc = 0.0
    for t in TOKEN.finditer(stream):
        if t.group(3) is not None:
            sc = float(t.group(6)); x0 = float(t.group(7)); y = float(t.group(8))
        elif t.group(9) is not None:
            x0 += float(t.group(9)) * sc; y += float(t.group(10)) * sc
        elif t.group(0) == b'T*':
            y -= sc * 1.2
        else:
            if t.group(2) is not None: txt = unescape(t.group(2))
            else: txt = ''.join(unescape(p) for p in re.findall(rb'\(((?:\\.|[^\\)])*)\)', t.group(1), re.S))
            chunks.append((round(y), round(x0), txt))
    lines = {}
    for yy, xx, tt in chunks: lines.setdefault(yy, []).append((xx, tt))
    return [(yy, sorted(lines[yy])) for yy in sorted(lines, reverse=True)]


def content_streams(data):
    out = []
    for m in re.finditer(rb'<<(.*?)>>\s*stream\r?\n', data, re.S):
        st = m.end(); en = data.find(b'endstream', st)
        try: s = zlib.decompress(data[st:en])
        except Exception: continue
        if b'BT' in s and b'Tf' in s and b'Image' not in m.group(1): out.append(s)
    return out


def table_rows(pages, columns, y_min=60):
    """Collect rows keyed by the 3-letter abbreviation at x~173. columns: [(name, x_lo, x_hi)]."""
    rows, cur = [], None
    for lines in pages:
        for y, items in lines:
            if y < y_min: continue                                   # page-number footer
            ab = [t.strip() for x, t in items if 165 <= x < 200 and t.strip() in ABBREV]
            if ab:
                cur = dict(abbrev=ab[0], taxon='', **{c: '' for c, _, _ in columns}); rows.append(cur)
            if cur is None: continue
            for x, t in items:
                if x < 165: cur['taxon'] += t.strip()
                for c, lo, hi in columns:
                    if lo <= x < hi: cur[c] += t
    for r in rows:
        for c, _, _ in columns: r[c] = re.sub(r'\s+', ' ', r[c]).strip()
    return rows


def parents(eq):
    return sorted(set(re.findall(r'\b[A-Za-z][A-Za-z0-9]{2}\b', eq)) & set(ABBREV))


def check(rows):
    import sympy
    loc = {n: sympy.Symbol(n) for n in SIM + ABBREV}
    for r in rows:
        for col in ('nonANN_equation', 'MAP_equation'):
            e = sympy.sympify(r[col], locals=loc)
            bad = {str(s) for s in e.free_symbols} - set(SIM + ABBREV)
            assert not bad, (r['abbrev'], col, bad)
            if col == 'nonANN_equation':
                assert not ({str(s) for s in e.free_symbols} & set(ABBREV)), (r['abbrev'], 'non-ANN uses an Order')
    graph = {r['abbrev']: parents(r['MAP_equation']) for r in rows}
    order, state = [], {}
    def visit(n):
        if state.get(n) == 1: raise SystemExit(f'cycle at {n}')
        if state.get(n) == 2: return
        state[n] = 1
        for p in graph[n]: visit(p)
        state[n] = 2; order.append(n)
    for n in sorted(graph): visit(n)
    return order


def main():
    data = open(PDF, 'rb').read()
    streams = content_streams(data)
    pages = {i + 1: page_lines(s) for i, s in enumerate(streams)}
    # Supp. Table 1 spans stream pages 7, 6, 5, 4 (streams are stored last page first)
    t1 = table_rows([pages[7], pages[6], pages[5], pages[4]],
                    [('nonANN_equation', 200, 366), ('MAP_equation', 366, 10000)])
    t1 = [r for r in t1 if r['abbrev'] in ABBREV]
    assert sorted(r['abbrev'] for r in t1) == sorted(ABBREV), [r['abbrev'] for r in t1]
    topo = check(t1)
    with open(os.path.join(HERE, 'MAP_equations_Larsen2012_SuppTable1.tsv'), 'w') as f:
        f.write('taxon\tabbrev\tnonANN_equation\tMAP_equation\tMAP_parents\tnonANN_r\tMAP_r\n')
        for r in sorted(t1, key=lambda r: r['taxon']):
            f.write(f"{r['taxon']}\t{r['abbrev']}\t{r['nonANN_equation']}\t{r['MAP_equation']}\t{','.join(parents(r['MAP_equation']))}\tNA\tNA\n")
    # Supp. Table 2 (stream page 3): in situ parents x 208..366, SIM parents x >= 366
    t2 = table_rows([pages[3]], [('insitu_parameters', 200, 366), ('SIM_parameters', 366, 10000)])
    # Layout quirk: Rhodospirillales' in-situ cell starts one line above its label, so
    # the row parser attaches 'NO3, SIL, TMP, NH4, SAL, CHL,' to Rhodobacterales.
    by = {r['abbrev']: r for r in t2}
    spill = 'NO3, SIL, TMP, NH4, SAL, CHL,'
    if by['rho']['insitu_parameters'].endswith(spill):
        by['rho']['insitu_parameters'] = by['rho']['insitu_parameters'][:-len(spill)].strip()
        by['rhs']['insitu_parameters'] = f"{spill} {by['rhs']['insitu_parameters']}"
    with open(os.path.join(HERE, 'MAP_parents_Larsen2012_SuppTable2.tsv'), 'w') as f:
        f.write('taxon\tabbrev\tinsitu_parameters\tSIM_parameters\n')
        for r in sorted(t2, key=lambda r: r['taxon']):
            f.write(f"{r['taxon']}\t{r['abbrev']}\t{r['insitu_parameters']}\t{r['SIM_parameters']}\n")
    print(f'Table 1: {len(t1)} Orders, all equations parse; MAP graph acyclic.')
    print('one valid topological order:', ' '.join(topo))
    print(f'Table 2: {len(t2)} rows')


if __name__ == '__main__':
    sys.exit(main())
