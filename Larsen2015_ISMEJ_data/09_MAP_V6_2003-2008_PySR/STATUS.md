# STATUS — MAP layer 1, V6 2003–2008, PySR

## Phase 1 (four direct predictors plus labelled proxies)

| Piece | State |
|---|---|
| Stage 0 `build_training_table()` | done; windows w24 (24 dates), w72 (72), w72noVib (70) |
| Patched trainer `train_map_v6.py` | done; nested CV, 5-fold Pareto selection, end-to-end MAP evaluation |
| Published Supp. Tables 1–2 + `apply_map.py` | reconstructed from the PDF and verified (SUBSTITUTIONS §8) |
| Data-quality checks | `data_quality_report.md` |
| PySR install + smoke test | PySR 2.7.2 in `~/.venvs/pysr`; 5-iteration run passed end to end |
| Full grid: sets A–D × 3 windows × seeds 0–2 (36 configs, 216 units) | done 2026-11-18 00:01, 216/216, 0 failed; `assemble` done (10,000 permutations) |

### Phase 1 results (`results/`)

Held-out Bray-Curtis similarity: mean over 3 seeds, with the range in brackets. MAP uses predicted parents end to end (`dfs` cycle rule). The mean-profile baseline predicts every held-out date as the training-fold mean composition.

| Window | A (4 vars) | B (+MPY) | C (+fpico, flarge) | D (7 vars) | nonANN, A | Mean profile |
|---|---|---|---|---|---|---|
| w24 (2007–08) | 66.1 [63.8–68.1] | 68.2 [63.8–72.0] | 69.1 [68.4–69.7] | 67.0 [64.2–69.1] | 71.5 | **73.9** |
| w72 (2003–08) | **77.7** [77.7–77.9] | 77.3 [76.7–77.6] | 77.1 [76.2–77.7] | 76.5 [76.0–77.2] | 77.5 | 73.3 |
| w72noVib | 77.5 [77.3–77.7] | 77.6 [77.2–77.8] | **78.6** [78.1–79.1] | 77.0 [75.4–78.0] | 78.2 | 74.7 |

- **24 dates:** MAP does *worse* than the mean-profile baseline in every configuration. The day-only permutation test is not significant (p = 0.01–0.39), so with 24 dates the equations overfit.
- **72 dates:** MAP beats the baseline by about 3–4 BC points in every configuration, and both permutation tests give p < 0.0001.
- **The extra predictors don't help.** B, C and D are within seed noise of A on w72.
- **Order→Order links add no held-out skill.**
  - End-to-end MAP ≈ nonANN.
  - Teacher-forced MAP, which uses the true parent abundances, reaches 79.3 on w72 A. The loss comes from propagating errors in the predicted parents.
- **The paper's numbers:** held-out 73.9 is at our mean-profile baseline level; in-sample 89.7 compares with our 80–85.
- **Per-Order held-out r (w72):** only about 3–4 of 23 Orders exceed r > 0.5. These are Desulfobacterales (≈0.76), Rhodospirillales (≈0.6–0.7), Rhodobacterales (≈0.5) and sometimes Rickettsiales (≈0.45–0.5). Most rare Orders are near 0 (see `metrics_per_order.tsv`).
- **DAG vs the published graph:**
  - We recover 27–58 Order→Order edges, of which 0–6 are among the published 33.
  - Edge precision is 0–0.14 (mean 0.075), against a chance level of 0.061.
  - Cycle breaking reverts 12–17 of the 23 Orders (median 14) to their Stage-1 equation in each full fit (`cycles_reverted.tsv`).
  - The recovered graph is not distinguishable from chance agreement.
- **Known instability:** three full-fit equation tables blow up when evaluated end to end with predicted parents: A_w72noVib_s1 (in-sample BC 69.3), B_w24_s1 (76.0) and B_w72_s2 (75.6, mostly the 2003-08-04 Vibrionales date).
  - The causes are a Stage-2 denominator that crosses zero, or a fifth-power chain of Orders. Example: Opitutales = 4.1e-8·alt·oce⁴·…, giving 1.9e9 on 2007-04-23.
  - Held-out scores already include this effect. Prefer another seed's table for those configs.
- **7-variable model to quote:** `equations/MAP_equations_D_w72_s2.tsv`.
  - This is set D (MO2, MNT, MPO, MCH, fpico, flarge, MPP) on all 72 dates, using the median seed by held-out BC rather than the best one.
  - It scores held-out BC 76.2, in-sample BC 81.2, against the baseline's 73.3.

Reproduce from `Larsen2015_ISMEJ_data/`:

```bash
PY=~/.venvs/pysr/bin/python
$PY 09_MAP_V6_2003-2008_PySR/data_quality.py
$PY 09_MAP_V6_2003-2008_PySR/train_map_v6.py stage0
$PY 09_MAP_V6_2003-2008_PySR/train_map_v6.py run --workers 6      # resumable
$PY 09_MAP_V6_2003-2008_PySR/train_map_v6.py assemble
```

## Phase 2: if NH4, silicate, bacteria and PAR become available

This is a column addition. Nothing in the fitting, evaluation or DAG code changes.

1. **Supply the data.** Provide one TSV per source with a `date` column (YYYY-MM-DD) and columns named exactly `NH4`, `SIL`, `PAR`, `BAC`, covering the window's dates.
   - **Units, to match the paper:** NH4 and SIL in mmol m⁻³ (equal to µmol L⁻¹), PAR in W m⁻², BAC in mg C m⁻³.
   - Symbolic-regression constants absorb a linear unit change, but not a change of quantity. The paper's BAC is *modelled bacterial carbon*. Flow-cytometry counts would be a proxy and must be logged in SUBSTITUTIONS.md.
2. **Wire it in.** Add the TSV paths to `EXTRA_PREDICTORS` in `train_map_v6.py`. Stage 0 joins them by date.
   - Set **E** (= A + NH4, SIL, PAR, BAC) is already defined; trim it if only some variables arrive.
   - Combinations with the proxies, e.g. E + fpico + flarge + MPP, are one more line in `PREDICTOR_SETS`.
   - `PUB_ENV` already maps the four names, so the DAG comparison scores them as direct matches.
3. **Decide a gap policy.** Stage 0 deliberately raises if any predictor is missing inside a window. Either restrict the window or add imputation; `08_.../train_map_pysr.py --impute` is the precedent. Whichever you choose, log it.
4. **Run.** `run --sets E` (or `--sets ABCDE` to refit everything together), then `assemble`.

### What it buys

- **Wider comparison with the published graph.** The published non-ANN equations reference NH4/SIL/PAR/BAC 24 times, and 19 of 24 use at least one of them. Today no published equation has all of its environmental inputs available; with Phase 2, equations limited to {DO2, NIT, SRP, CHL, NH4, SIL, PAR, BAC} become comparable.
- **It still doesn't make the published equations evaluable.** The three POC classes, labile DOC and semi-labile DOC/POC remain absent or proxied. Even Phase 2 compares graph structure and skill, not the published formulas.
- **Mixed sources.** In-situ NH4 or Si next to modelled O₂/NO₃/PO₄ mixes observation with model. Say so, or prefer modelled NH4/Si if Copernicus has them (below).

### Where to look (not verified; this session had no internet)

- **PAR:** probably the quickest win. The MODIS-Aqua PAR file already on disk names its product "Aqua MODIS … 2003-2022 (8 Day Composite)". Rerun `08_.../03_satellite_WEC_2009-2010/fetch_satellite_wec.sh` for PAR only with 2003-01-01 to 2008-12-31, then match to L4 dates as `match_satellite_to_L4.py` does (±3 days).
- **NH4 and silicate:** the L4 in-situ nutrient series from the Western Channel Observatory / BODC. The 2009–2010 Qiita sheet shows the format PML supplies.
- **Bacteria:** the L4 flow-cytometry series from PML. Check how far before 2009 it goes.
- **Copernicus:** the fetch script never *requested* NH4 or Si. Run `copernicusmarine describe --contains NWSHELF_MULTIYEAR_BGC_004_011` to see whether they're offered. If they are, they're modelled values consistent with the other predictors.
