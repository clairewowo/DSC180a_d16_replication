# SUBSTITUTIONS — every deviation from Larsen et al. 2015 / Larsen, Field & Gilbert 2012

Layer 1 (MAP) of Larsen et al. 2015 (ISME J 9:166). The MAP equations themselves
come from Larsen, Field & Gilbert 2012 (Nat Methods), "the 2012 paper" below.
Each entry gives the original, what we did instead, and why it matters. Run-specific
events (cycle reversions, constant equations) are in `results/`.

## 1. Symbolic-regression engine: Eureqa 0.83 beta → PySR

| | Original | Here |
|---|---|---|
| Engine | Eureqa 0.83 beta (Nutonian; acquired by DataRobot in 2017, discontinued) | PySR 2.7.2, SymbolicRegression.jl 2.7.2, Julia 1.13.4, in venv `~/.venvs/pysr`. Search runs in float32 (PySR default); CV scoring and `apply_map.py` evaluate in float64. |
| Search budget | not stated | `niterations=60`, `populations=15`, `population_size=33` (as in the 08_ script) |
| Operators | + − × ÷ (the only ones in the published equations) | `binary_operators=['+','-','*','/']`, no unary |
| Loss | not stated (Eureqa default is squared error) | squared error (`L2DistLoss`) |
| Complexity cap | not stated | `maxsize=25`; denominators ≤ 9 nodes; `parsimony=0.001` |
| Choosing from the Pareto front | not stated | lowest 5-fold date-blocked CV MSE, with each candidate's constants refitted per fold; CV MSEs within 1% count as a tie and the simplest wins |
| Stochasticity | single run, presumably | 3 seeds (0, 1, 2) per configuration, `deterministic=True` |

Eureqa's search heuristics and its default selection rule can't be reproduced. A different equation found by PySR doesn't disprove the original.

## 2. Taxonomy: GAST/RDP (VAMPS) → Greengenes 13_8 via SINTAX

- **Correction to the brief:** the V6 tables were classified against **Greengenes 13_8 with vsearch SINTAX, cutoff 0.5** (`process_V6.py`), not Silva 119.
- `process_V6.py` labels every non-chloroplast `p__Cyanobacteria` read "Cyanobacteria". **Prochlorales is therefore 0 on every date by construction.** We fit 23 effective Orders; Prochlorales is fixed at 0 and is never a candidate parent. That makes the two published edges into Prochlorales (cya←pro, leg←pro) unrecoverable.
- Reads classified as chloroplast at class level (14.8% of all reads) are removed upstream. Leakage of plastids into Cyanobacteria was checked and is not supported: 0.68% of its reads have Chloroplast as their best class call. See `data_quality_report.md`.
- The response is each Order's % of reads in the 24 Orders (`RENORM_TO_24 = True`), so each date sums to 100, as in the paper.

## 3. Samples and dates excluded

| What | Count | Why | Where |
|---|---|---|---|
| 2008 samples with suffixes `_D`, `_Dp`, `_Np` on 2008-01-28, 04-22, 08-27, 08-28 | 10 samples, 4 dates | Metagenome-paired sampling, absent from `V6_24orders_by_date_relabund.tsv`. **The exclusion is not in the committed `process_V6.py`**, which would keep 76 dates. | upstream, undocumented |
| 2003-06-09 (Vibrionales 40%) and 2003-08-04 (Vibrionales 78%, Rickettsiales 0.9%) | 2 dates | Extreme outliers, possibly real blooms or a library artifact | window `w72noVib` only; `w72` keeps them |
| Prochlorales | 1 Order | all zero (§2) | all fits |

No other filtering. The S4 PRMT dates (2008-03-17, 06-23, 09-22, 12-08) are asserted present in every window. Note that the S4 archive's files are named MAR/JUN/**AUG**/DEC while the paper's text says September. 2008-08-20 and 2008-09-22 are both retained.

## 4. Environmental data: NCOF "SIM" → Copernicus reanalysis

- **Original:** 13 "SIM" fields from Met Office POLCOMS-ERSEM (MRCS), served by NCOF's Godiva2 ncWMS. They were weekly means for Jan 2007–Apr 2010 on a 0.05° grid; the 2015 paper says "reported for 1200 hours each day". They are unrecoverable.
- **Here:** Copernicus NWSHELF_MULTIYEAR_BGC_004_011 / PHY_004_009 (NEMO-ERSEM, 7 km, daily means) at L4 on the sampling date, from `Copernicus/L4_copernicus_by_date_push1.tsv`.
  - Despite its name, that table was built from `Copernicus/copernicus_outputs_03-08_push/`. The `copernicus_outputs_push1/` folder is 2009–2010.
  - The extraction method (which cell, any averaging) isn't recorded in the repo. A nearest-surface-cell spot check doesn't reproduce exact values (e.g. MNT on 2003-01-13: 7.42 in the table vs 8.16 at the nearest cell), so it's worth writing down how the table was made.
- These are different model systems, physics and resolutions, and daily rather than weekly averages. They are also a reanalysis, not the operational run the paper used.

## 5. Predictors: 13 SIM variables → 4 direct matches plus labelled proxies

| Paper (code, unit) | Here | Status |
|---|---|---|
| Dissolved O₂ (DO2, mmol m⁻³) | MO2 (mmol m⁻³) | direct |
| Phosphate (SRP, mmol m⁻³) | MPO (mmol m⁻³) | direct |
| Nitrate (NIT, mmol m⁻³) | MNT (mmol m⁻³) | direct |
| Chlorophyll a (CHL, labelled "mg C/m³" in Supp. Table 3) | MCH (mg chl m⁻³) | direct |
| Ammonium (NH4) | — | **absent**: 0 of 72 dates anywhere in the tree (2009–2010 only) |
| Silicate (SIL) | — | **absent**, same |
| PAR (W m⁻²) | — | **absent**, same (MODIS PAR on disk is 2009–2010) |
| Total bacterial carbon (BAC, mg C m⁻³) | — | **absent**, same. Note the paper's variable is *modelled bacterial carbon*, not cell counts. |
| Small, medium and large POC (SPC, MPC, LPC) | Set B: MPY (phytoplankton carbon, mmol C m⁻³); Sets C/D: fpico = MPI/MCH and flarge = (MDI+MDN)/MCH | **proxy**. Chlorophyll-based size fractions describe phytoplankton only; ERSEM POC also includes detritus. MPY is a bulk stock with no size classes. |
| Labile DOC (LDC) | Set D: MPP (net primary production, mg C m⁻³ d⁻¹) | **proxy** (a production rate standing in for a stock) |
| Semi-labile DOC/POC (SDC) | — | **unobtainable** |

Notes:
- **Label inconsistency.** The 2012 Supp. Table 3 says "semi-labile (SL)-DOC"; the 2012 Table 1/2 legends and the 2015 paper say "semi-labile POC". ERSEM has semi-labile *DOC*, which is most likely what was served.
- **Set C uses two size fractions, not three.** fpico + fnano + flarge = 1 (to within 0.07%), so fnano is dropped. The raw size-class chlorophylls are never used with MCH, since they sum to it within 0.068%.
- **Remaining collinearity** (72 dates): MNT–MPO r = 0.97, MCH–MPP 0.96, MCH–MPY 0.88. Parent attribution among these pairs is not identifiable.
- **Never used as predictors:** MSP (modelled pCO₂, held out as a validation target to avoid circularity), MPH, MSL, MLD, MKD. MTM (temperature) is only in the non-replication set T, which is defined in code but was not run.
- **What this means for comparison:** 19 of the 24 published non-ANN equations use at least one absent variable, and no published equation uses only the four direct matches. None of the published equations can be evaluated on these inputs. The comparison is of graph structure and skill, not of equations.

## 6. Training window and sample size

- **SIM-based MAP (the 2015 model):** trained on 2007–2008. Per the 2012 supplement it was trained on 2007 dates and validated on 8 held-out dates. Our replication fit is window `w24` (24 dates, 2007–2008).
- **The 2012 in-situ model** used all 72 dates: trained on 48 from 2003–2006 and validated on the 24 from 2007–2008, scoring BC 88.6. So the README's "72 samples" is right for the in-situ model and wrong for the SIM model.
- **Our `w72` and `w72noVib` windows** use these 72 dates with *Copernicus* predictors. That's a contribution, not a replication.
- **`w24` is severely underdetermined.** Each Stage-1 outer fold trains on about 19 dates. In Stage 2 each Order has 4–7 environmental inputs plus 22 other Orders as candidates. Only the out-of-fold metrics are meaningful.

## 7. Evaluation

- **The published number isn't a held-out score.** The 2015 paper reports "Bray–Curtis similarity 89.7 (s.d. 2.32) over the 24 monthly observations", which includes the training dates. The 2012 supplement gives the SIM model's held-out score as **73.9 on 8 dates** (p = 0.00159). The 2015 paper also twice calls 0.897 a "Bray–Curtis dissimilarity"; it is a similarity.
- **What we report:**
  - out-of-fold BC from nested date-blocked 5-fold CV, where every date is predicted by a model refitted without it, including cycle breaking;
  - in-sample BC of the all-dates fit, which is closest in kind to the 89.7;
  - a mean-profile baseline (predict the training mean community);
  - two permutation tests: the paper's day+taxa permutation, and a stricter day-only permutation.
- **Bray–Curtis similarity:** 100 × (1 − Σ|p−o| / Σ(p+o)) per date, averaged over dates. The paper's "weighted" is read as abundance-weighted.
- **Prediction post-processing:** non-finite values → 0, negatives → 0, then rescale each date to 100. The paper doesn't say how it handled negative predictions; there are many when the published equations are applied out of range.
- **Per-Order held-out r:** Pearson correlation on raw out-of-fold equation outputs. It is undefined (NA, or "const" in the tables) when the selected equation is constant.
- **Stage 2 is teacher-forced, as in the original.** Equations are fitted with *observed* parent Orders and evaluated end-to-end with *predicted* parents (`apply_map.py`). End-to-end MAP scores are therefore expected to be worse than per-Order fits suggest. That is faithful to the method, not a bug. Teacher-forced held-out scores are reported alongside for diagnosis.
- **Cycles:** if Stage 2's chosen equations form a cycle, a node in it reverts to its Stage-1 equation until the graph is acyclic. Every reversion, for every fold, is logged in `results/cycles_reverted.tsv`. Which node reverts is computed both ways from the same fits:
  - `dfs` (as briefed, and as in `08_`): the node where the depth-first search closes the cycle. These are the primary `MAP` results.
  - `mincost`: the node whose Stage-1 equation loses the least CV skill, (cvMSE₁ − cvMSE₂)/var(y). Reported as `MAP_mincost`.

## 7b. How the graph is built: BANJO → Stage-2 symbolic regression plus cycle breaking

- **Original.** The 2015 Methods say the Environmental Interaction Network (EIN) is a DAG "created using Bayesian network prediction methods (BANJO v 2.0.1)", with environmental variables as the only root nodes. Each Order is then a function of its parents in that DAG.
- **Here (as briefed).** The graph comes out of Stage 2: PySR may use any of the 22 other live Orders, and cycles are broken afterwards. No structure-learning step precedes the regression.
- **Consequence.** With 22 candidate parents, the Stage-2 graph is dense with cycles. In the full-budget calibration fit (set D, 72 dates) the `dfs` rule reverted 17 of the 23 live Orders to Stage 1, so "MAP" is mostly the non-ANN model. The DAG comparison should be read with this in mind.
- **A more faithful variant, not run.** Learn a DAG first (a BANJO-like score-based Bayesian-network search), then run PySR on each Order's learned parents. It needs a structure learner that isn't installed, and on 24 dates any learned structure would be very weakly identified.

## 8. Published equations and evaluator: reconstructed

- `STUDENT_GUIDE.md` said `03_MAP_model_NatMethods2012/MAP_equations_Larsen2012_SuppTable1.tsv` and `apply_map.py` existed. Neither was on disk or anywhere on this machine.
- Both were rebuilt in this session. The tables come from the supplement PDF's text layer (`extract_supp_tables_from_pdf.py`, exact text with no OCR), plus `MAP_parents_Larsen2012_SuppTable2.tsv`.
- **Checks:** all 48 equations parse; the variables in each MAP equation match the Supp. Table 2 parameter list for 24 of 24 Orders; and the graph's topological order is the published one (`pro bac spb ric leg pla des pse bur oce act alt cya rhi ent opi fla met ver rho rhs sph vib xan`).
- **Typos in the original Supp. Table 2:** "sgm" (Actinomycetales, in-situ column); CHL and NH4 listed twice in some rows.

## 9. Code changes vs `08_L4_2009-2010_replication/train_map_pysr.py`

1. `build_training_table()` was replaced; the old one read the 2009–2010 Qiita/V4 data.
2. In-sample `r_all` was replaced by out-of-fold r from nested CV, which is what the `nonANN_r`/`MAP_r` columns now hold.
3. Pareto selection loops over all 5 folds instead of one 80/20 split.
4. The unused `model_selection='best'` was removed.
5. **Bug fixed:** the parent regex only matched 3-character names, so the 5-character `fpico`/`flarge` would have been missed as parents.
6. Every cycle reversion is now logged with its cycle path.

The `08_` script itself is unchanged and still has bugs 2–5.
