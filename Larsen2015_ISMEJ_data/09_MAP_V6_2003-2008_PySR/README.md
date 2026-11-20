# MAP layer 1: predicting bacterial Orders from the environment

## What it is

MAP is a set of **24 formulas, one per bacterial Order**. Each one predicts what percentage of the L4 bacterial community that Order makes up on a given date. The formulas use **7 environmental variables** from the Copernicus ocean model, and some also use other Orders' predicted percentages. For example:

```
Cyanobacteria = 449.8 / MO2
```

We rebuilt the formulas with PySR (symbolic regression) on the 72 L4 dates from 2003–2008. This follows Larsen et al. (2012, 2015), but with our own 16S data and Copernicus inputs.

| Variable | Meaning | Unit |
|---|---|---|
| `MO2` | dissolved oxygen | mmol m⁻³ |
| `MNT` | nitrate | mmol m⁻³ |
| `MPO` | phosphate | mmol m⁻³ |
| `MCH` | chlorophyll | mg m⁻³ |
| `fpico` | fraction of chlorophyll from picophytoplankton (`MPI / MCH`) | 0–1 |
| `flarge` | fraction from diatoms + dinoflagellates (`(MDI + MDN) / MCH`) | 0–1 |
| `MPP` | net primary production | mg C m⁻³ d⁻¹ |

The 24 Orders are the ones in the paper (Rickettsiales, Rhodobacterales, Flavobacteriales, …). Prochlorales is always 0 in our data, so its formula is just `0`.

## How to run it

You need Python 3 with pandas and numpy. You don't need PySR. Run from the repo root.

**On the 72 training dates:**

```bash
python3 Larsen2015_ISMEJ_data/03_MAP_model_NatMethods2012/apply_map.py \
  Larsen2015_ISMEJ_data/09_MAP_V6_2003-2008_PySR/results/equations/MAP_equations_D_w72_s2.tsv \
  Larsen2015_ISMEJ_data/09_MAP_V6_2003-2008_PySR/training_tables/training_table_w72.tsv \
  predictions.tsv
```

**On your own dates:** make a tab-separated file. The first column is the date, followed by the 7 columns above. To build it from the Copernicus table:

```python
import pandas as pd
c = pd.read_csv('Copernicus/L4_copernicus_by_date_push1.tsv', sep='\t', index_col='date')
c['fpico'] = c.MPI / c.MCH
c['flarge'] = (c.MDI + c.MDN) / c.MCH
c[['MO2', 'MNT', 'MPO', 'MCH', 'fpico', 'flarge', 'MPP']].to_csv('my_inputs.tsv', sep='\t')
```

Then pass `my_inputs.tsv` in place of the training table.

**Output:** one row per date and one column per Order. The values are predicted percentages: negatives are set to 0 and each row is scaled to sum to 100.

**Options:**
- `--raw` keeps the unscaled formula values.
- `--form nonANN` uses the environment-only formulas, without the links between Orders. They score about the same.

**Other equation files:** `results/equations/` holds 72 tables named `MAP_equations_<set>_<dates>_s<seed>.tsv`.

- **Set** (which variables):
  - A: the first 4 variables.
  - B: A + phytoplankton carbon.
  - C: A + `fpico`, `flarge`.
  - D: all 7.
- **Dates:**
  - `w24`: 2007–08.
  - `w72`: 2003–08.
  - `w72noVib`: 2003–08 without the two Vibrionales-bloom dates.
- **Seed** (0–2): the random seed of the formula search.
- **`_mincost`:** an alternative rule for removing loops between Orders.

`MAP_equations_D_w72_s2` is the one to quote. It uses all 7 variables and the 72 dates, and it's the middle-scoring of the three seeds, not the best one.

## How accurate it is

We scored each version on **dates it wasn't trained on**: each fifth of the dates was hidden in turn. The score is Bray-Curtis similarity between the predicted and measured mix of the 24 Orders, from 0 (nothing in common) to 100 (identical). For comparison, "guess the average mix every time" is the baseline.

| Trained on | 4 variables | 7 variables | Guess the average |
|---|---|---|---|
| 2007–08 (24 dates) | 66 | 67 | **74** |
| 2003–08 (72 dates) | **78** | 76 | 73 |

The model to quote scores **76.2** on unseen dates and 81.2 on the dates it was trained on. For reference, the paper reports 73.9 on unseen dates and 89.7 on training dates.

**What this means:**
- **With 72 dates the model beats guessing the average,** but only by about 3–4 points. A permutation test says this isn't luck (p < 0.0001).
- **With only 24 dates it does worse than guessing the average,** because it memorises the training dates.
- **The 7 variables are no better than 4.** The links between Orders don't improve predictions on new dates.
- **The model gets each Order's typical abundance right, but most of the date-to-date ups and downs wrong.**

The correlation r between predicted and measured % on unseen dates, for the 7-variable model (1 = tracks perfectly, 0 = no relation):

| Order | Avg % | r | | Order | Avg % | r |
|---|---|---|---|---|---|---|
| Rickettsiales | 31.5 | 0.44 | | Rhodospirillales | 0.66 | **0.61** |
| Rhodobacterales | 19.4 | **0.50** | | Verrucomicrobiales | 0.51 | −0.18 |
| Flavobacteriales | 17.0 | 0.26 | | Rhizobiales | 0.47 | 0.03 |
| Oceanospirillales | 10.2 | 0.44 | | Enterobacteriales | 0.41 | 0.10 |
| Alteromonadales | 9.3 | 0.33 | | Xanthomonadales | 0.41 | 0.15 |
| Vibrionales | 2.4 | −0.03 | | Desulfobacterales | 0.34 | **0.76** |
| Cyanobacteria | 1.8 | 0.03 | | Legionellales | 0.30 | 0.19 |
| Methylophilales | 1.5 | 0.00 | | Sphingomonadales | 0.25 | 0.05 |
| Pseudomonadales | 1.3 | −0.03 | | Sphingobacteriales | 0.13 | 0.03 |
| Burkholderiales | 0.76 | −0.01 | | Planctomycetales | 0.02 | 0.07 |
| Actinomycetales | 0.71 | −0.14 | | Opitutales | 0.004 | 0.01 |
| Bacillales | 0.67 | −0.03 | | Prochlorales | 0 | always 0 |

**Caveats:**
- **Only a few Orders carry real signal:** Desulfobacterales, Rhodospirillales, and the five most abundant Orders (r 0.26–0.50). The rest are near 0.
- **Inputs outside the 2003–08 L4 range are extrapolation.**
- **Three equation tables blow up on a few dates:** `A_w72noVib_s1`, `B_w24_s1` and `B_w72_s2`. Use another seed for those.
- **We couldn't use four of the paper's variables:** ammonium, silicate, light and bacterial carbon. `SUBSTITUTIONS.md` lists every difference from the paper.

## More detail

- **`STATUS.md`:** full results, and how to add the four missing variables later.
- **`results/metrics_by_config.tsv`, `results/metrics_per_order.tsv`:** every score for all 36 versions.
- **Retraining:** `train_map_v6.py` retrains everything. It needs PySR + Julia and takes about 6 hours with 6 workers:

  ```bash
  cd Larsen2015_ISMEJ_data
  python 09_MAP_V6_2003-2008_PySR/train_map_v6.py run --workers 6
  python 09_MAP_V6_2003-2008_PySR/train_map_v6.py assemble
  ```
