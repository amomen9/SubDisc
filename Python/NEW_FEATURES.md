# New Features

This file lists what the Python port can do that the Java SubDisc can not. The ported algorithms themselves are untouched: the result files of every setting in README.md are still byte for byte the same as Java's, and the old tests pass as is. Everything below sits next to them, in its own modules and windows.

Contents

1. Sorting and filtering in every table (GUI)
2. Random holdout split and the holdout test
3. MDL subgroup list
4. Multiple testing correction
5. Subsampling stability
6. Model error slicing
7. Survival target type (time-to-event)
8. Data files and tools
9. Tests and what changed in the ported code

Notation used in more than one section:

- **N** number of rows the mining run used (the selection, or the whole table)
- **S** a subgroup; **n** or **n_S** its coverage (number of rows it covers)
- **p_0** share of the target value in the data, **p_S** the share inside S
- **m** number of hypotheses (statistical tests) in a multiple testing correction
- **alpha** significance level, 0.05 unless said otherwise
- **log2** logarithm base 2, so lengths come out in bits

---

## 1. Sorting and filtering in every table (GUI)

### Introduction

Java has no sorting or filtering in its tables. We added it to every table with column headings: the result window, the pattern team tabs, browse, meta data, the ROC table and the cross table. The new windows of sections 2 to 6 have it too.

### How it works

- a click on a heading sorts on that column: ascending, then descending, then back to the original order. Sorting only changes the view, the **Nr.** of a subgroup stays the same
- the small arrow at the right end of a heading opens a filter box with a value field and =, >, <, >=, <=, include (the cell contains the text) or exclude (it does not)
- numbers are compared as numbers, text ignores upper / lower case; filters on more columns must all hold
- a column filter can have a second condition joined with **and** / **or**, e.g. Coverage >= 100 and <= 500, or Conditions include age or include sex. The second one only counts when its value field is not empty
- the filters of a table can be saved as a named preset (Presets in the filter box, or a right click on any heading) and applied again later, also after a restart. Columns are matched by their heading; a preset column the table does not have is reported
- presets made in another kind of table (e.g. a browse window) show up under **From other tables** in the presets menu
- the presets are kept on this computer only, in the same place as the light / dark choice (under HKCU\Software\SubDisc on Windows). Result and autorun files stay as Java writes them
- under a filtered table a line says e.g. "Showing 122 of 1,000 rows, filter: age >= 50 and < 60" with a link that clears the filters

### Filters that change more than the table

- in a ROC window the table filter also changes the picture: the curve, the points and the area under the curve are drawn again from the subgroups that pass, and the title says how many that are
- in the result window the filter carries over: ROC, Pattern Team, Save, Print, the p-value buttons, the regression test and the four new buttons of sections 2 to 5 use only the subgroups still shown. A line under the table says how many
- in the browse window Save writes only the rows the filter leaves

### Example

adult.txt, the single nominal run of section 3 (Cortana Quality, depth 1, 10 subgroups with coverages 443, 376, 327, 616, 268, 124, 671, 166, 50 and 124). A filter Coverage >= 300 on the result window leaves 5 of the 10. ROC then draws its curve from those 5, and Multiple Testing corrects 5 p-values, not 10.

---

## 2. Random holdout split and the holdout test

### Introduction

A subgroup is a description (age >= 33 AND sex = 'Male'), so it can be applied to rows it was never mined on. If its quality holds up there it is real; if it falls apart it was partly luck. Java can cross-validate, but it never checks a found description on unseen rows.

### Where

- **Tools > Random Holdout Split...** in the main window keeps a random fraction of the rows out (30% by default, with a seed so the split can be repeated). Mining then runs on the other rows as a selection. It replaces a selection made with Select Subset
- **Holdout Test** in the result window (Alt+O) asks for the test data:
  - the rows of the table that were not in the mining selection. With Select Subset this is the rest of the data, so mining on sex = 'Male' and testing on the women works too
  - or a test file with the same columns in the same order. It is loaded with the column types of the training data, and a column that comes out with another type is reported

Every subgroup is then scored on the test rows with the same quality measure; nothing is mined again. The window shows train and test coverage, train and test quality, the drop and the 4th result column (e.g. target share) on both sides.

### What it computes

    drop(S)          = quality_train(S) - quality_test(S)
    rank correlation = Spearman correlation of the train and test qualities

The quality on the test rows uses the base statistics (p_0, mean, ...) of the test rows, so a measure like WRAcc means the same thing on both sides. A rank correlation near 1 says the order of the subgroups also holds.

### Example

adult.txt, WRAcc, depth 2, top 20, a 70 / 30 split with seed 1:

| | train (700 rows) | test (300 rows) |
|---|---|---|
| mean quality | 0.0874 | 0.0934 |
| rank correlation | | 0.81 |

So the subgroups hold. And the same run on a column of pure random noise (30% ones) as target: mean quality 0.0161 on train and -0.0001 on test, rank correlation 0.43, and half of the subgroups end at 0 or below. That is what luck looks like.

With the full data files of section 8 (32,561 training rows, 16,281 test rows) the 50 depth-2 subgroups of WRAcc keep their quality almost exactly (0.0793 against 0.0790, rank correlation 0.99). That is the reason for the bigger files: on 300 test rows the numbers jump around much more.

---

## 3. MDL subgroup list

### Introduction

The top-k of a mining run overlaps a lot. On adult.txt the top 2 are **marital-status = 'Married-civ-spouse'** (443 rows) and **relationship = 'Husband'** (376 rows), and every single Husband row is also a married row; the second subgroup adds nothing. A subgroup list fixes that: it is an ordered list S_1, ..., S_k plus a default rule, and a row belongs to the first subgroup that covers it. The list is picked with the Minimum Description Length (MDL) principle: a subgroup only goes in when describing it costs fewer bits than it saves on encoding the target. The default rule always uses the distribution of the whole data, so a subgroup has to stand out from the data, not from the rest of the list. This is the idea of the SSD++ algorithm (Proença, Grünwald, Bäck and van Leeuwen, "Robust subgroup discovery", 2022); the code is our own.

### Where

**Subgroup List** in the result window (Alt+L), for single nominal and single numeric targets. The candidates are the subgroups shown, so it helps to mine with a larger maximum number of subgroups first. Settings: absolute or normalised gain, maximum list length and a minimum number of new rows per subgroup. **Test on Held Out Data...** in the list window scores the list on held out rows or a test file (section 2), with the parameters learnt on the training rows.

### What it computes

For a candidate S, R are the rows it would take over (covered by S and by no subgroup higher in the list) and n = |R|.

The bits saved on the target, binary target:

    saved(S) = n * KL(p_S || p_0) - 1/2 log2(n)
    KL(p || q) = p log2(p / q) + (1 - p) log2((1 - p) / (1 - q))

KL is the Kullback-Leibler divergence: the extra bits per row paid for coding rows with share p using a code made for share q. The 1/2 log2(n) is the price of sending the estimated p_S along.

Numeric target (mu_0, sigma_0 of the data; mu_S, sigma_S of R; the y are the target values in R):

    saved(S) = n/2 log2(sigma_0^2 / sigma_S^2) + (sum (y - mu_0)^2 / sigma_0^2 - n) / (2 ln 2) - log2(n)

The bits to describe S (D = number of conditions, A = number of enabled attributes minus the target):

    L(S) = log*(D) + sum over the conditions of [ log2(A) + value bits ]
    value bits: nominal = log2(#values), value set = #values,
                numeric <= or >= : 1 + log2(#cut points), interval: 2 log2(#cut points), binary: 1

log* is Rissanen's code for an integer (log2(2.865) + log2(D) + log2(log2(D)) + ... over the positive terms). Then

    gain(S) = saved(S) - L(S)           (normalised gain: gain(S) / n)

The greedy loop adds the candidate with the largest positive gain, takes its rows out, and stops when no gain is positive. The window reports the compression (total gain divided by the bits of the default rule alone). On test data it reports log-loss (bits per row) of the list against the overall rate, bits saved, accuracy, Brier score and AUC (Area Under the ROC Curve) for a binary target, and RMSE / MAE (Root Mean Squared Error / Mean Absolute Error) for a numeric one.

### Example

adult.txt, the default single nominal run (Cortana Quality, depth 1, 10 subgroups). Here N = 1000, p_0 = 232 / 1000 = 0.232. First candidate, Married-civ-spouse: n = 443, p_S = 195 / 443 = 0.440.

    KL     = 0.440 log2(0.440 / 0.232) + 0.560 log2(0.560 / 0.768) = 0.1514 bits per row
    saved  = 443 * 0.1514 - 1/2 log2(443) = 67.05 - 4.40 = 62.65 bits
    L(S)   = log*(1) + log2(14) + log2(7) = 1.52 + 3.81 + 2.81 = 8.13 bits
    gain   = 62.65 - 8.13 = 54.52 bits     -> goes in first

Next round, Husband has 0 rows left (all married), so it is skipped. **sex = 'Male'** still has 288 of its 671 rows free, with a share of only 0.069; that is far below 0.232 and saves 29.02 bits net. Then nothing else gains:

| position | subgroup | new rows | target share | gain (bits) |
|---|---|---|---|---|
| 1 | marital-status = 'Married-civ-spouse' | 443 | 0.440 | 54.52 |
| 2 | sex = 'Male' | 288 | 0.069 | 29.02 |
| 3 | default rule | 269 | 0.232 | |

Two subgroups instead of ten, 10.7% compression (83.5 of 781.5 bits). On the full data (section 8) the list from 50 depth-2 subgroups keeps 8, compresses 19.5%, and on the 16,281 test rows it has log-loss 0.630 against 0.789 bits per row for the overall rate, AUC 0.80.

---

## 4. Multiple testing correction

### Introduction

A search evaluates hundreds or thousands of candidates, so some look significant by chance alone: at alpha = 0.05 one null candidate in twenty passes. The p-value buttons of Java give one p-value per subgroup but never correct for how many were tested.

### Where

**Multiple Testing** in the result window (Alt+U). The window has four choices and recomputes on every change:

- p-values: an exact test from the counts (single nominal, single numeric, survival), or the p-Value column (after Gaussian or Empirical p-Values; values outside 0..1, which Java writes there for alternative descriptions, are skipped)
- method: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli or layered critical values
- m: the subgroups shown, or the candidates the search evaluated
- alpha

### What it computes

The exact tests:

- single nominal: the hypergeometric (Fisher) test of the 2 x 2 table subgroup / not subgroup against target / not target
- single numeric: Welch's t-test of the subgroup against its complement, with the Student t distribution
- survival: the log-rank test (section 7)

One sided or two sided follows the quality measure: WRAcc looks for a higher share (one sided), Abs WRAcc or Chi-squared for any difference (two sided), the inverse measures for a lower value.

The corrections, with the p-values sorted p_(1) <= p_(2) <= ...:

    Bonferroni   p_adj = min(1, m p)
    Holm         p_adj(i) = max over j <= i of min(1, (m - j + 1) p_(j))
    BH           q(i) = min over j >= i of min(1, m p_(j) / j)
    BY           BH times c(m) = 1 + 1/2 + ... + 1/m
    layered      subgroup at depth L passes when p <= alpha / (D * C_L)

Bonferroni and Holm control the chance of even one false subgroup. BH (Benjamini-Hochberg) controls the False Discovery Rate (FDR): the expected share of false ones among the reported ones. BY (Benjamini-Yekutieli) does the same under any dependence between the tests, which matters here because subgroups overlap. The layered critical values of Webb split alpha evenly over the D search depths and within a depth over its C_L candidates (counted during the search). When m is larger than the number of p-values, the missing ones count as 1.

### Example

Take the 4 p-values 0.01, 0.04, 0.03 and 0.005 (m = 4). Sorted: 0.005, 0.01, 0.03, 0.04.

    Bonferroni   0.02, 0.04, 0.12, 0.16
    Holm         4 * 0.005 = 0.02, 3 * 0.01 = 0.03, 2 * 0.03 = 0.06, max(0.06, 1 * 0.04) = 0.06
    BH           4 * 0.005 / 1 = 0.02, 4 * 0.01 / 2 = 0.02, 4 * 0.03 / 3 = 0.04, 4 * 0.04 / 4 = 0.04

At alpha = 0.05 Bonferroni and Holm keep 2, BH keeps all 4.

And on real data, a column of pure random noise as target (adult.txt, WRAcc, depth 2, top 50, minimum coverage 20; the search evaluated 391 candidates):

| p-values used | subgroups significant |
|---|---|
| no correction | 35 of 50 |
| BH, m = subgroups shown (50) | 26 |
| BH, m = candidates evaluated (391) | 0 |
| Bonferroni, Holm, BY, layered | 0 or 1 |

So m = subgroups shown is still far too optimistic; the search looked at all 391. With the real target of adult.txt all 100 subgroups of a depth-2 run stay significant under every method (the largest exact p-value is 3e-9), as they should.

---

## 5. Subsampling stability

### Introduction

A real subgroup should come back when the data changes a little. Stability runs the same search again on random subsamples and counts how often every subgroup is found again.

### Where

**Stability** in the result window (Alt+Y). Settings: number of runs (10), rows per run as a fraction (0.8), the Jaccard limit for "same rows" (0.8) and a seed. The runs are mined on a worker thread; Stop keeps the runs done so far.

### What it computes

For every subgroup S of the result, over the B runs:

    same description = share of runs with a subgroup with exactly the same conditions
    same rows        = share of runs with a subgroup T where J(S, T) >= tau
    J(A, B)          = |A and B| / |A or B|    (Jaccard similarity, the rows on the full data)
    mean Jaccard     = the best J per run, averaged

"Same rows" is the useful number for numeric conditions: in a subsample age >= 33 may become age >= 34, which is the same subgroup in all but the name. Subsampling (without replacement) is used instead of the bootstrap: duplicated rows would change coverage counts and the minimum coverage.

### Example

adult.txt, WRAcc, depth 2, top 20, 10 runs on 800 rows each:

| subgroup | same description | same rows | mean J |
|---|---|---|---|
| marital-status = 'Married-civ-spouse' AND age >= 29.0 | 1.0 | 1.0 | 1.000 |
| marital-status = 'Married-civ-spouse' AND hours-per-week >= 35.0 | 0.3 | 1.0 | 0.974 |
| relationship = 'Husband' AND age <= 76.0 | 0.3 | 1.0 | 0.985 |

Only half of the descriptions repeat between runs (mean overlap 0.51), but every subgroup comes back with nearly the same rows. Adult is a stable data set; the descriptions just wobble at the cut points.

---

## 6. Model error slicing

### Introduction

Where does a classifier go wrong? Train a model, mark every row it gets wrong, then let subgroup discovery find where the errors pile up. This is also a fairness check: is the model worse for one group?

### Where

**Tools > Add Model Error Columns...** in the main window. It trains a logistic regression with k-fold cross-validation, so every row gets its prediction from a model that never saw it (an out-of-fold prediction), and adds four columns:

- **model_score** predicted probability of the positive value
- **model_error** 1 when the prediction (score >= threshold) is wrong
- **model_false_positive** predicted positive, really negative
- **model_false_negative** predicted negative, really positive

Afterwards the target is set to model_error = 1, so Subgroup Discovery right away finds the subgroups with more errors than average. By default the new columns and the predicted column are switched off as search conditions (Meta Data turns them on again). The score column also fits the existing SCAPE target type (binary label plus score).

### What it computes

Features: numeric columns standardised (missing becomes 0 plus a 0/1 "missing" column), binary as 0/1, nominal one-hot without the most frequent level. The weights w maximise the L2 penalised log-likelihood

    l(w) = sum_i [ y_i log s(x_i w) + (1 - y_i) log(1 - s(x_i w)) ] - lambda/2 |w|^2,   s(t) = 1 / (1 + e^-t)

with Newton's method; the intercept is not penalised.

### Example

The full Adult training file (32,561 rows, 100 features after one-hot, 5 folds): accuracy 0.852, out-of-fold AUC 0.907, error rate 14.8%. Mining model_error = 1 with WRAcc puts **marital-status = 'Married-civ-spouse'** on top: 26.2% errors there against 14.8% overall. And the fairness view:

| | error rate | missed high earners (false negatives among > 50K) |
|---|---|---|
| sex = 'Female' | 7.4% | 49.0% |
| sex = 'Male' | 18.4% | 38.1% |

So the model looks better for women overall, but it misses half of the women who earn more than 50K. A single accuracy number would hide that.

---

## 7. Survival target type (time-to-event)

### Introduction

In medical data the target is often a time until something happens (death, relapse), and for some rows it did not happen before the study ended: those rows are censored, we only know the time is longer. A new target type, **survival (time-to-event)**, finds subgroups whose survival differs from the rest. This is Exceptional Model Mining with a Kaplan-Meier model.

### Where

Target type **survival (time-to-event)** in the main window. The target attribute is the time column (numeric), the second box the event column (binary, or numeric with 1 = it happened, 0 = censored); columns called time and event / status are picked by themselves. The target info shows the number of events and the median time. Base Model and Show Model draw Kaplan-Meier curves (subgroup, complement and all rows, with tick marks at the censored times). Random qualities, swap randomization, the p-value buttons, autorun files and the multiple testing window all work for it.

### What it computes

For every distinct time t_j with at least one event: n_j rows still at risk (time >= t_j) and d_j events at t_j; n_1j and d_1j the same within the subgroup.

    Kaplan-Meier   S(t) = product over t_j <= t of (1 - d_j / n_j)
    expected       E_1 = sum_j d_j n_1j / n_j
    observed       O_1 = sum_j d_1j
    variance       V   = sum_j d_j (n_1j / n_j)(1 - n_1j / n_j)(n_j - d_j) / (n_j - 1)
    log-rank       z   = (O_1 - E_1) / sqrt(V),   chi2 = z^2  (1 degree of freedom)
    hazard ratio   HR  = (O_1 / E_1) / (O_0 / E_0)    (0 = the complement)

Three quality measures:

- **Log-rank test** chi2, any difference (default minimum 3.84, the 5% point)
- **Log-rank z (worse survival)** z, more events than expected
- **Log-rank z (better survival)** -z

The result table shows the hazard ratio and the median survival time of the subgroup.

### Example

A small example by hand: 5 rows with times 1, 2, 3, 4, 5 where only 1 and 3 are events. At t = 1 all 5 are at risk, so S(1) = 1 - 1/5 = 0.8. Row 2 is censored and leaves. At t = 3 three are at risk: S(3) = 0.8 * (1 - 1/3) = 0.533. The curve never reaches 0.5, so the median is "not reached".

On data/survival_demo.txt (section 8, 1,500 made up patients, 932 events) We hid one effect on purpose: treatment B with biomarker >= 115 has a 3.5 times higher hazard. Log-rank z (worse survival), depth 2, finds it on top:

| subgroup | coverage | z | hazard ratio | median time |
|---|---|---|---|---|
| treatment = 'B' AND biomarker >= 115.0 | 117 | 8.76 | 2.46 | 9.2 months |
| age >= 69.0 AND biomarker <= 135.3 | 481 | 8.54 | 1.76 | 19.1 months |

The measured hazard ratio (2.46) is lower than the planted 3.5 because the complement also holds high risk patients (older, later stage).

---

## 8. Data files and tools

- **data/adult_full_train.txt** (32,561 rows) and **data/adult_full_test.txt** (16,281 rows): the full UCI Adult data in the format of ../adult.txt. "?" became an empty field, because the SubDisc loader only reads empty fields as missing; the labels became leq50K / gr50K. Source: the UCI Machine Learning Repository (Becker and Kohavi, "Adult", 1996, CC BY 4.0). The bundled adult.txt has 1,000 rows, so a 30% holdout leaves only about 70 people above 50K, and on so few rows the train / test difference is mostly noise
- **tools/make_adult_full.py** makes those two files from adult.data and adult.test of the UCI archive: **python tools/make_adult_full.py folder_with_the_uci_files**
- **data/survival_demo.txt**: made up survival data with the hidden effect of section 7. Nothing in it is real
- **tools/make_survival_demo.py** writes it again (rows and seed can be given)

---

## 9. Tests and what changed in the ported code

**python -m pytest tests** now runs 62 tests: the 38 ported ones (unchanged, all pass) plus

- **tests/test_new_features.py** (21 tests): the log-rank test against the textbook loop, Kaplan-Meier by hand, the planted survival subgroup, random qualities and swap randomization for survival, the subgroup list example above, holdout scoring (on the training rows it must give the mined quality back, and a test file must give the same numbers as the same rows in the table), the corrections by hand, the hypergeometric test against brute force, the t distribution against table values, stability and the model error columns
- **tests/test_gui_new_features.py** (3 tests): the new windows offscreen, the dialogs answer for themselves

New modules: **subdisc/holdout.py**, **subgroup_list.py**, **multiple_testing.py**, **stability.py**, **model_errors.py**, **survival.py** and in **subdisc/gui/** feature_common.py, holdout_window.py, subgroup_list_window.py, multiple_testing_window.py, stability_window.py, survival_window.py and model_error_dialog.py.

Small changes in ported files, none of them changes a result:

- **process.py**: the part that builds the SubgroupDiscovery for a target type is now its own function, **create_subgroup_discovery()**, so the holdout test can score subgroups on other rows with the same setup
- **subgroup_discovery.py**: counts the candidates per depth (for the layered critical values) and has the survival setup and evaluation
- **target_type.py**, **qm.py**, **result_table_model.py**, **target_concept.py**, **table.py**, **validation.py**: the survival target type and its three quality measures were added at the end (target type 11, measures 93 to 95)
- **gui/result_window.py**: a fourth row of buttons and the Kaplan-Meier Show Model
- **gui/mining_window.py**: the Tools menu and the survival target type

An autorun file with the survival target type can only be run by the Python port; the Java jar does not know that target type.

---
