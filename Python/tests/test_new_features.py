"""Tests for the features that are not in the java version (see
NEW_FEATURES.md): survival target, subgroup list, holdout test, multiple
testing, stability and model error columns."""

import math
import os

import numpy as np
import pytest

from conftest import load, search_parameters
from subdisc import holdout, model_errors, process, stability
from subdisc import multiple_testing as mt
from subdisc.qm import QM
from subdisc.subgroup_list import SubgroupList, kl_binary, log_star
from subdisc.survival import SurvivalBase, chi2_p_value, kaplan_meier, km_median
from subdisc.target_concept import TargetConcept
from subdisc.target_type import TargetType

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def data_file(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        pytest.skip("data file not found: " + p)
    return p


def load_data(name):
    from subdisc.data_loader_txt import DataLoaderTXT
    return DataLoaderTXT(data_file(name)).get_table()


def nominal_tc(t):
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NOMINAL)
    tc.set_primary_target(t.get_column("target"))
    tc.set_target_value("gr50K")
    return tc


def survival_tc(t):
    tc = TargetConcept()
    tc.set_target_type(TargetType.SURVIVAL)
    tc.set_primary_target(t.get_column("time"))
    tc.set_secondary_target(t.get_column("event"))
    return tc


# ----------------------------------------------------------------------
# survival
# ----------------------------------------------------------------------

def naive_log_rank(times, events, group):
    """the textbook loop, one event time after the other"""
    o = e = v = 0.0
    for t in sorted(set(times[events])):
        at_risk = times >= t
        n = at_risk.sum()
        n1 = (at_risk & group).sum()
        d = ((times == t) & events).sum()
        d1 = ((times == t) & events & group).sum()
        o += d1
        e += d * n1 / n
        if n > 1:
            v += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return (o - e) / math.sqrt(v), o, e


def test_log_rank_matches_the_textbook_loop():
    t = load_data("survival_demo.txt")
    base = SurvivalBase(t.get_column("time"), t.get_column("event"))
    times = np.asarray(t.get_column("time").floats_view()[:t.get_nr_rows()], dtype=np.float64)
    events = np.asarray(t.get_column("event").get_binaries()[:t.get_nr_rows()], dtype=bool)
    rng = np.random.default_rng(3)
    for _ in range(5):
        g = rng.random(t.get_nr_rows()) < 0.3
        z, chi2, o, e, hr = base.log_rank(g)
        z2, o2, e2 = naive_log_rank(times, events, g)
        assert z == pytest.approx(z2, rel=1e-9)
        assert o == o2 and e == pytest.approx(e2, rel=1e-9)
        assert chi2 == pytest.approx(z2 * z2, rel=1e-9)


def test_kaplan_meier_by_hand():
    # 5 rows: events at 1 and 3, censored at 2, 4 and 5
    xs, ys, cens = kaplan_meier([1, 2, 3, 4, 5], [True, False, True, False, False])
    # S(1) = 4/5, S(3) = 4/5 * (1 - 1/3) = 0.5333
    assert list(xs) == [0, 1, 3]
    assert ys == pytest.approx([1.0, 0.8, 0.8 * 2 / 3])
    assert list(cens) == [2, 4, 5]
    assert math.isnan(km_median(xs, ys))
    xs, ys, _ = kaplan_meier([1, 2, 3], [True, True, True])
    assert km_median(xs, ys) == 2


def test_chi2_p_value():
    assert chi2_p_value(3.841458820694124) == pytest.approx(0.05, abs=1e-9)
    assert chi2_p_value(0) == 1.0


def test_survival_mining_finds_the_planted_subgroup():
    t = load_data("survival_demo.txt")
    sp = search_parameters(survival_tc(t), QM.LOGRANK_WORSE, 0.0)
    sp.set_search_depth(2)
    sp.set_minimum_coverage(20)
    sp.set_maximum_subgroups(10)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    best = list(sd.get_result())[0]
    assert str(best) == "treatment = 'B' AND biomarker >= 115.0"
    assert best.get_secondary_statistic() > 2.0      # hazard ratio
    counts = sd.get_candidate_count_per_depth()
    assert sum(counts.values()) == sd.get_candidate_count()


def test_survival_random_qualities_and_swap_randomization():
    from subdisc.validation import Validation, RANDOM_SUBSETS, SWAP_RANDOMIZATION
    t = load_data("survival_demo.txt")
    sp = search_parameters(survival_tc(t), QM.LOGRANK, 0.0)
    v = Validation(sp, t, None, None)
    q = v.get_qualities([RANDOM_SUBSETS, "20"])
    assert len(q) == 20 and all(x >= 0 for x in q)
    q = v.get_qualities([SWAP_RANDOMIZATION, "3"])
    assert len(q) == 3
    # the target columns are put back afterwards
    assert t.get_column("time") is sp.get_target_concept().get_primary_target()


# ----------------------------------------------------------------------
# subgroup list
# ----------------------------------------------------------------------

def test_log_star_and_kl():
    assert log_star(1) == pytest.approx(math.log2(2.865064))
    assert kl_binary(0.5, 0.5) == 0.0
    assert kl_binary(195 / 443, 0.232) == pytest.approx(0.15135, abs=1e-4)


def test_subgroup_list_skips_the_nested_subgroup():
    t = load("adult.txt")
    sp = search_parameters(nominal_tc(t), QM.CORTANA_QUALITY, 0.1)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    res = list(sd.get_result())
    assert str(res[0]) == "marital-status = 'Married-civ-spouse'" and str(res[1]) == "relationship = 'Husband'"
    y = holdout.target_values(t, sp.get_target_concept())
    sl = SubgroupList.from_subgroups(res, "binary", y, np.ones(t.get_nr_rows(), dtype=bool), t, sp.get_nr_bins())
    first = sl.entries[0]
    assert str(first.subgroup) == "marital-status = 'Married-civ-spouse'"
    # the example of NEW_FEATURES.md
    assert first.new_rows == 443
    assert first.saved_bits == pytest.approx(443 * kl_binary(195 / 443, 0.232) - 0.5 * math.log2(443))
    assert first.model_bits == pytest.approx(log_star(1) + math.log2(14) + math.log2(7))
    # every Husband row is married, nothing is left for it
    assert all(str(e.subgroup) != "relationship = 'Husband'" for e in sl.entries)
    assert sum(e.new_rows for e in sl.entries) + sl.default_rows == t.get_nr_rows()
    assert 0 < sl.compression() < 1
    # scoring on the training rows: the list must beat the overall rate
    s = sl.score(t, y)
    assert s["bits saved on test"] > 0 and s["AUC"] > 0.5


def test_subgroup_list_numeric_target():
    t = load("adult.txt")
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NUMERIC)
    tc.set_primary_target(t.get_column("age"))
    sp = search_parameters(tc, QM.Z_SCORE, 0.0)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    y = holdout.target_values(t, tc)
    sl = SubgroupList.from_subgroups(list(sd.get_result()), "numeric", y, np.ones(t.get_nr_rows(), dtype=bool), t)
    assert sl.entries and sl.total_gain > 0
    s = sl.score(t, y)
    assert s["RMSE list"] < s["RMSE default"]


# ----------------------------------------------------------------------
# holdout
# ----------------------------------------------------------------------

def test_random_split():
    train, test = holdout.random_split(1000, 0.3, seed=5)
    assert test.sum() == 300 and train.sum() == 700 and not (train & test).any()
    a, _ = holdout.random_split(1000, 0.3, seed=5)
    assert (a == train).all()


def test_evaluate_conditions_is_column_evaluate():
    t = load("adult.txt")
    sp = search_parameters(nominal_tc(t), QM.WRACC, 0.0)
    sp.set_search_depth(2)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    for s in sd.get_result():
        assert (holdout.evaluate_conditions(t, s.get_conditions()) == s.get_members()[:t.get_nr_rows()]).all()


def test_score_on_the_training_rows_gives_the_mined_quality():
    t = load("adult.txt")
    train, test = holdout.random_split(t.get_nr_rows(), 0.3, seed=2)
    sp = search_parameters(nominal_tc(t), QM.WRACC, 0.0)
    sd = process.run_subgroup_discovery(t, 0, train, sp, False, 1, None)
    res = list(sd.get_result())
    again = holdout.score_on(res, sp, t, train)
    for r in again:
        assert r.test_coverage == r.train_coverage
        assert r.test_quality == pytest.approx(r.train_quality, abs=1e-6)
    out = holdout.score_on(res, sp, t, test)
    assert all(r.test_coverage <= 300 for r in out)
    s = holdout.summary(out, 700, 300, 0.0)
    assert s["subgroups"] == len(res) and -1 <= s["rank correlation"] <= 1


def test_load_test_table_round_trip(tmp_path):
    t = load("adult.txt")
    train, test = holdout.random_split(t.get_nr_rows(), 0.25, seed=9)
    path = str(tmp_path / "adult_test_part.txt")
    t.to_file(test, path)
    tt, warnings = holdout.load_test_table(t, path)
    assert tt.get_nr_rows() == 250
    sp = search_parameters(nominal_tc(t), QM.WRACC, 0.0)
    sd = process.run_subgroup_discovery(t, 0, train, sp, False, 1, None)
    res = list(sd.get_result())
    a = holdout.score_on(res, sp, t, test)
    b = holdout.score_on(res, sp, tt, None)
    for x, y in zip(a, b):
        assert x.test_coverage == y.test_coverage
        assert x.test_quality == pytest.approx(y.test_quality, abs=1e-6, nan_ok=True)


def test_load_test_table_wrong_columns(tmp_path):
    t = load("adult.txt")
    p = tmp_path / "other.txt"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        holdout.load_test_table(t, str(p))


# ----------------------------------------------------------------------
# multiple testing
# ----------------------------------------------------------------------

def test_corrections_by_hand():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005, 0.01, 0.03, 0.04
    assert list(mt.bonferroni(p)) == pytest.approx([0.04, 0.16, 0.12, 0.02])
    # holm: 4*0.005=0.02, 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04 -> max so far 0.06
    assert list(mt.holm(p)) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # bh: 4*0.005/1=0.02, 4*0.01/2=0.02, 4*0.03/3=0.04, 4*0.04/4=0.04
    assert list(mt.benjamini_hochberg(p)) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4
    assert list(mt.benjamini_yekutieli(p)) == pytest.approx([0.02 * c, 0.04 * c, 0.04 * c, 0.02 * c])
    # with m larger than the number of p-values
    assert list(mt.bonferroni([0.001], 100)) == pytest.approx([0.1])
    thr = mt.layered_thresholds([1, 2], {1: 10, 2: 100}, 0.05)
    assert list(thr) == pytest.approx([0.05 / 20, 0.05 / 200])


def test_hypergeometric_against_brute_force():
    N, K, n = 30, 12, 10
    def pmf(x):
        return math.comb(K, x) * math.comb(N - K, n - x) / math.comb(N, n)
    for x in range(0, 11):
        greater = sum(pmf(k) for k in range(x, min(n, K) + 1))
        assert mt.hypergeometric_p(N, K, n, x, "greater") == pytest.approx(greater, rel=1e-9)
        two = sum(pmf(k) for k in range(0, min(n, K) + 1) if pmf(k) <= pmf(x) * (1 + 1e-7))
        assert mt.hypergeometric_p(N, K, n, x, "two-sided") == pytest.approx(min(1.0, two), rel=1e-9)


def test_student_t():
    # table values: t = 2.228 is the two sided 5% point for 10 df, 1.96 for very many
    assert mt.student_t_sf(2.228138852, 10) == pytest.approx(0.025, abs=1e-6)
    assert mt.student_t_sf(1.959963985, 1e7) == pytest.approx(0.025, abs=1e-5)
    assert mt.student_t_sf(0.0, 5) == pytest.approx(0.5)


def test_exact_p_values_and_adjust():
    t = load("adult.txt")
    sp = search_parameters(nominal_tc(t), QM.WRACC, 0.0)
    sp.set_search_depth(2)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    res = list(sd.get_result())
    p = mt.exact_p_values(res, sp, t)
    assert len(p) == len(res) and all(0 <= x <= 1 for x in p)
    adj, sig, _ = mt.adjust(mt.METHODS[2], p, sd.get_candidate_count(), 0.05)
    adj_k, _, _ = mt.adjust(mt.METHODS[2], p, len(p), 0.05)
    assert (adj >= adj_k - 1e-15).all()   # more hypotheses, larger q-values
    _, sig_l, thr = mt.adjust(mt.METHODS[4], p, None, 0.05, [s.get_depth() for s in res],
                              sd.get_candidate_count_per_depth(), 2)
    assert thr.shape[0] == len(res)


# ----------------------------------------------------------------------
# stability
# ----------------------------------------------------------------------

def test_stability():
    t = load("adult.txt")
    sp = search_parameters(nominal_tc(t), QM.WRACC, 0.0)
    sp.set_maximum_subgroups(10)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    res = list(sd.get_result())
    rows, stats = stability.run(t, None, sp, res, nr_runs=4, fraction=0.8, seed=2)
    assert stats["runs done"] == 4 and len(rows) == len(res)
    for r in rows:
        assert 0 <= r.exact <= 1 and 0 <= r.cover <= 1 and 0 <= r.mean_jaccard <= 1
    # the strongest subgroup of adult comes back every time
    assert rows[0].cover == 1.0
    c = stability.Canceller()
    c.cancelled = True
    rows, stats = stability.run(t, None, sp, res, nr_runs=4, canceller=c)
    assert stats["runs done"] == 0


def test_jaccard_matrix():
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
    assert list(stability.jaccard_matrix(a, b)[0]) == pytest.approx([1.0, 1 / 3, 0.0])


# ----------------------------------------------------------------------
# model errors
# ----------------------------------------------------------------------

def test_model_error_columns():
    t = load("adult.txt")
    n_before = t.get_nr_columns()
    new, info = model_errors.add_model_error_columns(t, t.get_column("target"), "gr50K", k=5)
    assert new == ["model_score", "model_error", "model_false_positive", "model_false_negative"]
    assert t.get_nr_columns() == n_before + 4
    n = t.get_nr_rows()
    err = np.asarray(t.get_column("model_error").get_binaries()[:n])
    fp = np.asarray(t.get_column("model_false_positive").get_binaries()[:n])
    fn = np.asarray(t.get_column("model_false_negative").get_binaries()[:n])
    assert (err == (fp | fn)).all() and not (fp & fn).any()
    assert info["accuracy"] > 0.76     # better than always saying 'leq50K'
    assert info["AUC (out of fold)"] > 0.8
    # mining on the error column works like any other binary target
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NOMINAL)
    tc.set_primary_target(t.get_column("model_error"))
    tc.set_target_value("1")
    for name in new + ["target"]:
        t.get_column(name).set_is_enabled(False)
    sd = process.run_subgroup_discovery(t, 0, None, search_parameters(tc, QM.WRACC, 0.0), False, 1, None)
    assert len(sd.get_result()) > 0
    assert all("model_" not in str(s) for s in sd.get_result())
    # a second call does not overwrite
    new2, _ = model_errors.add_model_error_columns(t, t.get_column("target"), "gr50K", k=3)
    assert new2[0] == "model_score_2"


def test_logistic_fit_on_separable_free_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    w_true = np.array([-0.5, 2.0, -1.0])
    p = 1 / (1 + np.exp(-(w_true[0] + X @ w_true[1:])))
    y = (rng.random(2000) < p).astype(float)
    w = model_errors.fit_logistic(X, y, l2=0.0)
    assert w == pytest.approx(w_true, abs=0.2)
