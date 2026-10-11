"""检验影响结论的统计口径、数据泄漏、排序及业务评价边界。"""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, ARMS, SEED, RAW
from src.statistics import mean_difference, rate_difference
from src.evaluation import policy_effect, gain_curve, ranking, top_policy, ranking_metrics
from src.modeling import feature_frame, split_data, make_preprocessor
from src.data import audit

def test_rate_difference_is_absolute_probability_difference():
    treated = np.r_[np.ones(20), np.zeros(80)]
    control = np.r_[np.ones(10), np.zeros(90)]
    result = rate_difference(treated, control)
    assert result["effect"] == pytest.approx(.10)
    assert result["ci_low"] < result["effect"] < result["ci_high"]

def test_welch_unequal_variance_and_sign():
    treated = np.array([0., 0., 0., 4., 12.])
    control = np.array([0., 0., 0., 1., 2., 0.])
    forward = mean_difference(treated, control)
    reverse = mean_difference(control, treated)
    assert forward["effect"] == pytest.approx(-reverse["effect"])
    assert forward["ci_low"] == pytest.approx(-reverse["ci_high"])

def test_all_send_policy_matches_unadjusted_ate_with_unequal_allocation():
    y = np.array([1, 0, 1, 0, 0, 1])
    treatment = np.array([1, 1, 1, 1, 0, 0])
    effect = policy_effect(y, treatment, np.ones(len(y)))
    assert effect["incremental_rate_population"] == pytest.approx(
        y[treatment == 1].mean() - y[treatment == 0].mean()
    )
    fraction, gain = gain_curve(y, treatment, np.arange(len(y)))
    assert gain[-1] / len(y) == pytest.approx(effect["incremental_rate_population"])
    assert fraction[-1] == 1

def test_no_send_has_zero_incremental_effect_and_uncertainty():
    result = policy_effect(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.zeros(4))
    assert result["incremental_rate_population"] == 0
    assert result["ci_low"] == result["ci_high"] == 0

def test_masked_outcome_uses_population_denominator():
    y = np.array([1, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0])
    selected = np.array([1, 0, 0, 0])
    assert policy_effect(y, treatment, selected)["incremental_rate_population"] == .5

def test_paired_policy_contrast_not_independent_intervals():
    y = np.array([1, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0])
    first = np.array([1., 0, 0, 0])
    second = np.repeat(.25, 4)
    combined = policy_effect(y, treatment, first - second)["incremental_rate_population"]
    separate = (
        policy_effect(y, treatment, first)["incremental_rate_population"]
        - policy_effect(y, treatment, second)["incremental_rate_population"]
    )
    assert combined == pytest.approx(separate)

def test_known_responders_rank_better_than_reverse_order():
    # 低分群没有增益；高分群处理客户全部购买、对照客户全部不买。
    group = np.repeat([0, 1], 100)
    treatment = np.tile(np.repeat([0, 1], 50), 2)
    y = group * treatment
    correct = ranking_metrics(y, treatment, group)
    reversed_scores = ranking_metrics(y, treatment, 1 - group)
    assert correct["auuc_above_random_per_customer"] > 0
    assert reversed_scores["auuc_above_random_per_customer"] < 0

def test_ties_do_not_depend_on_outcomes():
    order = ranking(np.zeros(100))
    assert np.array_equal(order, ranking(np.zeros(100)))
    assert sorted(order.tolist()) == list(range(100))

def test_policy_budget_boundaries():
    scores = np.arange(11)
    assert top_policy(scores, 0).sum() == 0
    assert top_policy(scores, 1).sum() == 11
    assert top_policy(scores, .3).sum() == round(11 * .3)
    with pytest.raises(ValueError):
        top_policy(scores, 1.1)

def example_frame(n=300):
    return pd.DataFrame({
        "recency": np.tile(np.arange(1, 11), n // 10),
        "history": np.arange(n, dtype=float),
        "mens": np.arange(n) % 2, "womens": (np.arange(n) + 1) % 2,
        "zip_code": ["Urban"] * n, "newbie": np.arange(n) % 2,
        "channel": ["Web"] * n, "history_segment": ["1) $0 - $100"] * n,
        "segment": np.tile(ARMS, n // 3),
        "conversion": np.zeros(n, dtype=int), "visit": np.zeros(n, dtype=int),
        "spend": np.zeros(n),
    })

def test_feature_allowlist_excludes_all_post_treatment_variables():
    frame = example_frame()
    x = feature_frame(frame)
    assert list(x.columns) == FEATURES
    assert not set(["visit", "conversion", "spend", "segment", "history_segment"]) & set(x.columns)
    assert np.array_equal(x.history, np.log1p(frame.history))

def test_split_has_no_overlap_and_preserves_population():
    frame = example_frame()
    train, validation, test = split_data(frame)
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert set(train) | set(validation) | set(test) == set(frame.index)
    assert [len(train), len(validation), len(test)] == [180, 60, 60]

def test_unknown_test_category_does_not_refit_training_preprocessor():
    training = feature_frame(example_frame())
    test = training.iloc[:2].copy()
    test["channel"] = "unseen"
    preprocessor = make_preprocessor().fit(training)
    before = [list(a) for a in preprocessor.named_transformers_["categorical"].categories_]
    transformed = preprocessor.transform(test)
    after = [list(a) for a in preprocessor.named_transformers_["categorical"].categories_]
    assert before == after
    assert np.isfinite(transformed).all()

def test_audit_does_not_deduplicate_shared_profiles(tmp_path, monkeypatch):
    import src.data as data_module
    monkeypatch.setattr(data_module, "OUT", tmp_path)
    frame = example_frame()
    frame = pd.concat([frame, frame.iloc[:3]], ignore_index=True)
    original_count = len(frame)
    findings = audit(frame)
    assert findings["identical_rows"] >= 3
    assert len(frame) == original_count

@pytest.mark.skipif(not (RAW / "hillstrom.csv").exists(), reason="Raw data not downloaded")
def test_real_data_contract():
    frame = pd.read_csv(RAW / "hillstrom.csv")
    assert len(frame) == 64000
    assert set(frame.segment) == set(ARMS)
    assert frame.isna().sum().sum() == 0
    assert (frame.conversion <= frame.visit).all()
    assert ((frame.spend > 0) == (frame.conversion == 1)).all()

def test_zero_purchase_events_do_not_produce_nan_inference():
    result = rate_difference(np.zeros(100), np.zeros(200))
    assert result["effect"] == 0
    assert result["p_value"] == 1
    assert np.isfinite([result["ci_low"], result["ci_high"]]).all()
    assert result["ci_low"] < 0 < result["ci_high"]
