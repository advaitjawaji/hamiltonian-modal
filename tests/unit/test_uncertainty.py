"""Tests for quantile uncertainty estimation."""
import numpy as np
import pytest

from hamiltonian_modal.world_model.uncertainty import (
    QuantileUncertainty,
    UncertaintyEstimate,
    pinball_loss,
    QUANTILE_LEVELS,
    N_QUANTILES,
    EnsembleUncertainty,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def estimator():
    return QuantileUncertainty(n_modes=5, n_contact_modes=3, hidden=[16, 16], seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# test_quantile_uncertainty_shapes
# ---------------------------------------------------------------------------

def test_quantile_uncertainty_shapes(estimator, rng):
    """QuantileUncertainty returns UncertaintyEstimate with correct shapes."""
    n_modes = estimator.n_modes
    eta = rng.standard_normal(n_modes)
    est = estimator(eta, m=0)

    assert isinstance(est, UncertaintyEstimate)
    assert est.quantiles.shape == (n_modes, N_QUANTILES), (
        f"Expected ({n_modes}, {N_QUANTILES}), got {est.quantiles.shape}"
    )
    assert est.quantiles.dtype == np.float64


def test_quantile_uncertainty_all_modes(rng):
    """Works for various n_modes values."""
    for n_modes in [1, 3, 10, 20]:
        est = QuantileUncertainty(n_modes=n_modes, n_contact_modes=2, hidden=[8], seed=n_modes)
        eta = rng.standard_normal(n_modes)
        result = est(eta, m=0)
        assert result.quantiles.shape == (n_modes, N_QUANTILES)


# ---------------------------------------------------------------------------
# test_quantile_monotonicity
# ---------------------------------------------------------------------------

def test_quantile_monotonicity(estimator, rng):
    """For every mode, quantiles must be strictly increasing: q0.1 < q0.25 < ... < q0.9."""
    for _ in range(20):
        eta = rng.standard_normal(estimator.n_modes)
        est = estimator(eta, m=rng.integers(0, estimator.n_contact_modes))

        for mode_i in range(estimator.n_modes):
            q = est.quantiles[mode_i]
            for k in range(len(q) - 1):
                assert q[k] < q[k + 1], (
                    f"Monotonicity violated at mode {mode_i}, quantile {k}: "
                    f"q[{k}]={q[k]:.4f} >= q[{k+1}]={q[k+1]:.4f}"
                )


def test_quantile_monotonicity_edge_cases():
    """Monotonicity holds even for zero-initialized MLP biases."""
    est = QuantileUncertainty(n_modes=3, n_contact_modes=2, hidden=[4], seed=99)
    eta = np.zeros(3)
    result = est(eta, m=0)
    for mode_i in range(3):
        q = result.quantiles[mode_i]
        for k in range(len(q) - 1):
            assert q[k] <= q[k + 1]


# ---------------------------------------------------------------------------
# test_90_percent_interval
# ---------------------------------------------------------------------------

def test_90_percent_interval(estimator, rng):
    """interval_90 must return (q0.1, q0.9) for each mode."""
    eta = rng.standard_normal(estimator.n_modes)
    est = estimator(eta, m=1)

    lo, hi = est.interval_90
    assert lo.shape == (estimator.n_modes,)
    assert hi.shape == (estimator.n_modes,)

    lo_idx = list(QUANTILE_LEVELS).index(0.1)
    hi_idx = list(QUANTILE_LEVELS).index(0.9)
    np.testing.assert_array_equal(lo, est.quantiles[:, lo_idx])
    np.testing.assert_array_equal(hi, est.quantiles[:, hi_idx])

    # lo ≤ hi for every mode
    assert np.all(lo <= hi), "90% interval lower bound exceeds upper bound"


# ---------------------------------------------------------------------------
# test_pinball_loss_positive
# ---------------------------------------------------------------------------

def test_pinball_loss_positive(estimator, rng):
    """Pinball loss must be non-negative."""
    for _ in range(20):
        y_true = rng.standard_normal(estimator.n_modes)
        est = estimator(rng.standard_normal(estimator.n_modes), m=0)
        loss = pinball_loss(y_true, est.quantiles)
        assert loss >= 0.0, f"Pinball loss is negative: {loss}"


def test_pinball_loss_zero_at_truth():
    """Pinball loss is zero when all quantile predictions equal the true value."""
    n_modes = 4
    y_true = np.array([1.0, 2.0, -1.0, 0.5])
    # All quantiles set to the true value → error=0 for every quantile level
    quantile_preds = np.tile(y_true[:, None], (1, N_QUANTILES))
    loss = pinball_loss(y_true, quantile_preds)
    assert abs(loss) < 1e-14, f"Expected 0, got {loss}"


# ---------------------------------------------------------------------------
# test_pinball_loss_at_truth
# ---------------------------------------------------------------------------

def test_pinball_loss_at_truth(rng):
    """When quantile predictions are very close to y_true, loss is small."""
    n_modes = 6
    y_true = rng.standard_normal(n_modes)
    # Small random noise around the truth
    quantile_preds = y_true[:, None] + rng.standard_normal((n_modes, N_QUANTILES)) * 0.01
    # Sort to enforce monotonicity
    quantile_preds = np.sort(quantile_preds, axis=1)
    loss = pinball_loss(y_true, quantile_preds)
    assert loss < 0.05, f"Loss {loss:.4f} is not small when predictions are near truth"


def test_pinball_loss_higher_for_bad_preds(rng):
    """Pinball loss must be higher for bad predictions than near-truth predictions."""
    n_modes = 4
    y_true = rng.standard_normal(n_modes)

    good_preds = np.sort(y_true[:, None] + 0.01 * rng.standard_normal((n_modes, N_QUANTILES)), axis=1)
    bad_preds = np.sort(y_true[:, None] + 10.0 * rng.standard_normal((n_modes, N_QUANTILES)), axis=1)

    loss_good = pinball_loss(y_true, good_preds)
    loss_bad = pinball_loss(y_true, bad_preds)
    assert loss_good < loss_bad, f"Good loss {loss_good} should be < bad loss {loss_bad}"


# ---------------------------------------------------------------------------
# test_uncertainty_estimate_properties
# ---------------------------------------------------------------------------

def test_uncertainty_estimate_properties(estimator, rng):
    """UncertaintyEstimate.mean returns the median (0.5 quantile)."""
    eta = rng.standard_normal(estimator.n_modes)
    est = estimator(eta, m=0)

    mean = est.mean
    assert mean.shape == (estimator.n_modes,)
    median_idx = list(QUANTILE_LEVELS).index(0.5)
    np.testing.assert_array_equal(mean, est.quantiles[:, median_idx])


def test_uncertainty_estimate_quantile_levels(estimator, rng):
    """UncertaintyEstimate.quantile_levels matches the module-level QUANTILE_LEVELS."""
    eta = rng.standard_normal(estimator.n_modes)
    est = estimator(eta, m=0)
    assert est.quantile_levels == QUANTILE_LEVELS


def test_uncertainty_estimate_consistent_intervals(estimator, rng):
    """90% interval bounds must match the full quantile array slices."""
    eta = rng.standard_normal(estimator.n_modes)
    est = estimator(eta, m=2)

    lo, hi = est.interval_90
    lo_idx = list(QUANTILE_LEVELS).index(0.1)
    hi_idx = list(QUANTILE_LEVELS).index(0.9)

    np.testing.assert_array_equal(lo, est.quantiles[:, lo_idx])
    np.testing.assert_array_equal(hi, est.quantiles[:, hi_idx])


# ---------------------------------------------------------------------------
# EnsembleUncertainty (legacy)
# ---------------------------------------------------------------------------

def test_ensemble_uncertainty_shapes(rng):
    """EnsembleUncertainty.predict returns mean and variance with correct shapes."""
    ensemble = EnsembleUncertainty(n_members=5)
    preds = [rng.standard_normal(10) for _ in range(5)]
    out = ensemble.predict(preds)

    assert "mean" in out and "variance" in out
    assert out["mean"].shape == (10,)
    assert out["variance"].shape == (10,)
    assert np.all(out["variance"] >= 0.0)


def test_ensemble_uncertainty_identical_members():
    """If all members predict the same, variance is zero."""
    pred = np.array([1.0, 2.0, 3.0])
    ensemble = EnsembleUncertainty(n_members=3)
    out = ensemble.predict([pred, pred, pred])
    np.testing.assert_allclose(out["mean"], pred, rtol=1e-14)
    np.testing.assert_allclose(out["variance"], np.zeros(3), atol=1e-14)
