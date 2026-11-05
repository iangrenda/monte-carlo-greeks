import numpy as np
import pytest

from mc_greeks import PAPER_PARAMS, PAPER_PAYOFF, CashDigitalCall, EuropeanCall


@pytest.fixture
def params():
    return PAPER_PARAMS


@pytest.fixture
def payoffs():
    return {"interval": PAPER_PAYOFF, "digital": CashDigitalCall(100.0), "call": EuropeanCall(100.0)}


@pytest.fixture
def rng():
    return np.random.default_rng(20240607)


def assert_within_se(estimate, truth, se, k=4.5):
    """|estimate - truth| <= k * SE.

    With k = 4.5 a correct unbiased estimator fails with probability ~7e-6 per
    check (normal approximation), whereas the formula errors we guard against
    (wrong sign, missing term, wrong power of sigma or T) shift the mean by
    many standard errors at the sample sizes used.
    """
    z = abs(estimate - truth) / se
    assert z <= k, f"estimate {estimate} vs truth {truth}: {z:.2f} SE away"
