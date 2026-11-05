import math

import numpy as np
import pytest

from mc_greeks import BlackScholesParams, CashDigitalCall, EuropeanCall, IntervalDigital
from mc_greeks.simulation import (make_generator, pseudo_normals, sobol_normals, spawn_generators,
                                  terminal_price, z_of_price)


def test_terminal_mean_and_variance(params, rng):
    n = 400_000
    S = terminal_price(params, pseudo_normals(rng, n))
    s2T = params.sigma**2 * params.T
    mean = params.S0 * math.exp(params.r * params.T)
    var = mean**2 * (math.exp(s2T) - 1)
    assert abs(S.mean() - mean) <= 4.5 * math.sqrt(var / n)
    # SE of the sample variance uses the lognormal 4th central moment
    m4 = mean**4 * (math.exp(s2T) - 1) ** 2 * (math.exp(4 * s2T) + 2 * math.exp(3 * s2T) + 3 * math.exp(2 * s2T) - 3)
    assert abs(S.var() - var) <= 4.5 * math.sqrt((m4 - var**2) / n)


def test_discounted_price_is_martingale(params, rng):
    disc = params.discount * terminal_price(params, pseudo_normals(rng, 400_000))
    assert abs(disc.mean() - params.S0) <= 4.5 * disc.std() / math.sqrt(disc.size)


def test_log_price_is_exact_gaussian_map(params):
    Z = np.array([-1.0, 0.0, 2.0])
    logS = np.log(terminal_price(params, Z))
    expected = math.log(params.S0) + (params.r - 0.5 * params.sigma**2) * params.T + params.sigma * Z
    np.testing.assert_allclose(logS, expected, rtol=1e-14)


def test_z_of_price_inverts(params):
    for s in (80.0, 100.0, 110.0, 150.0):
        assert terminal_price(params, np.array([z_of_price(params, s)]))[0] == pytest.approx(s, rel=1e-12)


def test_seed_reproducibility():
    a = [g.standard_normal(5) for g in spawn_generators(123, 3)]
    b = [g.standard_normal(5) for g in spawn_generators(123, 3)]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.allclose(a[0], a[1])  # children are distinct streams
    np.testing.assert_array_equal(make_generator(7).standard_normal(3), make_generator(7).standard_normal(3))


def test_sobol_normals_finite_and_balanced():
    z = sobol_normals(4096, seed=1)
    assert np.all(np.isfinite(z))
    assert abs(z.mean()) < 1e-3  # far below the MC scale 1/sqrt(4096) ~ 0.016
    assert abs(z.var() - 1) < 5e-3
    assert np.all(np.isfinite(sobol_normals(16, scramble=False)))


@pytest.mark.parametrize("bad", [dict(S0=-1.0), dict(sigma=0.0), dict(sigma=-0.2), dict(T=-1.0), dict(T=0.0),
                                 dict(r=float("nan"))])
def test_invalid_params(bad):
    base = dict(S0=100.0, r=0.1, sigma=0.2, T=1.0)
    base.update(bad)
    with pytest.raises(ValueError):
        BlackScholesParams(**base)


def test_invalid_payoffs():
    with pytest.raises(ValueError):
        IntervalDigital(110, 100)
    with pytest.raises(ValueError):
        IntervalDigital(0, 100)
    with pytest.raises(ValueError):
        EuropeanCall(-5)
    with pytest.raises(ValueError):
        CashDigitalCall(0)


def test_payoff_values():
    s = np.array([90.0, 100.0, 105.0, 110.0, 120.0])
    np.testing.assert_array_equal(IntervalDigital(100, 110)(s), [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(CashDigitalCall(100)(s), [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(EuropeanCall(100)(s), [0, 0, 5, 10, 20])
