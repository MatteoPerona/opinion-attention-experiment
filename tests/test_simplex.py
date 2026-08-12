"""Unit tests for row-wise Euclidean simplex projection."""

import numpy as np
import pytest

from baselines import project_rows_simplex, project_simplex


def test_simplex_rows_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 50))
    P = project_rows_simplex(A)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(20), atol=1e-10)
    assert np.all(P >= -1e-12)


def test_simplex_idempotent_on_stochastic_rows():
    rng = np.random.default_rng(1)
    raw = rng.random((10, 30)) + 1e-6
    S = raw / raw.sum(axis=1, keepdims=True)
    P = project_rows_simplex(S)
    np.testing.assert_allclose(P, S, atol=1e-10)


def test_simplex_fj_2n_rows():
    """FJ operator: project each full row of length 2N."""
    rng = np.random.default_rng(2)
    N = 15
    Op = rng.normal(size=(N, 2 * N))
    P = project_rows_simplex(Op)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(N), atol=1e-10)
    assert np.all(P >= -1e-12)
    assert P.shape == (N, 2 * N)


def test_project_simplex_uniform_from_zeros():
    w = project_simplex(np.zeros(5))
    np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-10)


def test_joint_kappa_helper():
    from sim import (
        generate_dataset,
        joint_feature_covariance_condition_number,
        make_world,
        state_covariance_condition_number,
    )

    rng = np.random.default_rng(3)
    world = make_world(N=10, rng=rng, kind="dense", degroot=False)
    ds = generate_dataset(world, M=40, T=8, sigma=0.0, rng=rng)
    k_marg = state_covariance_condition_number(ds["pairs_X"])
    k_joint = joint_feature_covariance_condition_number(ds["pairs_X"], ds["pairs_x0"])
    assert k_marg > 0 and k_joint > 0
    # Under noiseless FJ, joint cov is typically far worse conditioned
    assert k_joint > k_marg
