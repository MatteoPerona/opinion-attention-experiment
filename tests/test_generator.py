"""Unit tests for the FJ / DeGroot generator (required before rung 0)."""

import numpy as np
import pytest

from sim import (
    fj_steady_state,
    generate_dataset,
    make_world,
    simulate_trajectory,
)


def test_W_rows_sum_to_one():
    rng = np.random.default_rng(0)
    for kind in ("dense", "clustered", "sparse"):
        world = make_world(N=20, rng=rng, kind=kind, degroot=True)
        sums = world.W.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(20), atol=1e-10)


def test_degroot_matches_matrix_power():
    """With Lambda=I and sigma=0, trajectory equals W^t x(0)."""
    rng = np.random.default_rng(1)
    N, T = 15, 12
    world = make_world(N=N, rng=rng, kind="dense", degroot=True)
    assert np.allclose(world.Lambda, 1.0)
    x0 = rng.normal(size=N)
    traj = simulate_trajectory(x0, world, T=T, sigma=0.0, rng=None)
    x = x0.copy()
    for t in range(T + 1):
        np.testing.assert_allclose(traj[t], x, atol=1e-10)
        x = world.W @ x


def test_fj_steady_state_matches_closed_form():
    """With all lambda_i < 1, empirical long-run state matches closed form."""
    rng = np.random.default_rng(2)
    N, T = 20, 80
    world = make_world(N=N, rng=rng, kind="dense", degroot=False)
    assert np.all(world.Lambda < 1.0)
    x0 = rng.normal(size=N)
    closed = fj_steady_state(x0, world)
    traj = simulate_trajectory(x0, world, T=T, sigma=0.0, rng=None)
    np.testing.assert_allclose(traj[-1], closed, atol=1e-6, rtol=1e-5)


def test_closed_form_rejected_for_degroot():
    """Closed form must NOT be used when Lambda = I."""
    rng = np.random.default_rng(3)
    world = make_world(N=10, rng=rng, kind="dense", degroot=True)
    x0 = rng.normal(size=10)
    with pytest.raises(ValueError, match="invalid"):
        fj_steady_state(x0, world)


def test_generate_dataset_shapes_and_no_closed_for_degroot():
    rng = np.random.default_rng(4)
    world = make_world(N=8, rng=rng, kind="dense", degroot=True)
    ds = generate_dataset(world, M=5, T=6, sigma=0.0, rng=rng)
    assert ds["trajectories"].shape == (5, 7, 8)
    assert ds["pairs_X"].shape == (30, 8)
    assert ds["steady_closed"] is None  # DeGroot: closed form not used
