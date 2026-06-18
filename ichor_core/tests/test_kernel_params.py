import numpy as np

from ichor.core.models.kernels import PeriodicKernel, RBFCyclic


def test_product_kernel_params_flatten_child_parameter_tuples():
    cyclic = RBFCyclic(
        "k1",
        np.array([1.0, 2.0, 3.0]),
        active_dims=np.array([0, 1, 2]),
    )
    periodic = PeriodicKernel(
        "k2",
        np.array([4.0, 5.0]),
        np.array([2.0 * np.pi, 2.0 * np.pi]),
        active_dims=np.array([3, 4]),
    )

    params = (cyclic * periodic).params

    assert params.shape == (7,)
    assert np.all(np.isfinite(params))
    assert np.allclose(params[:3], [1.0, 2.0, 3.0])
    assert np.allclose(params[3:5], [4.0, 5.0])


def test_sum_kernel_params_flatten_child_parameter_tuples():
    cyclic = RBFCyclic(
        "k1",
        np.array([1.0, 2.0, 3.0]),
        active_dims=np.array([0, 1, 2]),
    )
    periodic = PeriodicKernel(
        "k2",
        np.array([4.0, 5.0]),
        np.array([2.0 * np.pi, 2.0 * np.pi]),
        active_dims=np.array([3, 4]),
    )

    params = (cyclic + periodic).params

    assert params.shape == (7,)
    assert np.all(np.isfinite(params))
