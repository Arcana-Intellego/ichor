import numpy as np
import pytest

from ichor.core.models.kernels import ConstantKernel, PeriodicKernel, RBF, RBFCyclic
from ichor.core.models.kernels.interpreter.parser import Parser


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


def test_rbf_cyclic_mask_targets_only_alf_phi_features():
    kernel = RBFCyclic(
        "k",
        np.ones(12, dtype=float),
        active_dims=np.arange(12),
    )

    np.testing.assert_array_equal(kernel.mask, np.array([5, 8, 11]))


def test_rbf_cyclic_mask_respects_active_dimension_subset():
    kernel = RBFCyclic(
        "k",
        np.ones(3, dtype=float),
        active_dims=np.array([0, 5, 8]),
    )

    np.testing.assert_array_equal(kernel.mask, np.array([1, 2]))


def test_rbf_cyclic_wraps_phi_but_not_inter_axis_angle():
    kernel = RBFCyclic(
        "k",
        np.ones(12, dtype=float),
        active_dims=np.arange(12),
    )
    origin = np.zeros((1, 12), dtype=float)
    phi_wrapped = origin.copy()
    phi_wrapped[0, 5] = 2.0 * np.pi
    angle_unwrapped = origin.copy()
    angle_unwrapped[0, 2] = 2.0 * np.pi

    assert np.isclose(kernel.k(origin, phi_wrapped)[0, 0], 1.0)
    assert kernel.k(origin, angle_unwrapped)[0, 0] < 1.0e-8
    np.testing.assert_allclose(
        kernel.k(origin, origin).diagonal(),
        kernel.k_diag(origin),
    )


def test_rbf_serialises_with_its_exact_native_type():
    kernel = RBF(
        "k1",
        np.array([0.2, 0.3]),
        active_dims=np.array([0, 1]),
    )

    text = kernel.write_str()

    assert "type rbf\n" in text
    assert "type constant" not in text
    assert "active_dimensions 1 2\n" in text


def test_constant_kernel_constructor_and_serialisation_are_well_formed():
    kernel = ConstantKernel(
        "k1",
        2.5,
        active_dims=np.array([0, 2]),
    )

    assert kernel.value == 2.5
    assert repr(kernel) == "ConstantKernel(value=2.5)"
    assert kernel.write_str() == (
        "[kernel.k1]\n"
        "type constant\n"
        "number_of_dimensions 2\n"
        "active_dimensions 1 3\n"
        "value 2.5\n"
    )


def test_kernel_expression_parser_rejects_trailing_tokens():
    with pytest.raises(Exception, match="invalid syntax"):
        Parser("k1 k2").parse()
