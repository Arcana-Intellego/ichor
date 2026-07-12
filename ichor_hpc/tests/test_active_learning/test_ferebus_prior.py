from types import SimpleNamespace

import numpy as np
import pytest

from ichor.hpc.active_learning.config import CampaignConfig, ConfigValidationError
from ichor.hpc.active_learning.ferebus_prior import (
    FerebusPriorError,
    contract_from_payload,
    element_from_atom_label,
    resolve_ferebus_prior_contract,
    validate_model_prior_mean,
)


class _ConstantMean:
    def __init__(self, value):
        self.value_ha = float(value)

    def value(self, x):
        return np.full((len(x),), self.value_ha)


def _model(mean):
    return SimpleNamespace(nfeats=3, mean=_ConstantMean(mean))


def test_default_prior_resolves_from_gaussian_level():
    contract = resolve_ferebus_prior_contract(CampaignConfig())
    assert contract.mean_type == 21
    assert contract.level_of_theory == "b3lyp/aug-cc-pvtz"
    assert contract.feature_scaling is True
    assert contract.property_scaling is False
    assert contract.expected_mean_ha("iqa", "H2") == pytest.approx(
        -0.502259675743
    )
    assert contract.expected_mean_ha("iqa", "O1") == pytest.approx(
        -75.0941778191
    )
    assert contract.expected_mean_ha("q00", "O1") == 0.0
    assert contract_from_payload(contract.to_dict()) == contract


def test_explicit_prior_level_must_match_gaussian_training_level():
    cfg = CampaignConfig()
    cfg.ferebus.prior_mean_level_of_theory = "b3lyp/def2-tzvp"
    with pytest.raises(FerebusPriorError, match="does not match"):
        resolve_ferebus_prior_contract(cfg)


def test_type_21_rejects_property_scaling_and_unsupported_elements():
    cfg = CampaignConfig()
    cfg.ferebus.property_scaling = True
    with pytest.raises(ConfigValidationError, match="property_scaling must be false"):
        cfg._validate()
    assert element_from_atom_label("C1") == "C"
    with pytest.raises(FerebusPriorError, match="no isolated-atom"):
        element_from_atom_label("Cl1")


def test_model_prior_mean_is_enforced_for_iqa_and_auxiliary_properties():
    contract = resolve_ferebus_prior_contract(CampaignConfig())
    evidence = validate_model_prior_mean(
        _model(-75.0941778191),
        contract=contract,
        property_name="iqa",
        atom="O1",
    )
    assert evidence["observed_mean_ha"] == pytest.approx(-75.0941778191)
    validate_model_prior_mean(
        _model(0.0),
        contract=contract,
        property_name="q00",
        atom="O1",
    )
    with pytest.raises(FerebusPriorError, match="prior mean mismatch"):
        validate_model_prior_mean(
            _model(-1.0),
            contract=contract,
            property_name="iqa",
            atom="O1",
        )
