from ichor.core.common.constants import Orbitals, type2orbital


def test_orbitals_import_and_addition_contract():
    assert Orbitals.S.name == "S"
    assert Orbitals.S.value == 2

    sp = Orbitals.S + Orbitals.P
    assert sp.name == "SP"
    assert sp.value == 8

    assert type2orbital["H"].value == 2
    assert type2orbital["C"].name == "SP"
    assert type2orbital["C"].value == 8
