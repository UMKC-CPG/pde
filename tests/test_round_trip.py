"""C-5: XML round-trip invariants.

parse_system_spec(file) -> spec.to_xml_str() ->
parse_system_spec(tmp) must produce an equivalent
SystemSpec: same title, components, energy_form,
fields, and phases (within floating-point tolerance).
"""

import os
import tempfile

import numpy as np
import pytest

from pde_input import parse_system_spec


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

DEMO_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', 'jobs', 'demo')


def _round_trip(path):
    """Parse → serialise → re-parse and return both
    the original and round-tripped SystemSpec objects.

    The temp XML is written to the same directory as
    the original so that relative TDB paths resolve
    correctly in the re-parsed spec.
    """
    spec1 = parse_system_spec(path)
    xml = spec1.to_xml_str()
    parent = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile(
            suffix='.xml', mode='w',
            dir=parent, delete=False) as fh:
        fh.write(xml)
        tmp = fh.name
    try:
        spec2 = parse_system_spec(tmp)
    finally:
        os.unlink(tmp)
    return spec1, spec2


def _assert_specs_equal(s1, s2, tol=1e-8):
    """Assert two SystemSpec objects are equivalent
    within floating-point tolerance.
    """
    assert s1.title == s2.title
    assert s1.components == s2.components
    assert s1.energy_form == s2.energy_form
    assert len(s1.fields) == len(s2.fields)
    for f1, f2 in zip(s1.fields, s2.fields):
        assert f1.name == f2.name
        assert f1.symbol == f2.symbol
        assert f1.min_val == pytest.approx(
            f2.min_val, abs=tol)
        assert f1.max_val == pytest.approx(
            f2.max_val, abs=tol)
        assert f1.initial_val == pytest.approx(
            f2.initial_val, abs=tol)
    assert len(s1.phases) == len(s2.phases)
    for p1, p2 in zip(s1.phases, s2.phases):
        assert p1.name == p2.name
        assert p1.phase_type == p2.phase_type
        assert p1.xmin == pytest.approx(
            p2.xmin, abs=tol)
        assert p1.xmax == pytest.approx(
            p2.xmax, abs=tol)
        assert p1.model_type == p2.model_type


def _assert_systems_produce_same_G(s1, s2, fv,
                                   tol=1e-6):
    """Assert that to_system() from both specs
    produces identical G(x) curves at the given
    field values.
    """
    sys1 = s1.to_system()
    sys2 = s2.to_system()
    assert len(sys1.phases) == len(sys2.phases)
    for ph1, ph2 in zip(sys1.phases, sys2.phases):
        assert ph1.name == ph2.name
        x_ph = ph1.composition_grid(
            50 if not ph1.is_point else 1)
        G1 = ph1.gibbs(x_ph, fv)
        G2 = ph2.gibbs(x_ph, fv)
        np.testing.assert_allclose(
            G1, G2, atol=tol,
            err_msg=(
                f'G mismatch for {ph1.name}'))


# -------------------------------------------------------
# Tests — one per demo file
# -------------------------------------------------------

_DEMO_FILES = sorted(
    f for f in os.listdir(DEMO_DIR)
    if f.endswith('.xml'))


@pytest.mark.parametrize('fname', _DEMO_FILES)
def test_round_trip_spec(fname):
    """Spec metadata survives the round trip."""
    path = os.path.join(DEMO_DIR, fname)
    s1, s2 = _round_trip(path)
    _assert_specs_equal(s1, s2)


@pytest.mark.parametrize('fname', _DEMO_FILES)
def test_round_trip_G_values(fname):
    """G(x) curves match after the round trip."""
    path = os.path.join(DEMO_DIR, fname)
    s1, s2 = _round_trip(path)
    fv = {fs.name: fs.initial_val
          for fs in s1.fields}
    _assert_systems_produce_same_G(s1, s2, fv)
