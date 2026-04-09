"""C-7: Regression baselines for demo jobs.

Records tie-line endpoints at representative field
values for each demo XML file.  If a future change
shifts a tie-line by more than the tolerance, the
test fails — signalling a potential regression.

Baselines were generated on 2026-04-07 from the
Phase 0 + Phase 1 codebase.
"""

import os

import numpy as np
import pytest

from pde_compute import compute_equilibrium
from pde_input import parse_system_spec


DEMO_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', 'jobs', 'demo')


# -------------------------------------------------------
# Baseline data: (file, field_values, expected_ties)
#
# Each tie is (x0, x1, phase_idx_tuple).
# Tolerance on x endpoints: ±0.01 (hull is discrete
# with 500 points → dx ≈ 0.002).
# -------------------------------------------------------

_BASELINES = [
    # -- Symmetric eutectic (HS form) ---------
    # Phases: 0=liquid, 1=alpha, 2=beta
    ('pde.in.xml',
     {'temperature': 1200.0},
     [(0.006, 0.3868, (1, 0)),
      (0.6132, 0.994, (0, 2))]),

    ('pde.in.xml',
     {'temperature': 1100.0},
     [(0.0621, 0.9379, (1, 2))]),

    # -- Asymmetric eutectic -------------------
    ('asymmetric-eutectic.xml',
     {'temperature': 900.0},
     [(0.1046, 0.954, (1, 2)),
      (0.999, 1.0, (2, 0))]),

    # -- Azeotrope (minimum boiling) -----------
    ('azeotrope.in.xml',
     {'temperature': 375.0},
     [(0.0, 0.3066, (1, 0)),
      (0.6934, 1.0, (0, 1))]),

    # -- Eutectic with intermetallic -----------
    ('eutectic-with-compound.xml',
     {'temperature': 950.0},
     [(0.0, 0.4617, (1, 2)),
      (0.5383, 1.0, (2, 3))]),

    # -- Eutectoid -----------------------------
    ('eutectoid.xml',
     {'temperature': 650.0},
     [(0.0694, 0.8197, (1, 2))]),

    # -- Isomorphous ---------------------------
    ('isomorphous.xml',
     {'temperature': 1200.0},
     [(0.7174, 1.0, (1, 0))]),

    # -- Polynomial asymmetric -----------------
    ('polynomial-asymmetric.xml',
     {'temperature': 900.0},
     [(0.1433, 0.8567, (1, 2))]),

    # -- VLE with pressure ---------------------
    ('vle-pressure.xml',
     {'temperature': 350.0, 'pressure': 1.0},
     [(0.1984, 0.4429, (0, 1))]),
]


@pytest.mark.parametrize(
    'fname, fv, expected_ties',
    _BASELINES,
    ids=[f'{b[0]}@{list(b[1].values())}'
         for b in _BASELINES])
def test_tie_line_regression(fname, fv,
                             expected_ties):
    """Tie-line endpoints must match the baseline
    within tolerance.
    """
    path = os.path.join(DEMO_DIR, fname)
    spec = parse_system_spec(path)
    system = spec.to_system()
    result = compute_equilibrium(system, fv)

    actual_ties = result.two_phase_regions
    assert len(actual_ties) == len(expected_ties), (
        f'Expected {len(expected_ties)} tie lines, '
        f'got {len(actual_ties)}')

    tol = 0.01
    for i, (exp, act) in enumerate(
            zip(expected_ties, actual_ties)):
        exp_x0, exp_x1, exp_phases = exp
        assert act['x0'] == pytest.approx(
            exp_x0, abs=tol), (
            f'Tie {i} x0 mismatch')
        assert act['x1'] == pytest.approx(
            exp_x1, abs=tol), (
            f'Tie {i} x1 mismatch')
        assert tuple(act['phases']) == exp_phases, (
            f'Tie {i} phase mismatch')


# -------------------------------------------------------
# Additional: demos with no tie lines at initial T
# -------------------------------------------------------

_NO_TIE_AT_INITIAL = [
    'pde.in.xml',
    'asymmetric-eutectic.xml',
    'azeotrope.in.xml',
    'eutectic-with-compound.xml',
    'eutectoid.xml',
    'isomorphous.xml',
    'polynomial-asymmetric.xml',
    'vle-pressure.xml',
    'vle-pressure+alpha+beta.xml',
    'vle-pressure+alpha+beta+patch.xml',
]


@pytest.mark.parametrize('fname', _NO_TIE_AT_INITIAL)
def test_no_tie_at_initial_T(fname):
    """At the initial (high) temperature, no two-phase
    regions should exist for these demos — the system
    is fully miscible.
    """
    path = os.path.join(DEMO_DIR, fname)
    spec = parse_system_spec(path)
    system = spec.to_system()
    fv = {fs.name: fs.initial_val
          for fs in spec.fields}
    result = compute_equilibrium(system, fv)
    assert len(result.two_phase_regions) == 0, (
        f'Unexpected tie lines at initial T for '
        f'{fname}')
