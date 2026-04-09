"""C-6: Equilibrium invariants.

Structural properties that must hold for any correct
equilibrium computation:

1. Hull points lie on phase G(x) curves (not below).
2. Single-phase regions have d²G/dx² > 0 (convex).
3. Two-phase boundaries are common-tangent pairs
   (equal G-slope at both endpoints).
"""

import os

import numpy as np
import pytest

from pde_compute import compute_equilibrium
from pde_input import parse_system_spec


DEMO_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', 'jobs', 'demo')

_DEMO_FILES = sorted(
    f for f in os.listdir(DEMO_DIR)
    if f.endswith('.xml'))


def _n_components(fname):
    """Return component count for a demo file
    (cheap: parses spec without building models).
    """
    path = os.path.join(DEMO_DIR, fname)
    spec = parse_system_spec(path)
    return len(spec.components)


def _build_and_compute(fname, fv_override=None):
    """Parse a demo, build the system, compute
    equilibrium at initial (or overridden) field
    values.  Returns (system, result).
    """
    path = os.path.join(DEMO_DIR, fname)
    spec = parse_system_spec(path)
    system = spec.to_system()
    fv = {fs.name: fs.initial_val
          for fs in spec.fields}
    if fv_override:
        fv.update(fv_override)
    result = compute_equilibrium(system, fv)
    return system, result


# -------------------------------------------------------
# Invariant 1: hull points lie on phase curves
# -------------------------------------------------------

@pytest.mark.parametrize('fname', _DEMO_FILES)
def test_hull_on_curves(fname):
    """Every hull vertex G value must match the G(x)
    curve of the phase it belongs to, within
    tolerance.
    """
    system, result = _build_and_compute(fname)
    fv = result.field_values
    hull_x = result.hull_x
    is_binary = hull_x.ndim == 1
    for i in range(len(result.hull_G)):
        hG = result.hull_G[i]
        pi = int(result.hull_phase_idx[i])
        phase = system.phases[pi]
        if is_binary:
            x_pt = np.array([hull_x[i]])
        else:
            x_pt = hull_x[i:i+1, :]
        G_curve = phase.gibbs(x_pt, fv)
        assert hG == pytest.approx(
            float(G_curve.ravel()[0]),
            abs=1e-6), (
            f'Hull vertex {i} (phase '
            f'{phase.name}) off curve')


# -------------------------------------------------------
# Invariant 2: single-phase convexity (d²G/dx² > 0)
# -------------------------------------------------------

@pytest.mark.parametrize('fname', _DEMO_FILES)
def test_hull_envelope_convexity(fname):
    """The lower convex hull envelope must be convex
    (d²G_hull/dx² >= 0) across the entire composition
    range.  Binary only — ternary+ hull convexity
    requires a different check (simplex normals).
    """
    if _n_components(fname) > 2:
        pytest.skip('binary-only invariant')
    _, result = _build_and_compute(fname)
    hull_x = result.hull_x
    hull_G = result.hull_G
    if len(hull_x) < 3:
        return
    # Second derivative of the piecewise-linear hull.
    dx = np.diff(hull_x)
    slopes = np.diff(hull_G) / np.where(
        dx > 1e-12, dx, 1e-12)
    d_slopes = np.diff(slopes)
    # Convexity: slopes must be non-decreasing
    # (allow small numerical noise).
    assert np.all(d_slopes > -1e-3), (
        'Hull envelope is not convex')


# -------------------------------------------------------
# Invariant 3: common tangent at two-phase boundaries
# -------------------------------------------------------

@pytest.mark.parametrize('fname', _DEMO_FILES)
def test_common_tangent(fname):
    """At two-phase region boundaries, the G slope
    (dG/dx) must be approximately equal on both
    sides — the common-tangent condition.
    Binary only — ternary+ uses simplex-face
    tangent planes.
    """
    if _n_components(fname) > 2:
        pytest.skip('binary-only invariant')
    system, result = _build_and_compute(fname)
    fv = result.field_values

    for region in result.two_phase_regions:
        x0, x1 = region['x0'], region['x1']
        if x1 - x0 < 0.02:
            continue
        pi_left = region['phases'][0]
        pi_right = region['phases'][1]
        phase_L = system.phases[pi_left]
        phase_R = system.phases[pi_right]

        # G values at the two endpoints.
        G_L = float(phase_L.gibbs(
            np.array([x0]), fv))
        G_R = float(phase_R.gibbs(
            np.array([x1]), fv))

        # Tangent slope = (G_R - G_L) / (x1 - x0).
        tangent_slope = (G_R - G_L) / (x1 - x0)

        # Numerical dG/dx at x0 on phase_L.
        dx = 1e-5
        x0p = min(x0 + dx, phase_L.xmax)
        if x0p - x0 < 1e-10:
            continue
        G_L_plus = float(phase_L.gibbs(
            np.array([x0p]), fv))
        slope_L = (G_L_plus - G_L) / (x0p - x0)

        # Numerical dG/dx at x1 on phase_R.
        x1m = max(x1 - dx, phase_R.xmin)
        if x1 - x1m < 1e-10:
            continue
        G_R_minus = float(phase_R.gibbs(
            np.array([x1m]), fv))
        slope_R = (G_R - G_R_minus) / (x1 - x1m)

        # Both slopes should approximate the
        # tangent slope.  Tolerance is generous
        # because we use finite differences on
        # a discrete hull.  For CALPHAD models
        # (G in J/mol, slopes ~10⁴) the absolute
        # tolerance scales with the slope
        # magnitude; native models (kJ/mol,
        # slopes <100) keep the tight 0.5 bound.
        tol = max(
            0.5,
            abs(tangent_slope) * 0.01)
        assert slope_L == pytest.approx(
            tangent_slope, abs=tol), (
            f'Left slope mismatch at '
            f'x={x0:.4f} ({phase_L.name})')
        assert slope_R == pytest.approx(
            tangent_slope, abs=tol), (
            f'Right slope mismatch at '
            f'x={x1:.4f} ({phase_R.name})')
