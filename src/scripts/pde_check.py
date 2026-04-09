#!/usr/bin/env python3
"""
Thermodynamic consistency checker for PDE.

Public API
----------
run_all_checks(system, T=None, P=None) -> list[ConsistencyWarning]
    Run all available checks and return warnings sorted by severity then name.

Individual checks (each returns list[ConsistencyWarning]):
    check_phase_coverage(system)
    check_convexity(system, T=None, P=None)
    check_vle_terminal_tangency(system, P=None)
    check_vle_phase_ordering(system, P=None)
    check_end_member_gmatch(system, P=None)

Design principles
-----------------
* Non-blocking: all functions are pure and have no Qt dependency.
* Composable: each check is independent and can be run individually.
* Explanatory: every warning carries a one-line message *and* a detail string
  that explains the physical meaning and suggests a fix.
* Model-agnostic: checks work on evaluated G arrays via Phase.gibbs(); they do
  not inspect energy-model internals.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from scipy.optimize import brentq as _brentq
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO    = 'INFO'
    WARNING = 'WARNING'
    ERROR   = 'ERROR'


# ---------------------------------------------------------------------------
# ConsistencyWarning
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyWarning:
    """One diagnostic item returned by a consistency check.

    Attributes
    ----------
    severity    : Severity — ERROR > WARNING > INFO
    check_name  : str      — machine tag, e.g. 'vle_terminal_tangency'
    phase_names : list[str]— phases implicated; [] for system-level warnings
    message     : str      — one-line human-readable summary
    detail      : str      — physical meaning + suggested fix
    quantity    : float    — measured discrepancy (0.0 if not applicable)
    T_ref       : float | None — temperature at which the check was made
    P_ref       : float | None — pressure at which the check was made
    fix_delta   : float | None — when not None, adding this to hs_H[0] of
                                 phase_names[0] removes the G mismatch exactly
    """
    severity:    Severity
    check_name:  str
    phase_names: list[str]
    message:     str
    detail:      str
    quantity:    float = 0.0
    T_ref:       Optional[float] = None
    P_ref:       Optional[float] = None
    fix_delta:   Optional[float] = None
    fix_hs:      Optional[dict]  = None
    # fix_hs schema: {'target_name': str, 'dH': [ΔH₀,ΔH₁,ΔH₂], 'dS': [ΔS₀,ΔS₁,ΔS₂]}
    # Represents minimum-norm H+S correction (lstsq 4×6) to satisfy all 4
    # consistency conditions (equal G and slope at both pure-component endpoints).


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_G_TOL       = 5e-2   # absolute G tolerance, kJ/mol scale
_G_REL_TOL   = 1e-4   # relative tolerance for large-G systems
_SLOPE_TOL   = 1e-1   # dG/dx tolerance for tangency checks
_N_BP_SCAN   = 500    # T scan points for boiling-point / crossing search
_N_CONV_POINTS = 200  # composition points for convexity scan


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _numerical_slope(phase, x: float, T: float, P: float) -> float:
    """Central-difference dG/dx at composition x, clamped to the phase range."""
    dx = 1e-5
    x_lo = max(phase.xmin, x - dx)
    x_hi = min(phase.xmax, x + dx)
    if x_hi <= x_lo:
        return 0.0
    G_lo = float(phase.gibbs(x_lo, {'temperature': T, 'pressure': P}))
    G_hi = float(phase.gibbs(x_hi, {'temperature': T, 'pressure': P}))
    return (G_hi - G_lo) / (x_hi - x_lo)


def _find_crossing_T(
    phase_a, phase_b, x_val: float,
    T_min: float, T_max: float, P: float, n: int
) -> Optional[float]:
    """Find T where G_a(x_val, T) == G_b(x_val, T) via scan + brentq.

    Returns the crossing temperature as a float, or None if no crossing is
    found in [T_min, T_max] or if scipy is unavailable.
    """
    if not _HAVE_SCIPY:
        return None

    T_arr = np.linspace(T_min, T_max, n)
    try:
        dG = np.array([
            float(phase_a.gibbs(x_val, {'temperature': T, 'pressure': P})) - float(phase_b.gibbs(x_val, {'temperature': T, 'pressure': P}))
            for T in T_arr
        ])
    except Exception:
        return None

    # Find adjacent pairs with a sign change
    sign_changes = np.where(np.diff(np.sign(dG)))[0]
    if len(sign_changes) == 0:
        return None

    i = sign_changes[0]
    T_lo, T_hi = float(T_arr[i]), float(T_arr[i + 1])
    try:
        T_cross = _brentq(
            lambda T: float(phase_a.gibbs(x_val, {'temperature': T, 'pressure': P})) - float(phase_b.gibbs(x_val, {'temperature': T, 'pressure': P})),
            T_lo, T_hi, xtol=1e-6 * (T_max - T_min), maxiter=100
        )
        return float(T_cross)
    except Exception:
        return None


def _compute_fix_hs(target_name: str, endpoint_data: dict) -> Optional[dict]:
    """Compute minimum-norm (ΔH₀..₂, ΔS₀..₂) correction so that *target_name*
    (the gas phase) satisfies all 4 consistency conditions simultaneously:
      equal G and equal dG/dx with the liquid at both x=0 (component A) and
      x=1 (component B).

    The unknown vector is [ΔH₀, ΔH₁, ΔH₂, ΔS₀, ΔS₁, ΔS₂].
    The 4×6 system is underdetermined; numpy lstsq returns the minimum-norm
    solution.  Returns None when the solve fails.
    """
    try:
        (T_A, G_gas_A, G_liq_A, s_gas_A, s_liq_A) = endpoint_data[0.0]
        (T_B, G_gas_B, G_liq_B, s_gas_B, s_liq_B) = endpoint_data[1.0]
    except KeyError:
        return None
    eps = [
        G_liq_A - G_gas_A,    # ΔG at x=0
        s_liq_A - s_gas_A,    # Δslope at x=0
        G_liq_B - G_gas_B,    # ΔG at x=1
        s_liq_B - s_gas_B,    # Δslope at x=1
    ]
    A = np.array([
        [1, 0, 0, -T_A,      0,       0      ],
        [0, 1, 0,  0,        -T_A,    0      ],
        [1, 1, 1, -T_B,      -T_B,    -T_B   ],
        [0, 1, 2,  0,        -T_B,    -2*T_B ],
    ])
    try:
        x_sol, *_ = np.linalg.lstsq(A, eps, rcond=None)
        return {'target_name': target_name,
                'dH': x_sol[:3].tolist(),
                'dS': x_sol[3:].tolist()}
    except Exception:
        return None


def _coverage_warning(x0: float, x1: float) -> ConsistencyWarning:
    """Factory for a composition coverage gap warning."""
    return ConsistencyWarning(
        severity=Severity.WARNING,
        check_name='phase_coverage',
        phase_names=[],
        message=f'Composition gap: no phase covers x ∈ [{x0:.3f}, {x1:.3f}]',
        detail=(
            'The convex hull will be undefined in this composition interval. '
            'Add a phase whose x range covers the gap, or extend an existing one.'
        ),
        quantity=x1 - x0,
    )


# ---------------------------------------------------------------------------
# Public check functions
# ---------------------------------------------------------------------------

def check_phase_coverage(system) -> list[ConsistencyWarning]:
    """Check that the composition axis [0, 1] is fully covered by phases.

    Returns
    -------
    list[ConsistencyWarning]
        WARNING per uncovered gap; ERROR if no non-point phases exist at all.
    """
    non_point = [p for p in system.phases if not p.is_point]
    if not non_point:
        return [ConsistencyWarning(
            severity=Severity.ERROR,
            check_name='phase_coverage',
            phase_names=[],
            message='No extended phases defined',
            detail=(
                'All phases are end-members (point phases). '
                'Add at least one phase with a non-degenerate composition range.'
            ),
        )]

    intervals = sorted([(p.xmin, p.xmax) for p in non_point], key=lambda t: t[0])

    warnings = []
    # Check left edge
    if intervals[0][0] > _G_TOL:
        warnings.append(_coverage_warning(0.0, intervals[0][0]))

    # Sweep through
    covered_up_to = intervals[0][1]
    for xmin, xmax in intervals[1:]:
        if xmin > covered_up_to + _G_TOL:
            warnings.append(_coverage_warning(covered_up_to, xmin))
        covered_up_to = max(covered_up_to, xmax)

    # Check right edge
    if covered_up_to < 1.0 - _G_TOL:
        warnings.append(_coverage_warning(covered_up_to, 1.0))

    return warnings


def check_convexity(system, T=None, P=None) -> list[ConsistencyWarning]:
    """Check each phase for local non-convexity (d²G/dx² < 0).

    A negative second derivative implies spinodal decomposition within the
    phase, which may cause numerical instability in the convex-hull step.

    Returns
    -------
    list[ConsistencyWarning]
        INFO per phase that shows a non-convex region.
    """
    if T is None:
        T = system.T_initial
    if P is None:
        P = system.P_initial

    warnings = []
    for phase in system.phases:
        if phase.is_point:
            continue
        x = phase.composition_grid(_N_CONV_POINTS)
        try:
            fv = {'temperature': T,
                  'pressure': P}
            G = np.array([
                float(phase.gibbs(xi, fv))
                for xi in x], dtype=float)
        except Exception:
            continue

        d2G = np.gradient(np.gradient(G, x), x)
        min_d2G = float(np.min(d2G))
        if min_d2G < -_G_TOL:
            x_min_idx = int(np.argmin(d2G))
            x_at_min = float(x[x_min_idx])
            warnings.append(ConsistencyWarning(
                severity=Severity.INFO,
                check_name='convexity',
                phase_names=[phase.name],
                message=(
                    f'{phase.name}: non-convex at x ≈ {x_at_min:.3f} '
                    f'(d²G/dx² = {min_d2G:.3g})'
                ),
                detail=(
                    'A negative d²G/dx² indicates spinodal instability within the '
                    'phase. This is physically meaningful for miscibility-gap systems '
                    'but may cause the convex hull to miss a tie-line endpoint. '
                    'Verify that the G(x) shape is intentional.'
                ),
                quantity=min_d2G,
                T_ref=T,
                P_ref=P,
            ))
    return warnings


def check_vle_terminal_tangency(system, P=None) -> list[ConsistencyWarning]:
    """Check that each (gas, liquid) pair shares a common tangent at x=0 and x=1.

    At a pure-component boiling point the gas and liquid curves must be
    tangent (equal G *and* equal dG/dx).  Mismatches indicate that the
    energy parameters are inconsistent at the composition endpoints.

    For polynomial-form systems a note is appended to the detail string
    because the check uses phase_type labels (not the functional form).

    Returns
    -------
    list[ConsistencyWarning]
        INFO if no boiling point is found in the T range;
        WARNING if G or slope mismatch exceeds tolerance at the boiling point.
    """
    if not system.gas_phases or not system.liquid_phases:
        return []

    if P is None:
        P = system.P_initial

    T_min = system.T_min
    T_max = system.T_max
    poly_note = (
        ' [Note: check uses phase_type labels; '
        'verify that polynomial coefficients encode the intended phase.]'
        if system.energy_form == 'polynomial' else ''
    )

    warnings = []
    for gas in system.gas_phases:
        for liq in system.liquid_phases:
            # ---- Pass 1: collect data for both endpoints ----
            # endpoint_data: x_pure → (T_bp, G_gas, G_liq, slope_gas, slope_liq)
            endpoint_data = {}
            no_bp_endpoints = []   # (x_pure, comp_label) where T_bp was not found

            for x_pure, comp_label in ((0.0, 'A'), (1.0, 'B')):
                if not (gas.xmin <= x_pure <= gas.xmax):
                    continue
                if not (liq.xmin <= x_pure <= liq.xmax):
                    continue

                T_bp = _find_crossing_T(liq, gas, x_pure, T_min, T_max, P, _N_BP_SCAN)
                if T_bp is None:
                    no_bp_endpoints.append((x_pure, comp_label))
                    continue

                try:
                    fv_bp = {
                        'temperature': T_bp,
                        'pressure': P}
                    G_gas_val = float(
                        gas.gibbs(x_pure, fv_bp))
                    G_liq_val = float(
                        liq.gibbs(x_pure, fv_bp))
                    slope_gas  = _numerical_slope(gas, x_pure, T_bp, P)
                    slope_liq  = _numerical_slope(liq, x_pure, T_bp, P)
                except Exception:
                    continue
                endpoint_data[x_pure] = (T_bp, G_gas_val, G_liq_val, slope_gas, slope_liq)

            # ---- Compute fix_hs when both endpoints are available ----
            fix_hs = None
            if len(endpoint_data) == 2:
                fix_hs = _compute_fix_hs(gas.name, endpoint_data)

            # ---- Pass 2: emit no-boiling-point INFO warnings ----
            for x_pure, comp_label in no_bp_endpoints:
                warnings.append(ConsistencyWarning(
                    severity=Severity.INFO,
                    check_name='vle_terminal_tangency',
                    phase_names=[gas.name, liq.name],
                    message=(
                        f'No boiling point for {comp_label} '
                        f'({gas.name}/{liq.name}) in T range'
                    ),
                    detail=(
                        f'G_gas − G_liquid does not change sign at x={x_pure:.0f} '
                        f'over T ∈ [{T_min}, {T_max}]. '
                        'The gas and liquid may never cross in this range. '
                        'Extend the T range or adjust H/S values.'
                        + poly_note
                    ),
                    T_ref=None,
                    P_ref=P,
                ))

            # ---- Pass 3: emit mismatch WARNING per endpoint, attaching fix_hs ----
            for x_pure, comp_label in ((0.0, 'A'), (1.0, 'B')):
                if x_pure not in endpoint_data:
                    continue
                T_bp, G_gas, G_liq, slope_gas, slope_liq = endpoint_data[x_pure]

                dG_abs = abs(G_gas - G_liq)
                tol_G  = max(_G_TOL, _G_REL_TOL * max(abs(G_gas), abs(G_liq), 1.0))

                if dG_abs > tol_G:
                    fix_delta = G_liq - G_gas   # shifts gas H₀ so G_gas = G_liq
                    warnings.append(ConsistencyWarning(
                        severity=Severity.WARNING,
                        check_name='vle_terminal_tangency',
                        phase_names=[gas.name, liq.name],
                        message=(
                            f'G mismatch at {comp_label} boiling point '
                            f'({gas.name}/{liq.name}): |ΔG| = {dG_abs:.4g}'
                        ),
                        detail=(
                            f'At T_bp ≈ {T_bp:.1f}, x = {x_pure:.0f}: '
                            f'G_gas = {G_gas:.4g}, G_liq = {G_liq:.4g}. '
                            'The two phases should be degenerate at the boiling point. '
                            'Adjust H or S coefficients to equalise G at this endpoint.'
                            + poly_note
                        ),
                        quantity=dG_abs,
                        T_ref=T_bp,
                        P_ref=P,
                        fix_delta=fix_delta,
                        fix_hs=fix_hs,
                    ))
                else:
                    dslope = abs(slope_gas - slope_liq)
                    if dslope > _SLOPE_TOL:
                        warnings.append(ConsistencyWarning(
                            severity=Severity.WARNING,
                            check_name='vle_terminal_tangency',
                            phase_names=[gas.name, liq.name],
                            message=(
                                f'Slope mismatch at {comp_label} boiling point '
                                f'({gas.name}/{liq.name}): |Δ(dG/dx)| = {dslope:.4g}'
                            ),
                            detail=(
                                f'At T_bp ≈ {T_bp:.1f}, x = {x_pure:.0f}: '
                                f'dG/dx(gas) = {slope_gas:.4g}, '
                                f'dG/dx(liq) = {slope_liq:.4g}. '
                                'A common tangent requires equal slopes at the endpoint. '
                                'Adjust S₁ (or higher coefficients) to match slopes.'
                                + poly_note
                            ),
                            quantity=dslope,
                            T_ref=T_bp,
                            P_ref=P,
                            fix_hs=fix_hs,
                        ))

    return warnings


def check_end_member_gmatch(system, P=None) -> list[ConsistencyWarning]:
    """Check that each end-member's G matches any overlapping phase at some T.

    End-members represent stable stoichiometric compounds.  If no extended
    phase crosses the end-member's G in the system's T range, the compound
    will be stable at all temperatures, which may not be the user's intent.

    Returns
    -------
    list[ConsistencyWarning]
        INFO if no crossing is found (end-member always below/above);
        WARNING if the G values diverge at the crossing temperature.
    """
    if not system.end_members:
        return []

    if P is None:
        P = system.P_initial

    T_min = system.T_min
    T_max = system.T_max

    warnings = []
    for em in system.end_members:
        x_end = em.xmin   # end-members have xmin == xmax

        # Find non-point phases whose range covers x_end
        candidates = [
            p for p in system.phases
            if not p.is_point and p.xmin <= x_end <= p.xmax
        ]
        if not candidates:
            continue

        for cand in candidates:
            T_trans = _find_crossing_T(
                em, cand, x_end, T_min, T_max, P, _N_BP_SCAN
            )

            if T_trans is None:
                warnings.append(ConsistencyWarning(
                    severity=Severity.INFO,
                    check_name='end_member_gmatch',
                    phase_names=[em.name, cand.name],
                    message=(
                        f'No transition for end-member {em.name} vs {cand.name} '
                        f'at x = {x_end:.3f}'
                    ),
                    detail=(
                        f'G_{em.name} − G_{cand.name} does not change sign at '
                        f'x = {x_end:.3f} over T ∈ [{T_min}, {T_max}]. '
                        'The end-member may be stable (or unstable) across the '
                        'entire T range. Verify that the H/S values are correct.'
                    ),
                    P_ref=P,
                ))
                continue

            try:
                fv_tr = {
                    'temperature': T_trans,
                    'pressure': P}
                G_em = float(
                    em.gibbs(x_end, fv_tr))
                G_cand = float(
                    cand.gibbs(x_end, fv_tr))
            except Exception:
                continue

            dG_abs = abs(G_em - G_cand)
            tol_G = max(_G_TOL, _G_REL_TOL * max(abs(G_em), abs(G_cand), 1.0))

            if dG_abs > tol_G:
                fix_delta = G_cand - G_em   # shifts end-member H₀ so G_em = G_cand
                warnings.append(ConsistencyWarning(
                    severity=Severity.WARNING,
                    check_name='end_member_gmatch',
                    phase_names=[em.name, cand.name],
                    message=(
                        f'G mismatch at {em.name}/{cand.name} transition '
                        f'(x = {x_end:.3f}): |ΔG| = {dG_abs:.4g}'
                    ),
                    detail=(
                        f'At T_trans ≈ {T_trans:.1f}, x = {x_end:.3f}: '
                        f'G_{em.name} = {G_em:.4g}, G_{cand.name} = {G_cand:.4g}. '
                        'The end-member and host phase should be degenerate at the '
                        'transition temperature. Adjust H or S values to equalise G.'
                    ),
                    quantity=dG_abs,
                    T_ref=T_trans,
                    P_ref=P,
                    fix_delta=fix_delta,
                ))

    return warnings


def check_vle_phase_ordering(system, P=None) -> list[ConsistencyWarning]:
    """Check that G_vapor < G_liquid above the boiling point (and vice versa).

    At a pure-component boiling point T_bp, G_gas = G_liq.  For T > T_bp the
    vapour must be the thermodynamically stable phase (G_gas < G_liq), and for
    T < T_bp the liquid must be stable (G_liq < G_gas).

    An inverted ordering — where the vapour is already more stable *below* T_bp,
    or the liquid remains more stable *above* T_bp — indicates that the entropy
    difference S_gas − S_liq is too small or has the wrong sign.  The physical
    requirement is S_gas > S_liq so that G_gas decreases faster with T and
    crosses below G_liq at T_bp.

    This check complements check_vle_terminal_tangency: the terminal-tangency
    check verifies G equality and dG/dx (composition slope) at T_bp; this check
    verifies the sign of dG/dT (temperature slope, i.e. entropy ordering) on
    both sides of T_bp.

    Returns
    -------
    list[ConsistencyWarning]
        WARNING per pure-component endpoint where the stability ordering is
        inverted above or below the boiling point.
    """
    if not system.gas_phases or not system.liquid_phases:
        return []

    if P is None:
        P = system.P_initial

    T_min  = system.T_min
    T_max  = system.T_max
    delta_T = max(0.02 * (T_max - T_min), 1.0)

    warnings = []
    for gas in system.gas_phases:
        for liq in system.liquid_phases:
            for x_pure, comp_label in ((0.0, 'A'), (1.0, 'B')):
                if not (gas.xmin <= x_pure <= gas.xmax):
                    continue
                if not (liq.xmin <= x_pure <= liq.xmax):
                    continue

                T_bp = _find_crossing_T(
                    liq, gas, x_pure, T_min, T_max, P, _N_BP_SCAN
                )
                if T_bp is None:
                    continue   # no boiling point in range — flagged elsewhere

                T_above = T_bp + delta_T
                T_below = T_bp - delta_T

                try:
                    if T_above <= T_max:
                        fv_ab = {
                            'temperature': T_above,
                            'pressure': P}
                        G_gas_ab = float(
                            gas.gibbs(x_pure, fv_ab))
                        G_liq_ab = float(
                            liq.gibbs(x_pure, fv_ab))
                        dG_above = (
                            G_liq_ab - G_gas_ab)
                    else:
                        dG_above = None

                    if T_below >= T_min:
                        fv_bel = {
                            'temperature': T_below,
                            'pressure': P}
                        G_gas_bel = float(
                            gas.gibbs(
                                x_pure, fv_bel))
                        G_liq_bel = float(
                            liq.gibbs(
                                x_pure, fv_bel))
                        dG_below = (
                            G_liq_bel - G_gas_bel)
                    else:
                        dG_below = None
                except Exception:
                    continue

                if dG_above is not None and dG_above <= 0:
                    # fix: ΔS₀ shifts G_gas down at T>T_bp; ΔH₀ keeps G_gas(T_bp) fixed
                    deficit_ab = G_gas_ab - G_liq_ab          # > 0
                    dS0_ab     = deficit_ab / delta_T * 1.1
                    fix_hs_ab  = {'target_name': gas.name,
                                  'dH': [T_bp * dS0_ab, 0., 0.],
                                  'dS': [dS0_ab, 0., 0.]}
                    warnings.append(ConsistencyWarning(
                        severity=Severity.WARNING,
                        check_name='vle_phase_ordering',
                        phase_names=[gas.name, liq.name],
                        message=(
                            f'Vapour not stable above {comp_label} boiling point '
                            f'({gas.name}/{liq.name}): '
                            f'G_vapor ≥ G_liquid at T = {T_above:.1f}'
                        ),
                        detail=(
                            f'At T = {T_above:.1f} (above T_bp ≈ {T_bp:.1f}), '
                            f'G_vapor = {G_gas_ab:.4g}, G_liquid = {G_liq_ab:.4g}. '
                            'Above the boiling point the vapour must be the stable '
                            'phase (G_vapor < G_liquid).  The stability ordering is '
                            'governed by entropy: G decreases with T at rate −S, so '
                            'the phase with higher S wins at high T.  '
                            'Increase S₀ (or higher-order S coefficients) of the '
                            'vapour phase so that S_gas > S_liquid.'
                        ),
                        quantity=abs(dG_above),
                        T_ref=T_bp,
                        P_ref=P,
                        fix_hs=fix_hs_ab,
                    ))

                if dG_below is not None and dG_below >= 0:
                    # same root cause — same fix (deficit is on the other side of T_bp)
                    deficit_bel = G_liq_bel - G_gas_bel       # > 0
                    dS0_bel     = deficit_bel / delta_T * 1.1
                    fix_hs_bel  = {'target_name': gas.name,
                                   'dH': [T_bp * dS0_bel, 0., 0.],
                                   'dS': [dS0_bel, 0., 0.]}
                    warnings.append(ConsistencyWarning(
                        severity=Severity.WARNING,
                        check_name='vle_phase_ordering',
                        phase_names=[gas.name, liq.name],
                        message=(
                            f'Liquid not stable below {comp_label} boiling point '
                            f'({gas.name}/{liq.name}): '
                            f'G_liquid ≥ G_vapor at T = {T_below:.1f}'
                        ),
                        detail=(
                            f'At T = {T_below:.1f} (below T_bp ≈ {T_bp:.1f}), '
                            f'G_liquid = {G_liq_bel:.4g}, G_vapor = {G_gas_bel:.4g}. '
                            'Below the boiling point the liquid must be the stable '
                            'phase (G_liquid < G_vapor).  This is the complementary '
                            'failure of an ordering inversion: S_gas − S_liquid is '
                            'too small or negative.  '
                            'Increase S₀ of the vapour phase relative to the liquid.'
                        ),
                        quantity=abs(dG_below),
                        T_ref=T_bp,
                        P_ref=P,
                        fix_hs=fix_hs_bel,
                    ))

    return warnings


def check_vle_params_valid(system, P=None) -> list[ConsistencyWarning]:
    """Check that VLE parameters are physically valid.

    For each gas phase whose energy model carries a ``vle_params`` attribute
    (set by ``compute_vle_gas_hs()``), verify:

      * L_A, L_B > 0  (positive latent heats)
      * T_bp_A, T_bp_B are within the system's T range

    Returns an INFO warning when a boiling point lies outside the T range
    (not necessarily wrong, just worth flagging), and an ERROR when a latent
    heat is non-positive (physically meaningless).

    Parameters
    ----------
    system : System
    P      : unused (kept for API symmetry with other check functions)

    Returns
    -------
    list[ConsistencyWarning]
    """
    warnings = []
    T_min = system.T_min
    T_max = system.T_max

    for gas in system.gas_phases:
        vp = getattr(gas.energy_model, 'vle_params', None)
        if vp is None:
            continue

        for key, label in (('L_A', 'L_A'), ('L_B', 'L_B')):
            val = vp.get(key, 0.0)
            if val <= 0.0:
                warnings.append(ConsistencyWarning(
                    severity=Severity.ERROR,
                    check_name='vle_params_valid',
                    phase_names=[gas.name],
                    message=f'{gas.name}: {label} = {val:.4g} ≤ 0 (latent heat must be positive)',
                    detail=(
                        f'A non-positive latent heat ({label} = {val:.4g}) is unphysical. '
                        'VLE thermodynamics requires L > 0 so that the vapour phase has '
                        'higher entropy than the liquid.  Set a positive value.'
                    ),
                    quantity=val,
                ))

        for key, label, x_label in (
            ('T_bp_A', 'T_bp_A', 'x=0 (component A)'),
            ('T_bp_B', 'T_bp_B', 'x=1 (component B)'),
        ):
            T_bp = vp.get(key, 0.0)
            if not (T_min <= T_bp <= T_max):
                warnings.append(ConsistencyWarning(
                    severity=Severity.INFO,
                    check_name='vle_params_valid',
                    phase_names=[gas.name],
                    message=(
                        f'{gas.name}: {label} = {T_bp:.1f} K is outside '
                        f'T range [{T_min:.1f}, {T_max:.1f}] K'
                    ),
                    detail=(
                        f'The boiling point at {x_label} ({T_bp:.1f} K) lies '
                        f'outside the displayed temperature range [{T_min:.1f}, {T_max:.1f}] K. '
                        'The VLE geometry is still well-defined, but the phase boundary '
                        'will not appear in the diagram.  Adjust T_min/T_max or the '
                        'boiling-point parameter if this is unintentional.'
                    ),
                    quantity=T_bp,
                ))

    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_checks(system, T=None, P=None) -> list[ConsistencyWarning]:
    """Run all consistency checks and return results sorted by severity.

    Parameters
    ----------
    system : System
    T      : float | None — reference temperature (defaults to system.T_initial)
    P      : float | None — reference pressure (defaults to system.P_initial)

    Returns
    -------
    list[ConsistencyWarning] — sorted ERROR → WARNING → INFO, then by check_name.
    """
    # Determine whether to run the old tangency / ordering checks.
    # For gas phases with vle_params, all constraints are satisfied by
    # construction; running the numerical checks would be redundant noise.
    # Run them only when at least one gas phase lacks vle_params.
    vle_gas   = [g for g in system.gas_phases
                 if getattr(g.energy_model, 'vle_params', None) is not None]
    plain_gas = [g for g in system.gas_phases
                 if getattr(g.energy_model, 'vle_params', None) is None]
    run_tangency = bool(plain_gas) or not bool(system.gas_phases)

    all_warnings = check_phase_coverage(system)
    if run_tangency:
        all_warnings += check_vle_terminal_tangency(system, P=P)
        all_warnings += check_vle_phase_ordering(system, P=P)
    if vle_gas:
        all_warnings += check_vle_params_valid(system, P=P)
    all_warnings += (
        check_end_member_gmatch(system, P=P)
        + check_convexity(system, T=T, P=P)
    )
    severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    all_warnings.sort(key=lambda w: (severity_order[w.severity], w.check_name))
    return all_warnings
