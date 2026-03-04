#!/usr/bin/env python3
"""
Energy model classes for PDE.

Hierarchy:
  EnergyModel  (ABC)
  ├── HSModel     G(x,T,P) = H(x) - T·S(x) + P·V(x) [+ R·T·ln(P/P°)]
  └── PolyModel   G(x,T,P) = Σᵢ cᵢ(T)·xⁱ + P·V(x) [+ R·T·ln(P/P°)]

Both classes use ascending-order coefficient conventions throughout:
  [c0, c1, c2, ...] means c0 + c1·x + c2·x² + ...
  [a0, a1, a2, ...] means a0 + a1·T + a2·T² + ...

End-member (point) phases are represented naturally:
  HSModel  with single-element H and S lists  → G(T) = H[0] - T·S[0]
  PolyModel with only an x⁰ T-coefficient list → G(T) = Σⱼ a₀ⱼ·Tʲ

Generalised interface
---------------------
EnergyModel.gibbs() accepts both calling conventions:

  Old (positional):  model.gibbs(x, T)  or  model.gibbs(x, T, P)
  New (field dict):  model.gibbs(x, {'temperature': T, 'pressure': P})

Both map to the internal _gibbs_impl(x, field_values) which subclasses
implement.  The shim handles type-dispatch; call sites need not change.

Each model also exposes a `couplings` property returning a list of
CouplingTerm objects that describes how each field enters G(x).  This
is used by Phase 3 (visualisation generalisation) to discover field
dependencies without parsing the energy model internals.

Variable pressure
-----------------
All models accept an optional pressure argument P (default 0.0, which
contributes zero to G so existing call sites are unaffected):

  G(x, T, P) = G_base(x, T)
             + P · V(x)                          if V_coeffs is given
             + R_gas · T · ln(P / P_ref)         if ideal_gas=True and P > 0

V(x) = V_coeffs[0] + V_coeffs[1]·x + V_coeffs[2]·x² + ...
R_gas and P_ref are set per-model from the system-level pressure block
by the parser (pde_input.py).

Physical guidance on which terms to use:
  Ideal gas phase:  set ideal_gas=True and omit V_coeffs.
    The R·T·ln(P/P°) term IS the full pressure dependence, derived from
    integrating (dG/dP)_T = V = RT/P.  Adding P·V(x) on top would
    double-count the pressure contribution.
  Condensed phase (liquid/solid):  set ideal_gas=False and supply V_coeffs.
    V is small and nearly P-independent; P·V(x) is the Poynting correction.
  Advanced non-ideal gas:  both terms can be used together if V_coeffs
    represents a correction beyond ideal (e.g. van der Waals co-volume),
    with the understanding that the user is responsible for avoiding
    double-counting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.polynomial import polyval


# ---------------------------------------------------------------------------
# CouplingTerm — describes how one field (or field combination) enters G(x)
# ---------------------------------------------------------------------------

@dataclass
class CouplingTerm:
    """One field-coupling contribution to the Gibbs energy.

    G(x, {λ}) += R(x) · f({λ})

    where R(x) is a composition polynomial (response function) and
    f({λ}) is a function of one or more field values.

    Attributes
    ----------
    response_coeffs : list[float]
        Coefficients of R(x) in ascending order; R(x) = Σᵢ cᵢ·xⁱ.
    coupling_type   : str
        How f({λ}) is computed:
          'linear'    — f = λ  (single field).  Covers −S·T, V·P, −M·H.
          'ideal_gas' — f = T · ln(P / params['P_ref']).
          'power'     — f = λʲ for a single field (T-power in PolyModel).
    field_names     : list[str]
        Names of the Field objects this term uses (e.g. ['temperature']).
    params          : dict
        Extra parameters (e.g. {'P_ref': 1.0} for ideal_gas coupling).
    """
    response_coeffs: list
    coupling_type:   str
    field_names:     list
    params:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EnergyModel(ABC):
    """Abstract base for all Gibbs energy models.

    Subclasses implement _gibbs_impl(x, field_values) where field_values
    is a dict mapping field names to scalar values (e.g.
    {'temperature': 800.0, 'pressure': 1.5}).

    The public gibbs() method is a backward-compatible shim that accepts
    both the old positional signature gibbs(x, T, P=0.0) and the new
    dict-based signature gibbs(x, {'temperature': T, 'pressure': P}).
    """

    @abstractmethod
    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        """Return G(x, {fields}) as a numpy array with the same shape as x."""

    def gibbs(self, x, T=None, P=0.0, field_values=None, **kwargs):
        """Return G evaluated at composition x and the given field values.

        Accepts two calling conventions (both remain supported):

          Old:  model.gibbs(x, T)
                model.gibbs(x, T, P)
          New:  model.gibbs(x, {'temperature': T, 'pressure': P, ...})
                model.gibbs(x, field_values={'temperature': T, ...})

        Parameters
        ----------
        x             : array-like   — composition values
        T             : float or dict — temperature (old style) or
                        field_values dict (new style, positional shortcut)
        P             : float        — pressure (old style only)
        field_values  : dict or None — explicit field dict (new style)
        **kwargs      : extra field values merged into field_values (new style)
        """
        if field_values is not None:
            fv = dict(field_values)
            fv.update(kwargs)
        elif isinstance(T, dict):
            # New style passed positionally: gibbs(x, {'temperature': T, ...})
            fv = dict(T)
            fv.update(kwargs)
        else:
            # Old style: gibbs(x, T) or gibbs(x, T, P)
            fv = {
                'temperature': float(T) if T is not None else 0.0,
                'pressure':    float(P),
            }
            fv.update(kwargs)
        return self._gibbs_impl(np.asarray(x, dtype=float), fv)

    @property
    def couplings(self) -> list:
        """Return the list of CouplingTerm objects describing this model.

        Subclasses override this to provide inspectable coupling structure.
        Default: empty list (no explicit coupling description available).
        """
        return []


# ---------------------------------------------------------------------------
# HSModel
# ---------------------------------------------------------------------------

class HSModel(EnergyModel):
    """G(x, T, P) = H(x) - T·S(x) [+ P·V(x)] [+ R·T·ln(P/P°)]

    H(x) = H_coeffs[0] + H_coeffs[1]·x + H_coeffs[2]·x² + ...
    S(x) = S_coeffs[0] + S_coeffs[1]·x + S_coeffs[2]·x² + ...
    V(x) = V_coeffs[0] + V_coeffs[1]·x + V_coeffs[2]·x² + ...  (optional)

    For an end-member phase at a single composition, pass single-element lists:
      HSModel([H0], [S0])  →  G(T) = H0 - T·S0

    Parameters
    ----------
    H_coeffs  : array-like  — enthalpy polynomial coefficients (ascending x)
    S_coeffs  : array-like  — entropy polynomial coefficients (ascending x)
    V_coeffs  : array-like or None — molar volume polynomial; None → no PV term
    ideal_gas : bool — if True, add R·T·ln(P/P_ref) (ideal gas chemical potential)
    R_gas     : float — gas constant in the user's energy units (e.g. 8.314e-3 kJ/mol/K)
    P_ref     : float — reference pressure in the user's pressure units
    """

    def __init__(self, H_coeffs, S_coeffs,
                 V_coeffs=None, ideal_gas=False,
                 R_gas=0.0, P_ref=1.0):
        self.H_coeffs = np.asarray(H_coeffs, dtype=float)
        self.S_coeffs = np.asarray(S_coeffs, dtype=float)
        self.V_coeffs = np.asarray(V_coeffs, dtype=float) if V_coeffs is not None else None
        self.ideal_gas = bool(ideal_gas)
        self.R_gas = float(R_gas)
        self.P_ref = float(P_ref)
        self.vle_params = None   # set by callers using compute_vle_gas_hs()

    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        T = float(field_values.get('temperature', 0.0))
        P = float(field_values.get('pressure', 0.0))
        G = polyval(x, self.H_coeffs) - T * polyval(x, self.S_coeffs)
        if self.V_coeffs is not None:
            G = G + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            G = G + self.R_gas * T * np.log(P / self.P_ref)
        return G

    @property
    def couplings(self) -> list:
        """CouplingTerm list describing this model's field dependencies."""
        terms = [
            CouplingTerm(
                response_coeffs=(-self.S_coeffs).tolist(),
                coupling_type='linear',
                field_names=['temperature'],
            ),
        ]
        if self.V_coeffs is not None:
            terms.append(CouplingTerm(
                response_coeffs=self.V_coeffs.tolist(),
                coupling_type='linear',
                field_names=['pressure'],
            ))
        if self.ideal_gas:
            terms.append(CouplingTerm(
                response_coeffs=[self.R_gas],
                coupling_type='ideal_gas',
                field_names=['temperature', 'pressure'],
                params={'P_ref': self.P_ref},
            ))
        return terms


# ---------------------------------------------------------------------------
# VLE reparameterisation helper
# ---------------------------------------------------------------------------

def compute_vle_gas_hs(liq_H, liq_S, T_bp_A, T_bp_B, L_A, L_B):
    """Derive (H_gas_coeffs, S_gas_coeffs) satisfying all VLE conditions.

    Given a liquid phase's H and S polynomials and the latent heats L_A, L_B
    at the pure-component boiling points T_bp_A (x=0) and T_bp_B (x=1),
    returns gas H and S coefficients such that all 6 VLE consistency conditions
    are satisfied by construction:
      - G_gas = G_liq and dG_gas/dx = dG_liq/dx  at  x=0  (T = T_bp_A)
      - G_gas = G_liq and dG_gas/dx = dG_liq/dx  at  x=1  (T = T_bp_B)
      - S_gas > S_liq  so that vapour is stable above the boiling point

    The ΔH and ΔS corrections follow from a 4-constraint derivation with
    a quadratic Ansatz ΔH(x) = L_A·(1−x)² + L_B·x²:

        ΔH = [L_A,  −2·L_A,   L_A+L_B]
        ΔS = [L_A/T_A, −2·L_A/T_A, L_A/T_A+L_B/T_B]

    Parameters
    ----------
    liq_H   : sequence  — liquid enthalpy coefficients (ascending x)
    liq_S   : sequence  — liquid entropy coefficients (ascending x)
    T_bp_A  : float     — boiling point of pure A (x=0), K
    T_bp_B  : float     — boiling point of pure B (x=1), K
    L_A     : float     — latent heat of A (same units as H)
    L_B     : float     — latent heat of B (same units as H)

    Returns
    -------
    (H_gas_coeffs, S_gas_coeffs) : (list[float], list[float])
        Length-3 lists ready to pass to HSModel.
    """
    # Pad liquid coefficients to at least 3 terms.
    H = list(liq_H) + [0.0] * max(0, 3 - len(liq_H))
    S = list(liq_S) + [0.0] * max(0, 3 - len(liq_S))
    dH = [L_A, -2.0 * L_A, L_A + L_B]
    dS = [L_A / T_bp_A, -2.0 * L_A / T_bp_A, L_A / T_bp_A + L_B / T_bp_B]
    H_gas = [H[i] + dH[i] for i in range(3)]
    S_gas = [S[i] + dS[i] for i in range(3)]
    return H_gas, S_gas


# ---------------------------------------------------------------------------
# PolyModel
# ---------------------------------------------------------------------------

class PolyModel(EnergyModel):
    """G(x, T, P) = Σᵢ cᵢ(T)·xⁱ [+ P·V(x)] [+ R·T·ln(P/P°)]

    t_poly_coeffs is a list indexed by x-power i. Each element is an array
    of T-polynomial coefficients in ascending order:
      t_poly_coeffs[i][j]  is the coefficient for  xⁱ·Tʲ

    For an end-member phase, pass a single-element outer list:
      PolyModel([[a00, a01, a02, ...]])  →  G(T) = a00 + a01·T + a02·T² + ...

    Parameters
    ----------
    t_poly_coeffs : list of array-like — T-polynomials for each x power
    V_coeffs      : array-like or None — molar volume polynomial; None → no PV term
    ideal_gas     : bool — if True, add R·T·ln(P/P_ref)
    R_gas         : float — gas constant in the user's energy units
    P_ref         : float — reference pressure in the user's pressure units
    """

    def __init__(self, t_poly_coeffs,
                 V_coeffs=None, ideal_gas=False,
                 R_gas=0.0, P_ref=1.0):
        self.coeffs = [np.asarray(tc, dtype=float) for tc in t_poly_coeffs]
        self.V_coeffs = np.asarray(V_coeffs, dtype=float) if V_coeffs is not None else None
        self.ideal_gas = bool(ideal_gas)
        self.R_gas = float(R_gas)
        self.P_ref = float(P_ref)

    def _gibbs_impl(self, x, field_values: dict) -> np.ndarray:
        T = float(field_values.get('temperature', 0.0))
        P = float(field_values.get('pressure', 0.0))
        result = np.zeros_like(x)
        for i, t_coeffs in enumerate(self.coeffs):
            c_i = polyval(T, t_coeffs)   # scalar: T-polynomial evaluated at T
            result = result + c_i * x**i
        if self.V_coeffs is not None:
            result = result + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            result = result + self.R_gas * T * np.log(P / self.P_ref)
        return result

    @property
    def couplings(self) -> list:
        """CouplingTerm list describing this model's field dependencies."""
        # Represent the full T-polynomial structure as a single 'poly_T' term.
        # The coefficient grid is stored in params for Phase 3 inspection.
        terms = [
            CouplingTerm(
                response_coeffs=[1.0],   # placeholder; full structure in params
                coupling_type='poly_T',
                field_names=['temperature'],
                params={'t_poly_coeffs': [c.tolist() for c in self.coeffs]},
            ),
        ]
        if self.V_coeffs is not None:
            terms.append(CouplingTerm(
                response_coeffs=self.V_coeffs.tolist(),
                coupling_type='linear',
                field_names=['pressure'],
            ))
        if self.ideal_gas:
            terms.append(CouplingTerm(
                response_coeffs=[self.R_gas],
                coupling_type='ideal_gas',
                field_names=['temperature', 'pressure'],
                params={'P_ref': self.P_ref},
            ))
        return terms
