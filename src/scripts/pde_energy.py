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

Variable pressure
-----------------
All models accept an optional pressure argument P (default 0.0, which
contributes zero to G so existing call sites are unaffected):

  G(x, T, P) = G_base(x, T)
             + P · V(x)                          if V_coeffs is given
             + R_gas · T · ln(P / P_ref)         if ideal_gas=True and P > 0

V(x) = V_coeffs[0] + V_coeffs[1]·x + V_coeffs[2]·x² + ...
R_gas and P_ref are set from the system-level <pressure> block by the parser.

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

import numpy as np
from numpy.polynomial.polynomial import polyval


class EnergyModel(ABC):
    """Abstract base for all Gibbs energy models.

    All subclasses must implement gibbs(x, T, P=0.0) where x is a numpy
    array of compositions in [0, 1], T is a scalar temperature in kelvin
    (or whatever units the user has chosen), and P is an optional scalar
    pressure. The return value is a numpy array of the same shape as x.
    """

    @abstractmethod
    def gibbs(self, x, T, P=0.0):
        """Return G(x, T, P) as a numpy array with the same shape as x."""


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

    def gibbs(self, x, T, P=0.0):
        x = np.asarray(x, dtype=float)
        T = float(T)
        P = float(P)
        G = polyval(x, self.H_coeffs) - T * polyval(x, self.S_coeffs)
        if self.V_coeffs is not None:
            G = G + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            G = G + self.R_gas * T * np.log(P / self.P_ref)
        return G


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

    def gibbs(self, x, T, P=0.0):
        x = np.asarray(x, dtype=float)
        T = float(T)
        P = float(P)
        result = np.zeros_like(x)
        for i, t_coeffs in enumerate(self.coeffs):
            c_i = polyval(T, t_coeffs)   # scalar: T-polynomial evaluated at T
            result = result + c_i * x**i
        if self.V_coeffs is not None:
            result = result + P * polyval(x, self.V_coeffs)
        if self.ideal_gas and P > 0.0 and self.R_gas > 0.0:
            result = result + self.R_gas * T * np.log(P / self.P_ref)
        return result
