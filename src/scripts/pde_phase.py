#!/usr/bin/env python3
"""
Phase and System data structures for PDE.

Phase  — one thermodynamic phase: name, type, energy model, composition range.
System — the full system: components, all phases, energy form, T range,
         and optional pressure range.
"""

import numpy as np


class Phase:
    """One thermodynamic phase.

    Attributes
    ----------
    name         : str   — user-given label (e.g. 'liquid', 'alpha')
    phase_type   : str   — 'gas', 'liquid', 'solid', or 'end_member'
    energy_model : EnergyModel — instance from pde_energy
    xmin, xmax   : float — valid composition range [0, 1]
                           For end-members xmin == xmax (degenerate point).
    """

    def __init__(self, name, phase_type, energy_model, xmin=0.0, xmax=1.0):
        self.name = name
        self.phase_type = phase_type
        self.energy_model = energy_model
        self.xmin = xmin
        self.xmax = xmax

    @property
    def is_point(self):
        """True when this phase exists at a single composition (end-member)."""
        return self.xmin == self.xmax

    def gibbs(self, x, T, P=0.0):
        """Evaluate G(x, T, P) via the phase's energy model."""
        return self.energy_model.gibbs(x, T, P)

    def composition_grid(self, n_points=500):
        """Return a linearly spaced composition array over [xmin, xmax]."""
        if self.is_point:
            return np.array([self.xmin])
        return np.linspace(self.xmin, self.xmax, n_points)


class System:
    """The full thermodynamic system.

    Attributes
    ----------
    components   : list[str]   — component names in composition order
    phases       : list[Phase] — all phases (gas, liquid, solid, end-members)
    energy_form  : str         — 'HS' or 'polynomial'
    T_min        : float       — lower bound of temperature range
    T_max        : float       — upper bound of temperature range
    T_initial    : float       — starting temperature for the visualization
    has_pressure : bool        — True when a <pressure> block was present in XML
    P_min        : float       — lower bound of pressure range
    P_max        : float       — upper bound of pressure range
    P_initial    : float       — starting pressure for the visualization
    R_gas        : float       — gas constant in user energy units (0 → inactive)
    P_ref        : float       — reference pressure for ideal-gas term
    P_unit       : str         — pressure unit label (e.g. 'atm'); '' if unspecified
    title        : str         — display title; '' if unset (viz falls back to filename)
    """

    def __init__(self, components, phases, energy_form, T_min, T_max, T_initial,
                 has_pressure=False,
                 P_min=1.0, P_max=1.0, P_initial=1.0,
                 R_gas=0.0, P_ref=1.0, P_unit='', title=''):
        self.components = components
        self.phases = phases
        self.energy_form = energy_form
        self.T_min = T_min
        self.T_max = T_max
        self.T_initial = T_initial
        self.has_pressure = has_pressure
        self.P_min = P_min
        self.P_max = P_max
        self.P_initial = P_initial
        self.R_gas = R_gas
        self.P_ref = P_ref
        self.P_unit = P_unit
        self.title = title

    @property
    def n_components(self):
        return len(self.components)

    @property
    def gas_phases(self):
        return [p for p in self.phases if p.phase_type == 'gas']

    @property
    def liquid_phases(self):
        return [p for p in self.phases if p.phase_type == 'liquid']

    @property
    def solid_phases(self):
        return [p for p in self.phases if p.phase_type == 'solid']

    @property
    def end_members(self):
        return [p for p in self.phases if p.phase_type == 'end_member']
