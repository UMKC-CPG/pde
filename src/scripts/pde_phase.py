#!/usr/bin/env python3
"""
Phase and System data structures for PDE.

Phase  — one thermodynamic phase: name, type, energy model, composition range.
Field  — one sweepable intensive parameter (T, P, H, E, …).
System — the full system: components, all phases, energy form, and a list of
         Field objects.  fields[0] is temperature by convention.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Field:
    """One sweepable intensive thermodynamic parameter (or analogue).

    Role (primary sweep axis, secondary axis, fixed) is NOT stored here —
    it belongs to the view configuration, not to the field itself.

    Attributes
    ----------
    name        : str   — unique identifier, e.g. 'temperature', 'pressure'
    symbol      : str   — display symbol, e.g. 'T', 'P', 'H'
    unit        : str   — display unit string, e.g. 'K', 'atm', 'T'; '' if none
    min_val     : float — lower bound of the sweep range
    max_val     : float — upper bound of the sweep range
    initial_val : float — starting value for the visualisation
    """
    name:        str
    symbol:      str
    unit:        str
    min_val:     float
    max_val:     float
    initial_val: float


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
    components  : list[str]    — component names in composition order
    phases      : list[Phase]  — all phases (gas, liquid, solid, end-members)
    energy_form : str          — 'HS' or 'polynomial'
    fields      : list[Field]  — sweepable intensive parameters; fields[0] is
                                 temperature by convention; any number of
                                 additional fields are allowed
    title       : str          — display title; '' if unset

    R_gas, P_ref are no longer stored on System; they live on each phase's
    EnergyModel.  The backward-compat properties below scan self.phases to
    retrieve them for code that has not yet migrated to reading from the
    energy model directly.  Both properties will be removed in Phase 3.
    """

    def __init__(self, components, phases, energy_form, fields, title=''):
        self.components = components
        self.phases = phases
        self.energy_form = energy_form
        self.fields = list(fields)   # list[Field]; fields[0] is temperature
        self.title = title

    # ------------------------------------------------------------------
    # Convenience accessors — backward-compatible properties
    # ------------------------------------------------------------------

    @property
    def T_field(self) -> Field:
        return self.fields[0]

    @property
    def T_min(self) -> float:
        return self.fields[0].min_val

    @property
    def T_max(self) -> float:
        return self.fields[0].max_val

    @property
    def T_initial(self) -> float:
        return self.fields[0].initial_val

    @property
    def P_field(self):
        """The pressure Field, or None if no pressure field is defined."""
        return next((f for f in self.fields if f.name == 'pressure'), None)

    @property
    def has_pressure(self) -> bool:
        return self.P_field is not None

    @property
    def P_min(self) -> float:
        f = self.P_field
        return f.min_val if f is not None else 1.0

    @property
    def P_max(self) -> float:
        f = self.P_field
        return f.max_val if f is not None else 1.0

    @property
    def P_initial(self) -> float:
        f = self.P_field
        return f.initial_val if f is not None else 1.0

    @property
    def P_unit(self) -> str:
        f = self.P_field
        return f.unit if f is not None else ''

    @property
    def R_gas(self) -> float:
        """Gas constant from the first ideal-gas phase's energy model.

        Returns 0.0 when no ideal-gas phases are present.
        Backward-compatible shim; will be removed in Phase 3.
        """
        for p in self.phases:
            m = p.energy_model
            if getattr(m, 'ideal_gas', False) and getattr(m, 'R_gas', 0.0) > 0.0:
                return m.R_gas
        return 0.0

    @property
    def P_ref(self) -> float:
        """Reference pressure from the first ideal-gas phase's energy model.

        Returns 1.0 when no ideal-gas phases are present.
        Backward-compatible shim; will be removed in Phase 3.
        """
        for p in self.phases:
            m = p.energy_model
            if getattr(m, 'ideal_gas', False) and getattr(m, 'R_gas', 0.0) > 0.0:
                return m.P_ref
        return 1.0

    # ------------------------------------------------------------------
    # Phase-type filters
    # ------------------------------------------------------------------

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
