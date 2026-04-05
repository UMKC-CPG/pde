#!/usr/bin/env python3
"""
Phase and System data structures for PDE.

Spec layer (mutable, serialisable, UI-friendly):
  FieldSpec  — one sweepable field before construction.
  PhaseSpec  — one phase before construction; model-
               specific data lives in model_params dict.
  SystemSpec — the full system spec; single canonical
               input form for all construction paths.

Runtime layer (derived, computation-optimised):
  Field   — one sweepable intensive parameter.
  Phase   — one thermodynamic phase with energy model.
  System  — the full system consumed by compute/viz.

Construction flow:
  raw input (XML, builder, direct)
    → SystemSpec + list[PhaseSpec]
      → SystemSpec.to_system()  (only place models
                                  are built)
        → System                (derived; never
                                  round-tripped)
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------
# Spec layer — mutable canonical form (DESIGN §11.2,
#              PSEUDOCODE §7)
# ---------------------------------------------------------------

@dataclass
class FieldSpec:
    """Spec for one sweepable intensive parameter.

    Mirrors Field one-to-one, plus an extras dict for
    field-level constants that energy models need at
    build time (e.g. R_gas, P_ref on the pressure
    field).

    Attributes
    ----------
    name        : str   — unique id, e.g. 'temperature'
    symbol      : str   — display symbol, e.g. 'T'
    unit        : str   — display unit, e.g. 'K'
    min_val     : float — lower bound of sweep range
    max_val     : float — upper bound of sweep range
    initial_val : float — starting value for the UI
    extras      : dict  — field-level constants; keys
                          vary by field name
    """
    name:        str
    symbol:      str
    unit:        str
    min_val:     float
    max_val:     float
    initial_val: float
    extras:      dict = field(default_factory=dict)


@dataclass
class PhaseSpec:
    """Spec for one thermodynamic phase.

    All model-specific data lives in model_params,
    keyed by model_type.  Adding a new energy model
    requires one new model_type string and new keys
    in model_params — PhaseSpec itself does not change.

    model_params keys by model_type
    --------------------------------
    'HS':
        H_coeffs   : list[float]  — ascending order
        S_coeffs   : list[float]  — ascending order
        V_coeffs   : list[float] | None
        ideal_gas   : bool
        vle_params  : dict | None
            liquid_phase : str
            T_bp_A, T_bp_B : float
            L_A, L_B       : float

    'polynomial':
        poly_coeffs : list[list[float]]
        V_coeffs    : list[float] | None
        ideal_gas   : bool

    'piecewise_patch':
        H_coeffs          : list[float]
        S_coeffs          : list[float]
        patch_left_phase  : str | None
        patch_left_x      : float | None
        patch_right_phase : str | None
        patch_right_x     : float | None

    Cross-phase references (liquid_phase,
    patch_left_phase, etc.) store phase *names*,
    resolved at build time by topological sort.

    Attributes
    ----------
    name        : str   — user label (e.g. 'liquid')
    phase_type  : str   — 'gas'|'liquid'|'solid'|
                          'end_member'
    xmin        : float — lower composition bound
    xmax        : float — upper composition bound
    model_type  : str   — 'HS'|'polynomial'|
                          'piecewise_patch'
    model_params: dict  — model-specific parameters
    """
    name:         str
    phase_type:   str
    xmin:         float = 0.0
    xmax:         float = 1.0
    model_type:   str   = 'HS'
    model_params: dict  = field(default_factory=dict)

    def make_energy_model(self, specs_by_name,
                          built, field_specs):
        """Build the EnergyModel for this PhaseSpec.

        Dispatches on model_type, constructing the appropriate EnergyModel
        subclass.  Resolves cross-phase references (VLE liquid, patch
        targets) via specs_by_name for coefficient data and built for
        models already constructed during the topological sort.

        Parameters
        ----------
        specs_by_name : dict[str, PhaseSpec]
            Every phase spec in the system, keyed by name.  Supplies
            coefficient data from dependency phases (e.g. the liquid
            phase providing H/S for VLE gas derivation).
        built : dict[str, EnergyModel]
            Models already built during the topological walk.  Used by
            piecewise-patch phases to obtain the target phase's H and S
            coefficient arrays for patch slope matching.
        field_specs : list[FieldSpec]
            System field specs.  Pressure-field extras supply R_gas and
            P_ref; the temperature field's initial_val gives the reference
            temperature for patch-H computation.

        Returns
        -------
        EnergyModel

        Raises
        ------
        ValueError
            If model_type is not 'HS', 'polynomial', or 'piecewise_patch'.
        """
        # Local import avoids a module-level spec-to-energy dependency.
        from pde_energy import (
            HSModel, PolyModel,
            PiecewisePatchModel,
            compute_vle_gas_hs,
            compute_left_patch_H,
            compute_right_patch_H,
        )

        # -- Field-level constants -----------------------------------------
        # R_gas and P_ref live in the pressure field's extras dict; T_ref
        # is the temperature field's initial_val (reference temperature
        # for patch-H slope matching).
        R_gas, P_ref, T_ref = 0.0, 1.0, 0.0
        for fspec in field_specs:
            if fspec.name == 'pressure':
                R_gas = fspec.extras.get(
                    'R_gas', 0.0)
                P_ref = fspec.extras.get(
                    'P_ref', 1.0)
            elif fspec.name == 'temperature':
                T_ref = fspec.initial_val

        mp = self.model_params

        # -- HS model (with optional VLE) ----------------------------------
        if self.model_type == 'HS':
            vle = mp.get('vle_params')
            if vle is not None:
                # Derive gas H and S from the liquid-phase coefficients
                # using VLE boiling-point and latent-heat data.
                liq_name = vle['liquid_phase']
                liq_mp = specs_by_name[
                    liq_name].model_params
                H_gas, S_gas = compute_vle_gas_hs(
                    liq_mp['H_coeffs'],
                    liq_mp['S_coeffs'],
                    vle['T_bp_A'], vle['T_bp_B'],
                    vle['L_A'], vle['L_B'],
                )
                model = HSModel(
                    H_gas, S_gas,
                    V_coeffs=mp.get('V_coeffs'),
                    ideal_gas=mp.get(
                        'ideal_gas', False),
                    R_gas=R_gas, P_ref=P_ref,
                )
                # Preserve boiling-point metadata for serialisation
                # round-trips.
                model.vle_params = {
                    'T_bp_A': vle['T_bp_A'],
                    'T_bp_B': vle['T_bp_B'],
                    'L_A':    vle['L_A'],
                    'L_B':    vle['L_B'],
                }
                return model

            # Plain HS — coefficients used directly.
            return HSModel(
                mp['H_coeffs'], mp['S_coeffs'],
                V_coeffs=mp.get('V_coeffs'),
                ideal_gas=mp.get(
                    'ideal_gas', False),
                R_gas=R_gas, P_ref=P_ref,
            )

        # -- Polynomial model ----------------------------------------------
        if self.model_type == 'polynomial':
            return PolyModel(
                mp['poly_coeffs'],
                V_coeffs=mp.get('V_coeffs'),
                ideal_gas=mp.get(
                    'ideal_gas', False),
                R_gas=R_gas, P_ref=P_ref,
            )

        # -- Piecewise-patch model -----------------------------------------
        if self.model_type == 'piecewise_patch':
            H_orig = mp['H_coeffs']
            S_coeffs = mp['S_coeffs']
            x_cut_left = mp.get('patch_left_x')
            x_cut_right = mp.get('patch_right_x')
            left_phase = mp.get(
                'patch_left_phase')
            right_phase = mp.get(
                'patch_right_phase')
            H_left, H_right = None, None

            # Left patch: match G and dG/dx of the target phase at
            # x_cut_left.
            if (x_cut_left is not None
                    and left_phase):
                tgt = built.get(left_phase)
                if (tgt is not None
                        and isinstance(
                            tgt, HSModel)):
                    H_left = compute_left_patch_H(
                        H_orig, S_coeffs,
                        tgt.H_coeffs.tolist(),
                        tgt.S_coeffs.tolist(),
                        self.xmin, x_cut_left,
                        T_ref)

            # Right patch: symmetric construction, matching the target
            # at x_cut_right.
            if (x_cut_right is not None
                    and right_phase):
                tgt = built.get(right_phase)
                if (tgt is not None
                        and isinstance(
                            tgt, HSModel)):
                    H_right = compute_right_patch_H(
                        H_orig, S_coeffs,
                        tgt.H_coeffs.tolist(),
                        tgt.S_coeffs.tolist(),
                        self.xmax, x_cut_right,
                        T_ref)

            patch = PiecewisePatchModel(
                H_orig, S_coeffs,
                H_left=H_left,
                x_cut_left=x_cut_left,
                H_right=H_right,
                x_cut_right=x_cut_right,
                V_coeffs=mp.get('V_coeffs'),
                ideal_gas=mp.get(
                    'ideal_gas', False),
                R_gas=R_gas, P_ref=P_ref,
            )
            patch.patch_left_phase_name = (
                left_phase or '')
            patch.patch_right_phase_name = (
                right_phase or '')
            return patch

        raise ValueError(
            f"Unknown model_type: "
            f"{self.model_type!r}")


# ---------------------------------------------------------------
# Topological helpers for SystemSpec.to_system()
# ---------------------------------------------------------------

def _dependency_edges(phase_specs):
    """Extract cross-phase dependency edges.

    Returns a list of (dependency_name, dependent_name) tuples derived
    from model_params keys that reference other phases by name.  Only
    truthy values produce edges (None and '' are skipped).
    """
    edges = []
    for ps in phase_specs:
        mp = ps.model_params
        vle = mp.get('vle_params')
        if vle is not None:
            liq = vle.get('liquid_phase')
            if liq:
                edges.append((liq, ps.name))
        left = mp.get('patch_left_phase')
        if left:
            edges.append((left, ps.name))
        right = mp.get('patch_right_phase')
        if right:
            edges.append((right, ps.name))
    return edges


def _topo_sort(phase_specs, edges):
    """Return phase specs in dependency-safe build order (Kahn's algorithm).

    Raises ValueError when a circular dependency is detected among phases.
    """
    names = [ps.name for ps in phase_specs]
    spec_map = {
        ps.name: ps for ps in phase_specs}
    adjacency = {n: [] for n in names}
    in_degree = {n: 0 for n in names}

    for dep_name, dependent_name in edges:
        if dep_name in adjacency:
            adjacency[dep_name].append(
                dependent_name)
            in_degree[dependent_name] += 1

    queue = deque(
        n for n in names
        if in_degree[n] == 0)
    ordered = []

    while queue:
        node = queue.popleft()
        ordered.append(spec_map[node])
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(ordered) != len(names):
        stuck = [
            n for n in names
            if in_degree[n] > 0]
        raise ValueError(
            "Circular phase dependency among: "
            + ", ".join(stuck))

    return ordered


@dataclass
class SystemSpec:
    """Spec for a complete thermodynamic system.

    The single authoritative input form.  All
    construction paths (XML parse, builder apply,
    direct construction) produce a SystemSpec.
    to_system() is the only place EnergyModel
    instances are built.

    Attributes
    ----------
    title       : str        — display title
    components  : list[str]  — component names
    energy_form : str        — 'HS' or 'polynomial'
    fields      : list[FieldSpec] — sweepable fields
    phases      : list[PhaseSpec] — phase specs
    """
    title:       str
    components:  list
    energy_form: str
    fields:      list = field(default_factory=list)
    phases:      list = field(default_factory=list)

    def to_system(self):
        """Build a System from this SystemSpec.

        Topologically sorts phases by cross-phase dependencies, builds
        each phase's EnergyModel via make_energy_model(), then assembles
        the runtime System.  Phase ordering in the result preserves the
        spec's original declaration order (not the build order), keeping
        UI legend and colour assignment stable.
        """
        edges = _dependency_edges(self.phases)
        order = _topo_sort(self.phases, edges)

        specs_by_name = {
            ps.name: ps for ps in self.phases}
        built = {}

        for ps in order:
            model = ps.make_energy_model(
                specs_by_name, built,
                self.fields)
            built[ps.name] = model

        # Preserve declaration order for UI/legend
        # stability (not the topo build order).
        phases = [
            Phase(ps.name, ps.phase_type,
                  built[ps.name],
                  ps.xmin, ps.xmax)
            for ps in self.phases
        ]

        fields = [
            Field(fs.name, fs.symbol, fs.unit,
                  fs.min_val, fs.max_val,
                  fs.initial_val)
            for fs in self.fields
        ]

        return System(
            self.components, phases,
            self.energy_form, fields,
            self.title)


# ---------------------------------------------------------------
# Runtime layer — derived from Spec via to_system()
# ---------------------------------------------------------------

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
