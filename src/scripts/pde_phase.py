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

    'calphad':
        calphad_phase : str
            Phase name in the TDB file (e.g.
            'LIQUID', 'FCC_A1', 'HCP_A3').
        components    : list[str]
            Element names in PDE composition
            order (e.g. ['AL', 'MG']).  Set
            automatically by the parser from
            the system-level <components>.

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
                          'piecewise_patch'|
                          'calphad'
    model_params: dict  — model-specific parameters
    """
    name:         str
    phase_type:   str
    xmin:         float = 0.0
    xmax:         float = 1.0
    model_type:   str   = 'HS'
    model_params: dict  = field(default_factory=dict)

    # ---------------------------------------------------------
    # Convenience properties — permanent read/write API for
    # the model_params dict.  Used by the builder UI, the
    # fitting functions (apply_handle_drag, etc.), and the
    # G(x) canvas edit overlay.
    # ---------------------------------------------------------

    @property
    def H_coeffs(self):
        """H(x) polynomial coefficients (HS/patch)."""
        return self.model_params.get(
            'H_coeffs', [0.0])

    @H_coeffs.setter
    def H_coeffs(self, value):
        self.model_params['H_coeffs'] = value

    @property
    def S_coeffs(self):
        """S(x) polynomial coefficients (HS/patch)."""
        return self.model_params.get(
            'S_coeffs', [0.0])

    @S_coeffs.setter
    def S_coeffs(self, value):
        self.model_params['S_coeffs'] = value

    @property
    def V_coeffs(self):
        """V(x) molar-volume coefficients, or None."""
        return self.model_params.get('V_coeffs')

    @V_coeffs.setter
    def V_coeffs(self, value):
        self.model_params['V_coeffs'] = value

    @property
    def ideal_gas(self):
        """True when ideal-gas R*T*ln(P/P0) applies."""
        return self.model_params.get(
            'ideal_gas', False)

    @ideal_gas.setter
    def ideal_gas(self, value):
        self.model_params['ideal_gas'] = value

    @property
    def poly_coeffs(self):
        """Polynomial table [x_power][T_power]."""
        return self.model_params.get(
            'poly_coeffs', [[0.0]])

    @poly_coeffs.setter
    def poly_coeffs(self, value):
        self.model_params['poly_coeffs'] = value

    @property
    def vle_params(self):
        """VLE reparametrisation dict, or None."""
        return self.model_params.get('vle_params')

    @vle_params.setter
    def vle_params(self, value):
        if value is None:
            self.model_params.pop(
                'vle_params', None)
        else:
            self.model_params['vle_params'] = value

    @property
    def patch_left_x(self):
        """Left-patch cut-off composition, or None."""
        return self.model_params.get(
            'patch_left_x')

    @patch_left_x.setter
    def patch_left_x(self, value):
        self.model_params['patch_left_x'] = value

    @property
    def patch_right_x(self):
        """Right-patch cut-off composition, or None."""
        return self.model_params.get(
            'patch_right_x')

    @patch_right_x.setter
    def patch_right_x(self, value):
        self.model_params['patch_right_x'] = value

    @property
    def patch_left_phase(self):
        """Left-patch target phase name, or ''."""
        return self.model_params.get(
            'patch_left_phase', '')

    @patch_left_phase.setter
    def patch_left_phase(self, value):
        self.model_params['patch_left_phase'] = value

    @property
    def patch_right_phase(self):
        """Right-patch target phase name, or ''."""
        return self.model_params.get(
            'patch_right_phase', '')

    @patch_right_phase.setter
    def patch_right_phase(self, value):
        self.model_params[
            'patch_right_phase'] = value

    @property
    def calphad_phase(self):
        """TDB phase name (CALPHAD only)."""
        return self.model_params.get(
            'calphad_phase', '')

    @calphad_phase.setter
    def calphad_phase(self, value):
        self.model_params[
            'calphad_phase'] = value

    @property
    def is_vle_gas(self):
        """True when this is a VLE-derived gas phase."""
        return (self.phase_type == 'gas'
                and self.vle_params is not None)

    def make_energy_model(self, specs_by_name,
                          built, field_specs,
                          tdb_db=None):
        """Build the EnergyModel for this PhaseSpec.

        Dispatches on model_type, constructing the
        appropriate EnergyModel subclass.  Resolves
        cross-phase references (VLE liquid, patch
        targets) via specs_by_name for coefficient
        data and built for models already constructed
        during the topological sort.

        Parameters
        ----------
        specs_by_name : dict[str, PhaseSpec]
            Every phase spec in the system, keyed by
            name.  Supplies coefficient data from
            dependency phases (e.g. the liquid phase
            providing H/S for VLE gas derivation).
        built : dict[str, EnergyModel]
            Models already built during the topo walk.
            Used by piecewise-patch phases to obtain
            the target phase's H and S coefficient
            arrays for patch slope matching.
        field_specs : list[FieldSpec]
            System field specs.  Pressure-field extras
            supply R_gas and P_ref; the temperature
            field's initial_val gives the reference
            temperature for patch-H computation.
        tdb_db : pycalphad.Database or None
            Pre-loaded TDB database for CALPHAD
            systems.  Shared across all phases in
            the system; loaded once by
            SystemSpec.to_system().  None for
            non-CALPHAD systems.

        Returns
        -------
        EnergyModel

        Raises
        ------
        ValueError
            If model_type is unrecognised.
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

        # -- CALPHAD model (pycalphad TDB) --------
        if self.model_type == 'calphad':
            from pde_energy import CALPHADModel
            if tdb_db is None:
                raise ValueError(
                    "CALPHAD model requires a "
                    "loaded TDB database "
                    "(tdb_db is None)")
            return CALPHADModel(
                tdb_db,
                mp['calphad_phase'],
                mp['components'])

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
    energy_form : str        — 'HS'|'polynomial'|
                               'calphad'
    fields      : list[FieldSpec] — sweepable fields
    phases      : list[PhaseSpec] — phase specs
    tdb_path    : str        — path to TDB file
                               (CALPHAD only; '' when
                               not applicable).  Stored
                               as written in the XML
                               (usually relative).
    base_dir    : str        — directory used to resolve
                               relative tdb_path at
                               runtime.  Set by the parser
                               to the XML file's parent;
                               defaults to cwd.
    units       : dict       — declared unit system,
                               e.g. {'energy': 'kJ/mol',
                               'temperature': 'K',
                               'pressure': 'atm'}.
                               Human-readable record;
                               used for R_gas validation
                               at parse time.  Empty
                               dict means unspecified.
    """
    title:       str
    components:  list
    energy_form: str
    fields:      list = field(default_factory=list)
    phases:      list = field(default_factory=list)
    tdb_path:    str  = ''
    base_dir:    str  = ''
    units:       dict = field(default_factory=dict)

    def to_system(self):
        """Build a System from this SystemSpec.

        Topologically sorts phases by cross-phase
        dependencies, builds each phase's EnergyModel
        via make_energy_model(), then assembles the
        runtime System.  Phase ordering in the result
        preserves the spec's original declaration
        order (not the build order), keeping UI legend
        and colour assignment stable.

        For CALPHAD systems the TDB file is loaded
        once here and the resulting Database object is
        shared across all phase model constructors.
        """
        # Load TDB database once for CALPHAD
        # systems.  Resolve relative tdb_path against
        # base_dir (set by the parser to the XML's
        # parent directory, or cwd if unset).
        tdb_db = None
        if (self.energy_form == 'calphad'
                and self.tdb_path):
            try:
                import pycalphad as _pc
            except ImportError as exc:
                raise ImportError(
                    "pycalphad is required for "
                    "CALPHAD energy models.  "
                    "Install it with:  pip "
                    "install pycalphad") from exc
            import pathlib
            tdb = pathlib.Path(self.tdb_path)
            if not tdb.is_absolute():
                base = pathlib.Path(
                    self.base_dir or '.')
                tdb = (base / tdb).resolve()
            tdb_db = _pc.Database(str(tdb))

        edges = _dependency_edges(self.phases)
        order = _topo_sort(self.phases, edges)

        specs_by_name = {
            ps.name: ps for ps in self.phases}
        built = {}

        for ps in order:
            model = ps.make_energy_model(
                specs_by_name, built,
                self.fields, tdb_db=tdb_db)
            built[ps.name] = model

        # Preserve declaration order for UI/legend
        # stability (not the topo build order).
        n_comp = len(self.components)
        phases = [
            Phase(ps.name, ps.phase_type,
                  built[ps.name],
                  ps.xmin, ps.xmax,
                  n_components=n_comp)
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

    # ---------------------------------------------------------
    # XML serialisation
    # ---------------------------------------------------------

    def to_xml_str(self):
        """Serialise this SystemSpec to a pretty-printed
        XML string parseable by parse_system_spec().

        Uses the <fields> schema (not the legacy
        separate-element schema).
        """
        from lxml import etree

        root = etree.Element('pde')
        if self.title:
            etree.SubElement(
                root, 'title').text = self.title

        sys_el = etree.SubElement(root, 'system')
        etree.SubElement(
            sys_el, 'components'
        ).text = ' '.join(self.components)
        etree.SubElement(
            sys_el, 'energy_form'
        ).text = self.energy_form
        if self.tdb_path:
            etree.SubElement(
                sys_el, 'tdb'
            ).text = self.tdb_path
        if self.units:
            units_el = etree.SubElement(
                sys_el, 'units')
            for key in sorted(self.units):
                units_el.set(
                    key, self.units[key])

        # -- Fields ---------------------------------
        fields_el = etree.SubElement(root, 'fields')
        for fs in self.fields:
            f_el = etree.SubElement(
                fields_el, 'field')
            f_el.set('name', fs.name)
            f_el.set('symbol', fs.symbol)
            if fs.unit:
                f_el.set('unit', fs.unit)
            f_el.set('min', _fmt(fs.min_val))
            f_el.set('max', _fmt(fs.max_val))
            f_el.set(
                'initial', _fmt(fs.initial_val))
            for k, v in fs.extras.items():
                f_el.set(k, _fmt(v))

        # -- Phases -------------------------------------------
        for ps in self.phases:
            self._emit_phase_xml(root, ps)

        return etree.tostring(
            root, pretty_print=True,
            encoding='unicode')

    @staticmethod
    def _emit_phase_xml(root, ps):
        """Append a <phase> element for *ps* to *root*.

        Handles all three model types (HS, polynomial,
        piecewise_patch) and VLE gas phases.
        """
        from lxml import etree

        phase_el = etree.SubElement(root, 'phase')
        phase_el.set('name', ps.name)
        phase_el.set('type', ps.phase_type)
        if ps.ideal_gas:
            phase_el.set('ideal_gas', 'true')

        is_point = (ps.xmin == ps.xmax)
        if ps.xmin != 0.0 or ps.xmax != 1.0:
            cr = etree.SubElement(
                phase_el, 'composition_range')
            cr.set('xmin', _fmt(ps.xmin))
            cr.set('xmax', _fmt(ps.xmax))

        # VLE gas: emit <vle> instead of <energy>.
        vle = ps.vle_params
        if vle is not None:
            vle_el = etree.SubElement(
                phase_el, 'vle')
            for key in ('T_bp_A', 'T_bp_B',
                        'L_A', 'L_B'):
                if key in vle:
                    vle_el.set(key, _fmt(vle[key]))
            return

        # CALPHAD: <energy model="calphad" phase="…"/>
        if ps.model_type == 'calphad':
            energy_el = etree.SubElement(
                phase_el, 'energy')
            energy_el.set('model', 'calphad')
            energy_el.set(
                'phase',
                ps.model_params.get(
                    'calphad_phase', ''))
            return

        energy_el = etree.SubElement(
            phase_el, 'energy')
        energy_el.set(
            'model',
            'point' if is_point else 'quadratic')

        mt = ps.model_type
        if mt in ('HS', 'piecewise_patch'):
            H = ps.H_coeffs
            S = ps.S_coeffs
            if is_point:
                etree.SubElement(
                    energy_el, 'H'
                ).text = _fmt(H[0] if H else 0.0)
                etree.SubElement(
                    energy_el, 'S'
                ).text = _fmt(S[0] if S else 0.0)
            else:
                H_el = etree.SubElement(
                    energy_el, 'H')
                for i, c in enumerate(H):
                    H_el.set(f'x{i}', _fmt(c))
                S_el = etree.SubElement(
                    energy_el, 'S')
                for i, c in enumerate(S):
                    S_el.set(f'x{i}', _fmt(c))
            V = ps.V_coeffs
            if V:
                V_el = etree.SubElement(
                    energy_el, 'V')
                for i, c in enumerate(V):
                    V_el.set(f'x{i}', _fmt(c))

        elif mt == 'polynomial':
            poly = ps.poly_coeffs
            for i, t_row in enumerate(poly):
                x_el = etree.SubElement(
                    energy_el, f'x{i}')
                for j, c in enumerate(t_row):
                    x_el.set(f'a{j}', _fmt(c))
            V = ps.V_coeffs
            if V:
                V_el = etree.SubElement(
                    energy_el, 'V')
                for i, c in enumerate(V):
                    V_el.set(f'x{i}', _fmt(c))

        # Patch metadata (piecewise_patch only).
        if mt == 'piecewise_patch':
            lx = ps.patch_left_x
            lp = ps.patch_left_phase
            if lx is not None and lp:
                pl = etree.SubElement(
                    phase_el, 'patch_left')
                pl.set('x_cut', _fmt(lx))
                pl.set('phase', lp)
            rx = ps.patch_right_x
            rp = ps.patch_right_phase
            if rx is not None and rp:
                pr = etree.SubElement(
                    phase_el, 'patch_right')
                pr.set('x_cut', _fmt(rx))
                pr.set('phase', rp)


def _fmt(v):
    """Format a float for XML (no trailing zeros)."""
    return f'{v:.10g}'


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


def _simplex_grid(n_components, n_divisions):
    """Uniform grid on the (N-1)-simplex.

    Generates all compositions (x_1, ..., x_N) with
    x_i = k_i / n_divisions for non-negative integers
    k_i summing to n_divisions.  Returns the
    independent coordinates [x_2, ..., x_N] as an
    (m, N-1) array.  x_1 = 1 - sum is implied.

    For binary (N=2), returns shape (n_divisions+1, 1).

    Parameters
    ----------
    n_components : int — number of components N
    n_divisions  : int — grid resolution

    Returns
    -------
    ndarray, shape (m, N-1)
    """
    N = n_components
    n = n_divisions
    points = []

    def _recurse(remaining, depth, current):
        if depth == N - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i)
            _recurse(
                remaining - i,
                depth + 1, current)
            current.pop()

    _recurse(n, 0, [])

    arr = np.array(points, dtype=float) / n
    return arr[:, 1:]   # drop x_1; shape (m, N-1)


class Phase:
    """One thermodynamic phase.

    Attributes
    ----------
    name         : str   — user-given label
    phase_type   : str   — 'gas'|'liquid'|'solid'|
                           'end_member'
    energy_model : EnergyModel — from pde_energy
    xmin, xmax   : float — composition range (binary)
    n_components : int   — number of components in
                           the system (default 2)
    """

    def __init__(self, name, phase_type,
                 energy_model,
                 xmin=0.0, xmax=1.0,
                 n_components=2):
        self.name = name
        self.phase_type = phase_type
        self.energy_model = energy_model
        self.xmin = xmin
        self.xmax = xmax
        self.n_components = n_components

    @property
    def is_point(self):
        """True when this phase exists at a single
        composition (end-member).
        """
        return self.xmin == self.xmax

    def gibbs(self, x, field_values):
        """Evaluate G(x) at the given field values.

        Parameters
        ----------
        x : array-like
            Binary: shape (n,).
            Ternary+: shape (n, N-1).
        field_values : dict[str, float]
        """
        return self.energy_model.gibbs(
            x, field_values)

    def composition_grid(self, n_points=500):
        """Composition sample points.

        Binary:   (n,) array via linspace.
        Ternary+: (m, N-1) array on the simplex.
        """
        if self.is_point:
            return np.array([self.xmin])
        if self.n_components == 2:
            return np.linspace(
                self.xmin, self.xmax,
                n_points)
        # Simplex grid.  Choose n_divisions so
        # the point count is close to n_points.
        # For N=3: m = (d+1)(d+2)/2 ≈ n_points
        #   → d ≈ sqrt(2·n_points) - 1.5
        N = self.n_components
        d = max(10, int(
            (2.0 * n_points) ** (1.0 / (N - 1))
        ))
        return _simplex_grid(N, d)


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
