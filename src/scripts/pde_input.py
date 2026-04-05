#!/usr/bin/env python3
"""
XML input parser for PDE.

Public entry point
------------------
  parse_system(infile) -> System

The input file selects one of two energy forms at the system level
(energy_form = 'HS' or 'polynomial'). That choice applies uniformly to
all phases; mixing forms within a single input file is not supported.

HS form energy blocks
---------------------
  <energy model="quadratic">
    <H x0="8.0" x1="-2.0" x2="2.0"/>
    <S x0="0.01" x1="0.0" x2="0.0"/>
    <V x0="0.02" x1="0.0" x2="0.0"/>   <!-- optional molar volume -->
  </energy>

  <energy model="point">   <!-- end-member: scalar H and S -->
    <H>0.5</H>
    <S>0.004</S>
  </energy>

Polynomial form energy blocks
------------------------------
  <energy model="quadratic">
    <x2 a0="-1.0" a1="0.05" a2="0.001"/>
    <x1 a0="0.0"  a1="0.0"/>
    <x0 a0="5.0"  a1="-2.0" a2="0.0003"/>
    <V  x0="0.02" x1="0.0"/>            <!-- optional molar volume -->
  </energy>

  <energy model="point">   <!-- end-member: only x0 term -->
    <x0 a0="-500.0" a1="-2.0" a2="0.001"/>
  </energy>

Pressure block (optional)
--------------------------
  <pressure>
    <min>0.5</min>
    <max>5.0</max>
    <initial>1.0</initial>
    <R_gas>8.314e-3</R_gas>   <!-- gas constant; kJ/mol/K if energy in kJ/mol -->
    <P_ref>1.0</P_ref>         <!-- reference pressure, same units as P -->
  </pressure>

Ideal-gas phases
----------------
  <phase name="vapor" type="gas" ideal_gas="true">
    ...
  </phase>

When ideal_gas="true", R·T·ln(P/P°) is added to G. Requires R_gas > 0 in
the <pressure> block and P > 0 at runtime.

Coefficient conventions (both forms)
--------------------------------------
  x-polynomial ascending order: x0 is the constant term, x1 the linear, etc.
  T-polynomial ascending order: a0 is the constant term, a1 the linear, etc.
"""

import pathlib

from lxml import etree

from pde_phase import FieldSpec, PhaseSpec, SystemSpec


# ---------------------------------------------------------------------------
# Coefficient readers
# ---------------------------------------------------------------------------

def _read_x_coeffs_from_attrs(element):
    """Read x0, x1, x2, ... attributes from *element* in ascending order.

    Returns a list of floats. Stops at the first missing index, so gaps are
    not supported (intentional: the XML should be unambiguous).
    """
    coeffs = []
    i = 0
    while f'x{i}' in element.attrib:
        coeffs.append(float(element.attrib[f'x{i}']))
        i += 1
    return coeffs


def _read_t_coeffs_from_attrs(element):
    """Read a0, a1, a2, ... attributes from *element* in ascending order.

    Returns a list of floats.
    """
    coeffs = []
    j = 0
    while f'a{j}' in element.attrib:
        coeffs.append(float(element.attrib[f'a{j}']))
        j += 1
    return coeffs


def _read_v_coeffs(energy_el):
    """Read the optional <V> child of *energy_el* and return V coefficients.

    Returns a list of floats (possibly empty → None returned to caller),
    or None if no <V> element is present.
    """
    v_el = energy_el.find('V')
    if v_el is None:
        return None
    return _read_x_coeffs_from_attrs(v_el) or None


# ---------------------------------------------------------------------------
# Energy coefficient extractors (data only — no model construction)
# ---------------------------------------------------------------------------

def _extract_hs_coeffs(energy_el):
    """Extract H, S, V coefficient lists from an HS-form energy block.

    Handles both quadratic (x0/x1/x2 attributes on <H>, <S>) and point
    (scalar text inside <H>, <S>) layouts.  Returns three plain lists
    ready to store in PhaseSpec.model_params.
    """
    V_coeffs = _read_v_coeffs(energy_el)
    if energy_el.get('model') == 'point':
        H_coeffs = [float(
            energy_el.find('H').text.strip())]
        S_coeffs = [float(
            energy_el.find('S').text.strip())]
    else:
        H_el = energy_el.find('H')
        S_el = energy_el.find('S')
        H_coeffs = _read_x_coeffs_from_attrs(H_el)
        S_coeffs = _read_x_coeffs_from_attrs(S_el)
    return H_coeffs, S_coeffs, V_coeffs


def _extract_poly_coeffs(energy_el):
    """Extract polynomial coefficient table from a poly-form energy block.

    Reads all <xi> children and builds the T-polynomial table indexed by
    x-power.  Returns (poly_coeffs, V_coeffs) — both plain lists ready
    to store in PhaseSpec.model_params.
    """
    t_coeffs_by_x = {}
    for child in energy_el:
        tag = child.tag
        if (tag.startswith('x')
                and tag[1:].isdigit()):
            i = int(tag[1:])
            t_coeffs_by_x[i] = (
                _read_t_coeffs_from_attrs(child))
    V_coeffs = _read_v_coeffs(energy_el)
    if not t_coeffs_by_x:
        return [[0.0]], V_coeffs
    max_order = max(t_coeffs_by_x.keys())
    poly_coeffs = [
        t_coeffs_by_x.get(i, [0.0])
        for i in range(max_order + 1)]
    return poly_coeffs, V_coeffs


# ---------------------------------------------------------------------------
# Phase spec parser
# ---------------------------------------------------------------------------

def _parse_phase_spec(phase_el, energy_form):
    """Parse a <phase> element into a PhaseSpec.

    Always returns a PhaseSpec — no sentinel dicts, no model objects.
    Cross-phase references (VLE liquid_phase) are left blank here and
    filled in by parse_system() after all phases have been parsed.
    """
    name = phase_el.get('name')
    phase_type = phase_el.get('type')
    ideal_gas = (phase_el.get('ideal_gas', 'false')
                 .lower() == 'true')

    cr_el = phase_el.find('composition_range')
    if cr_el is not None:
        xmin = float(cr_el.get('xmin'))
        xmax = float(cr_el.get('xmax'))
    else:
        xmin, xmax = 0.0, 1.0

    # -- VLE gas (no <energy> block; derives H/S from liquid at build time)
    vle_el = phase_el.find('vle')
    if vle_el is not None:
        return PhaseSpec(
            name=name,
            phase_type=phase_type,
            xmin=xmin, xmax=xmax,
            model_type='HS',
            model_params={
                'ideal_gas': ideal_gas,
                'vle_params': {
                    'liquid_phase':
                        vle_el.get(
                            'liquid_phase', ''),
                    'T_bp_A': float(
                        vle_el.get('T_bp_A')),
                    'T_bp_B': float(
                        vle_el.get('T_bp_B')),
                    'L_A': float(
                        vle_el.get('L_A')),
                    'L_B': float(
                        vle_el.get('L_B')),
                },
            })

    # -- Normal energy block → extract coefficients
    energy_el = phase_el.find('energy')

    if energy_form == 'HS':
        H_coeffs, S_coeffs, V_coeffs = (
            _extract_hs_coeffs(energy_el))
        model_params = {
            'H_coeffs': H_coeffs,
            'S_coeffs': S_coeffs,
            'V_coeffs': V_coeffs,
            'ideal_gas': ideal_gas,
        }
        # Check for patch elements (HS form only).
        patch_left_el = phase_el.find('patch_left')
        patch_right_el = phase_el.find(
            'patch_right')
        if (patch_left_el is not None
                or patch_right_el is not None):
            model_params['patch_left_phase'] = (
                patch_left_el.get('phase', '')
                if patch_left_el is not None
                else None)
            model_params['patch_left_x'] = (
                float(patch_left_el.get('x_cut'))
                if patch_left_el is not None
                else None)
            model_params['patch_right_phase'] = (
                patch_right_el.get('phase', '')
                if patch_right_el is not None
                else None)
            model_params['patch_right_x'] = (
                float(patch_right_el.get('x_cut'))
                if patch_right_el is not None
                else None)
            return PhaseSpec(
                name=name,
                phase_type=phase_type,
                xmin=xmin, xmax=xmax,
                model_type='piecewise_patch',
                model_params=model_params)

        return PhaseSpec(
            name=name,
            phase_type=phase_type,
            xmin=xmin, xmax=xmax,
            model_type='HS',
            model_params=model_params)

    # -- Polynomial energy form
    poly_coeffs, V_coeffs = (
        _extract_poly_coeffs(energy_el))
    return PhaseSpec(
        name=name,
        phase_type=phase_type,
        xmin=xmin, xmax=xmax,
        model_type='polynomial',
        model_params={
            'poly_coeffs': poly_coeffs,
            'V_coeffs': V_coeffs,
            'ideal_gas': ideal_gas,
        })


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_system(infile):
    """Parse *infile* and return a System.

    Internally constructs a SystemSpec from the XML, then calls
    spec.to_system() which handles all dependency resolution (VLE,
    patches) via topological sort.

    Accepts two field schemas (new <fields> block or legacy separate
    <temperature>/<pressure> blocks) — both produce identical results.
    """
    tree = etree.parse(infile)
    root = tree.getroot()

    # -- System metadata ---------------------------------------------------
    sys_el = root.find('system')
    components = (
        sys_el.find('components')
        .text.strip().split())
    energy_form = (
        sys_el.find('energy_form')
        .text.strip())

    # -- Title (optional; fall back to filename) ---------------------------
    title_el = root.find('title')
    if title_el is not None and title_el.text:
        title = title_el.text.strip()
    else:
        fname = pathlib.Path(infile).name
        for ext in ('.xml', '.in'):
            if fname.endswith(ext):
                fname = fname[:-len(ext)]
        title = fname

    # -- Fields ------------------------------------------------------------
    fields_el = root.find('fields')
    if fields_el is not None:
        field_specs = _parse_field_specs_block(
            fields_el)
    else:
        field_specs = _parse_legacy_field_specs(
            root)

    # -- Phases ------------------------------------------------------------
    phase_specs = [
        _parse_phase_spec(ph_el, energy_form)
        for ph_el in root.findall('phase')]

    # Post-process: fill in VLE liquid_phase references.  The XML does not
    # name the liquid explicitly; the convention is "first liquid phase
    # in declaration order".
    liquid_name = ''
    for ps in phase_specs:
        if ps.phase_type == 'liquid':
            liquid_name = ps.name
            break
    for ps in phase_specs:
        vle = ps.model_params.get('vle_params')
        if vle is not None and not vle.get(
                'liquid_phase'):
            vle['liquid_phase'] = liquid_name

    # -- Assemble spec and build -------------------------------------------
    spec = SystemSpec(
        title=title,
        components=components,
        energy_form=energy_form,
        fields=field_specs,
        phases=phase_specs,
    )
    return spec.to_system()


# ---------------------------------------------------------------------------
# Field spec parsers
# ---------------------------------------------------------------------------

def _parse_field_specs_block(fields_el):
    """Parse a <fields> element into a list of FieldSpec objects.

    R_gas and P_ref are stored in the pressure FieldSpec's extras dict
    (consumed by make_energy_model at build time).
    """
    field_specs = []
    for f_el in fields_el.findall('field'):
        name = f_el.get('name')
        symbol = f_el.get('symbol', name)
        unit = f_el.get('unit', '')
        min_val = float(f_el.get('min'))
        max_val = float(f_el.get('max'))
        initial_val = float(f_el.get('initial'))
        extras = {}
        if name == 'pressure':
            if f_el.get('R_gas') is not None:
                extras['R_gas'] = float(
                    f_el.get('R_gas'))
            if f_el.get('P_ref') is not None:
                extras['P_ref'] = float(
                    f_el.get('P_ref'))
        field_specs.append(FieldSpec(
            name=name, symbol=symbol,
            unit=unit, min_val=min_val,
            max_val=max_val,
            initial_val=initial_val,
            extras=extras))
    return field_specs


def _parse_legacy_field_specs(root):
    """Parse legacy <temperature> and optional <pressure> blocks.

    Returns a list of FieldSpec objects — same shape as
    _parse_field_specs_block.
    """
    temp_el = root.find('temperature')
    T_min = float(temp_el.find('min').text)
    T_max = float(temp_el.find('max').text)
    T_initial = float(
        temp_el.find('initial').text)
    t_spec = FieldSpec(
        name='temperature', symbol='T',
        unit='K', min_val=T_min,
        max_val=T_max,
        initial_val=T_initial)

    pres_el = root.find('pressure')
    if pres_el is None:
        return [t_spec]

    P_min = float(pres_el.find('min').text)
    P_max = float(pres_el.find('max').text)
    P_initial = float(
        pres_el.find('initial').text)
    R_gas_el = pres_el.find('R_gas')
    P_ref_el = pres_el.find('P_ref')
    unit_el = pres_el.find('unit')
    extras = {}
    if R_gas_el is not None:
        extras['R_gas'] = float(R_gas_el.text)
    if P_ref_el is not None:
        extras['P_ref'] = float(P_ref_el.text)
    P_unit = (unit_el.text.strip()
              if unit_el is not None else '')

    p_spec = FieldSpec(
        name='pressure', symbol='P',
        unit=P_unit, min_val=P_min,
        max_val=P_max,
        initial_val=P_initial,
        extras=extras)
    return [t_spec, p_spec]
