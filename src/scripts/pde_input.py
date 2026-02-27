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

from lxml import etree

from pde_energy import HSModel, PolyModel
from pde_phase import Phase, System


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
# HS form parsers
# ---------------------------------------------------------------------------

def _parse_hs_quadratic(energy_el, V_coeffs, ideal_gas, R_gas, P_ref):
    """Parse an HS-form quadratic energy block."""
    H_el = energy_el.find('H')
    S_el = energy_el.find('S')
    H_coeffs = _read_x_coeffs_from_attrs(H_el)
    S_coeffs = _read_x_coeffs_from_attrs(S_el)
    return HSModel(H_coeffs, S_coeffs,
                   V_coeffs=V_coeffs, ideal_gas=ideal_gas,
                   R_gas=R_gas, P_ref=P_ref)


def _parse_hs_point(energy_el, V_coeffs, ideal_gas, R_gas, P_ref):
    """Parse an HS-form point (end-member) energy block.

    Expects scalar text content inside <H> and <S>.
    A <V> element at point phases is unusual but allowed (single V0 scalar).
    """
    H_val = float(energy_el.find('H').text.strip())
    S_val = float(energy_el.find('S').text.strip())
    return HSModel([H_val], [S_val],
                   V_coeffs=V_coeffs, ideal_gas=ideal_gas,
                   R_gas=R_gas, P_ref=P_ref)


# ---------------------------------------------------------------------------
# Polynomial form parser
# ---------------------------------------------------------------------------

def _parse_poly_energy(energy_el, V_coeffs, ideal_gas, R_gas, P_ref):
    """Parse a polynomial-form energy block (quadratic or point).

    Reads all <xi> child elements and builds the coefficient table
    t_poly_coeffs[i] = T-polynomial for the x^i term.
    """
    t_coeffs_by_x = {}
    for child in energy_el:
        tag = child.tag
        if tag.startswith('x') and tag[1:].isdigit():
            i = int(tag[1:])
            t_coeffs_by_x[i] = _read_t_coeffs_from_attrs(child)
    if not t_coeffs_by_x:
        return PolyModel([[0.0]],
                         V_coeffs=V_coeffs, ideal_gas=ideal_gas,
                         R_gas=R_gas, P_ref=P_ref)
    max_order = max(t_coeffs_by_x.keys())
    coeffs = [t_coeffs_by_x.get(i, [0.0]) for i in range(max_order + 1)]
    return PolyModel(coeffs,
                     V_coeffs=V_coeffs, ideal_gas=ideal_gas,
                     R_gas=R_gas, P_ref=P_ref)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _parse_energy(energy_el, energy_form, ideal_gas, R_gas, P_ref):
    """Create and return the appropriate EnergyModel from an <energy> element."""
    model = energy_el.get('model')
    V_coeffs = _read_v_coeffs(energy_el)
    if energy_form == 'HS':
        if model == 'point':
            return _parse_hs_point(energy_el, V_coeffs, ideal_gas, R_gas, P_ref)
        return _parse_hs_quadratic(energy_el, V_coeffs, ideal_gas, R_gas, P_ref)
    else:
        return _parse_poly_energy(energy_el, V_coeffs, ideal_gas, R_gas, P_ref)


# ---------------------------------------------------------------------------
# Phase parser
# ---------------------------------------------------------------------------

def _parse_phase(phase_el, energy_form, R_gas, P_ref):
    """Create a Phase from a <phase> element."""
    name = phase_el.get('name')
    phase_type = phase_el.get('type')
    ideal_gas = phase_el.get('ideal_gas', 'false').lower() == 'true'

    cr_el = phase_el.find('composition_range')
    if cr_el is not None:
        xmin = float(cr_el.get('xmin'))
        xmax = float(cr_el.get('xmax'))
    else:
        xmin, xmax = 0.0, 1.0

    energy_el = phase_el.find('energy')
    model = _parse_energy(energy_el, energy_form, ideal_gas, R_gas, P_ref)

    return Phase(name=name, phase_type=phase_type, energy_model=model,
                 xmin=xmin, xmax=xmax)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_system(infile):
    """Parse *infile* (path to an XML input file) and return a System object."""
    tree = etree.parse(infile)
    root = tree.getroot()

    # System block
    sys_el = root.find('system')
    components = sys_el.find('components').text.strip().split()
    energy_form = sys_el.find('energy_form').text.strip()

    # Temperature block
    temp_el = root.find('temperature')
    T_min = float(temp_el.find('min').text)
    T_max = float(temp_el.find('max').text)
    T_initial = float(temp_el.find('initial').text)

    # Pressure block (optional)
    pres_el = root.find('pressure')
    if pres_el is not None:
        has_pressure = True
        P_min = float(pres_el.find('min').text)
        P_max = float(pres_el.find('max').text)
        P_initial = float(pres_el.find('initial').text)
        R_gas_el = pres_el.find('R_gas')
        R_gas = float(R_gas_el.text) if R_gas_el is not None else 0.0
        P_ref_el = pres_el.find('P_ref')
        P_ref = float(P_ref_el.text) if P_ref_el is not None else 1.0
    else:
        has_pressure = False
        P_min = P_max = P_initial = 1.0
        R_gas = 0.0
        P_ref = 1.0

    # Phases (preserve document order)
    phases = [_parse_phase(ph_el, energy_form, R_gas, P_ref)
              for ph_el in root.findall('phase')]

    return System(
        components=components,
        phases=phases,
        energy_form=energy_form,
        T_min=T_min,
        T_max=T_max,
        T_initial=T_initial,
        has_pressure=has_pressure,
        P_min=P_min,
        P_max=P_max,
        P_initial=P_initial,
        R_gas=R_gas,
        P_ref=P_ref,
    )
