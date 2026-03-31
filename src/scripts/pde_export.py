#!/usr/bin/env python3
"""
HDF5 + XDMF export for PDE phase diagrams.

Public API
----------
  export_binary_Tx(system, filepath, n_x=200, n_field=200,
                   primary_field_idx=0, fixed_field_values=None) -> str
      Compute equilibrium on a structured triangulated mesh over
      (composition, primary_field) and write HDF5 + XDMF files.
      Returns the path to the XDMF file.

  export_3d_TPx(system, filepath, n_x=100, n_T=100, n_P=50,
                progress_cb=None) -> str
      Compute equilibrium on a 3D structured mesh over
      (composition, temperature, pressure) and write HDF5 + XDMF files.
      Returns the path to the XDMF file.

2D HDF5 layout
---------------
  /mesh/nodes           Float64 (N_nodes, 3)   — (x, field_value, 0) coordinates
  /mesh/triangles       Int32   (N_tri, 3)     — triangle connectivity
  /fields/phase_id      Int16   (N_nodes,)     — stable phase index
  /fields/G_min         Float64 (N_nodes,)     — minimum G at each node
  /fields/phase_frac    Float64 (N_nodes,)     — lever-rule fraction (0 or 1 in
                                                  single-phase regions)
  /boundaries/...       boundary polylines
  /phases/names         String  (N_phases,)
  /phases/types         String  (N_phases,)
  /system/...           metadata scalars

3D HDF5 layout
---------------
  /mesh/nodes           Float64 (n_P, n_T, n_x, 3) — (x, T, P) coordinates
  /fields/phase_id      Int16   (n_P, n_T, n_x)    — stable phase index
  /fields/G_min         Float64 (n_P, n_T, n_x)    — minimum G at each node
  /fields/phase_frac    Float64 (n_P, n_T, n_x)    — lever-rule fraction
  /phases/names         String  (N_phases,)
  /phases/types         String  (N_phases,)
  /system/...           metadata scalars

File naming
-----------
  Given an input stem (e.g. ``ABC``), the exported files are:
    ABC.hdf5   — HDF5 data
    ABC.xdmf   — XDMF wrapper referencing the HDF5
    ABC.pv.py  — ParaView Python helper script (3D only)

XDMF wrapper
-------------
  A companion .xdmf file references the HDF5 arrays so ParaView can
  open the dataset directly.
"""

import os

import numpy as np
import h5py

from pde_compute import compute_equilibrium


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def _build_structured_tri_mesh(n_x, n_field, x_range, field_range):
    """Build a structured triangular mesh on the rectangle [x_range] x [field_range].

    Each rectangular cell is split into two triangles (lower-left and upper-right).

    Parameters
    ----------
    n_x         : int — number of composition nodes
    n_field     : int — number of field (T, P, ...) nodes
    x_range     : (float, float) — (x_min, x_max)
    field_range : (float, float) — (field_min, field_max)

    Returns
    -------
    nodes     : ndarray (N_nodes, 3)  — (x, field_value, 0) per node
    triangles : ndarray (N_tri, 3)    — node indices per triangle
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    f_vals = np.linspace(field_range[0], field_range[1], n_field)

    # Node coordinates — row-major: field varies slowest.
    # Store as XYZ (z=0) for ParaView compatibility.
    xx, ff = np.meshgrid(x_vals, f_vals, indexing='xy')
    nodes = np.column_stack([xx.ravel(), ff.ravel(),
                             np.zeros(xx.size)])

    # Triangle connectivity — two triangles per quad.
    triangles = []
    for j in range(n_field - 1):
        for i in range(n_x - 1):
            # Node indices of the four quad corners.
            bl = j * n_x + i          # bottom-left
            br = bl + 1               # bottom-right
            tl = (j + 1) * n_x + i   # top-left
            tr = tl + 1               # top-right
            triangles.append([bl, br, tl])
            triangles.append([br, tr, tl])

    return nodes, np.array(triangles, dtype=np.int32)


# ---------------------------------------------------------------------------
# Phase classification at grid nodes
# ---------------------------------------------------------------------------

def _classify_nodes_vec(x_vals, eq_result):
    """Classify an array of x values against one EqResult.

    Vectorised replacement for the per-node _classify_node().

    Parameters
    ----------
    x_vals    : ndarray (n_x,) — composition values
    eq_result : EqResult

    Returns
    -------
    phase_id   : ndarray[int16]   (n_x,)
    G_min      : ndarray[float64] (n_x,)
    phase_frac : ndarray[float64] (n_x,)
    """
    n = len(x_vals)
    phase_id = np.zeros(n, dtype=np.int16)
    G_min = np.zeros(n, dtype=np.float64)
    phase_frac = np.ones(n, dtype=np.float64)

    regions = eq_result.regions
    if not regions:
        return phase_id, G_min, phase_frac

    # Interpolate G for all x in one call.
    G_min[:] = np.interp(
        x_vals, eq_result.hull_x, eq_result.hull_G,
        left=eq_result.hull_G[0],
        right=eq_result.hull_G[-1])

    # Build region lookup: assign each x to a region.
    assigned = np.zeros(n, dtype=bool)
    for r in regions:
        x0, x1 = r['x0'], r['x1']
        mask = (~assigned
                & (x_vals >= x0 - 1e-9)
                & (x_vals <= x1 + 1e-9))
        if not mask.any():
            continue

        if r['type'] == 'single':
            phase_id[mask] = r['phases'][0]
            # phase_frac already 1.0
        else:
            span = x1 - x0
            if span < 1e-12:
                frac = np.full(mask.sum(), 0.5)
            else:
                frac = (x_vals[mask] - x0) / span
            phase_frac[mask] = frac
            pid = np.where(
                frac < 0.5,
                r['phases'][0],
                r['phases'][1])
            phase_id[mask] = pid

        assigned |= mask

    # Fallback for unassigned nodes: clamp to nearest.
    if not assigned.all():
        left = x_vals < regions[0]['x0']
        right = ~assigned & ~left
        if left.any():
            phase_id[left] = regions[0]['phases'][0]
        if right.any():
            phase_id[right] = regions[-1]['phases'][0]

    return phase_id, G_min, phase_frac


def _compute_fields_on_mesh(system, nodes, n_x, n_field, field_range,
                            primary_field_idx, fixed_field_values,
                            progress_cb=None):
    """Evaluate phase_id, G_min, phase_frac at every mesh node.

    Computes one EqResult per field-value row, then classifies all x nodes
    in that row — O(n_field) hull computations, not O(n_field * n_x).

    Parameters
    ----------
    progress_cb : callable(done, total) or None

    Returns
    -------
    phase_id   : ndarray[int16] (N_nodes,)
    G_min      : ndarray[float64] (N_nodes,)
    phase_frac : ndarray[float64] (N_nodes,)
    eq_results : list[EqResult] — one per field row (for boundary extraction)
    """
    n_nodes = nodes.shape[0]
    phase_id = np.zeros(n_nodes, dtype=np.int16)
    G_min = np.zeros(n_nodes, dtype=np.float64)
    phase_frac = np.zeros(n_nodes, dtype=np.float64)

    f_vals = np.linspace(field_range[0], field_range[1], n_field)
    x_vals = nodes[:n_x, 0]  # first row of x values

    pf = system.fields[primary_field_idx]
    fv = dict(fixed_field_values) if fixed_field_values else {}

    eq_results = []
    for j, fval in enumerate(f_vals):
        if progress_cb is not None:
            progress_cb(j, n_field)

        fv[pf.name] = fval
        T = fv.get('temperature', 0.0)
        P = fv.get('pressure', 0.0)
        eq = compute_equilibrium(system, T, P)
        eq_results.append(eq)

        row_start = j * n_x
        row_end = row_start + n_x
        pid, g, fr = _classify_nodes_vec(x_vals, eq)
        phase_id[row_start:row_end] = pid
        G_min[row_start:row_end] = g
        phase_frac[row_start:row_end] = fr

    if progress_cb is not None:
        progress_cb(n_field, n_field)

    return phase_id, G_min, phase_frac, eq_results


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _extract_boundaries(eq_results, f_vals):
    """Extract phase boundary polylines from a sequence of EqResult objects.

    For each consecutive pair of EqResults, finds two-phase region endpoints
    and connects them into polylines.

    Returns list of dicts: {'points': ndarray (M, 2), 'phase_left': int,
                            'phase_right': int}
    """
    # Collect boundary segments keyed by (phase_a, phase_b) with a < b.
    from collections import defaultdict
    boundary_segments = defaultdict(list)

    for j, eq in enumerate(eq_results):
        fval = f_vals[j]
        for r in eq.regions:
            if r['type'] == 'two_phase':
                pa, pb = r['phases']
                key = (min(pa, pb), max(pa, pb))
                # Store left and right boundary x at this field value.
                boundary_segments[key].append((fval, r['x0'], r['x1']))

    boundaries = []
    for (pa, pb), segments in boundary_segments.items():
        segments.sort(key=lambda s: s[0])  # sort by field value
        f_arr = np.array([s[0] for s in segments])
        x_left = np.array([s[1] for s in segments])
        x_right = np.array([s[2] for s in segments])

        # Left boundary polyline (x0 side).
        pts_left = np.column_stack([x_left, f_arr])
        boundaries.append({
            'points': pts_left,
            'phase_left': pa,
            'phase_right': pb,
        })
        # Right boundary polyline (x1 side).
        pts_right = np.column_stack([x_right, f_arr])
        boundaries.append({
            'points': pts_right,
            'phase_left': pa,
            'phase_right': pb,
        })

    return boundaries


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def _write_hdf5(filepath, nodes, triangles, phase_id, G_min, phase_frac,
                boundaries, system, primary_field_idx):
    """Write the phase diagram data to an HDF5 file."""
    pf = system.fields[primary_field_idx]

    with h5py.File(filepath, 'w') as f:
        # -- mesh --
        grp = f.create_group('mesh')
        grp.create_dataset('nodes', data=nodes, dtype='float64')
        grp.create_dataset('triangles', data=triangles, dtype='int32')

        # -- fields --
        grp = f.create_group('fields')
        grp.create_dataset('phase_id', data=phase_id, dtype='int16')
        grp.create_dataset('G_min', data=G_min, dtype='float64')
        grp.create_dataset('phase_frac', data=phase_frac, dtype='float64')

        # -- boundaries --
        grp = f.create_group('boundaries')
        grp.attrs['n_boundaries'] = len(boundaries)
        for i, bnd in enumerate(boundaries):
            bgrp = grp.create_group(f'boundary_{i}')
            bgrp.create_dataset('points', data=bnd['points'], dtype='float64')
            bgrp.attrs['phase_left'] = bnd['phase_left']
            bgrp.attrs['phase_right'] = bnd['phase_right']

        # -- phases --
        grp = f.create_group('phases')
        names = [p.name for p in system.phases]
        types = [p.phase_type for p in system.phases]
        dt = h5py.string_dtype()
        grp.create_dataset('names', data=names, dtype=dt)
        grp.create_dataset('types', data=types, dtype=dt)

        # -- system metadata --
        grp = f.create_group('system')
        grp.attrs['title'] = system.title or ''
        grp.attrs['components'] = system.components
        grp.attrs['energy_form'] = system.energy_form
        grp.attrs['field_name'] = pf.name
        grp.attrs['field_symbol'] = pf.symbol
        grp.attrs['field_unit'] = pf.unit
        grp.attrs['field_min'] = pf.min_val
        grp.attrs['field_max'] = pf.max_val


# ---------------------------------------------------------------------------
# XDMF writer
# ---------------------------------------------------------------------------

def _write_xdmf_2d(xdmf_path, h5_filename, n_nodes, n_tri):
    """Write an XDMF file referencing the companion HDF5 (2D triangle mesh)."""
    xml = f'''\
<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="PhaseDiagram" GridType="Uniform">

      <Topology TopologyType="Triangle"
                NumberOfElements="{n_tri}">
        <DataItem Format="HDF" DataType="Int"
                  Dimensions="{n_tri} 3">
          {h5_filename}:/mesh/triangles
        </DataItem>
      </Topology>

      <Geometry GeometryType="XYZ">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_nodes} 3">
          {h5_filename}:/mesh/nodes
        </DataItem>
      </Geometry>

      <Attribute Name="phase_id" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Int"
                  Dimensions="{n_nodes}">
          {h5_filename}:/fields/phase_id
        </DataItem>
      </Attribute>

      <Attribute Name="G_min" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_nodes}">
          {h5_filename}:/fields/G_min
        </DataItem>
      </Attribute>

      <Attribute Name="phase_frac" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_nodes}">
          {h5_filename}:/fields/phase_frac
        </DataItem>
      </Attribute>

    </Grid>
  </Domain>
</Xdmf>
'''
    with open(xdmf_path, 'w') as f:
        f.write(xml)


def _write_xdmf_3d(xdmf_path, h5_filename, n_x, n_T, n_P):
    """Write an XDMF file for a 3D structured mesh (T-P-x)."""
    n_nodes = n_x * n_T * n_P
    xml = f'''\
<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="PhaseDiagram3D" GridType="Uniform">

      <Topology TopologyType="3DSMesh"
                Dimensions="{n_P} {n_T} {n_x}"/>

      <Geometry GeometryType="XYZ">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_nodes} 3">
          {h5_filename}:/mesh/nodes
        </DataItem>
      </Geometry>

      <Attribute Name="phase_id" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Int"
                  Dimensions="{n_P} {n_T} {n_x}">
          {h5_filename}:/fields/phase_id
        </DataItem>
      </Attribute>

      <Attribute Name="G_min" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_P} {n_T} {n_x}">
          {h5_filename}:/fields/G_min
        </DataItem>
      </Attribute>

      <Attribute Name="phase_frac" Center="Node"
                 AttributeType="Scalar">
        <DataItem Format="HDF" DataType="Float"
                  Dimensions="{n_P} {n_T} {n_x}">
          {h5_filename}:/fields/phase_frac
        </DataItem>
      </Attribute>

    </Grid>
  </Domain>
</Xdmf>
'''
    with open(xdmf_path, 'w') as f:
        f.write(xml)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_binary_Tx(system, filepath, n_x=200, n_field=200,
                     primary_field_idx=0, fixed_field_values=None,
                     progress_cb=None):
    """Export a binary field-x phase diagram to HDF5 + XDMF.

    Parameters
    ----------
    system              : System
    filepath            : str — file stem or .hdf5 path (derives .xdmf, .pv.py)
    n_x                 : int — composition grid points
    n_field             : int — field (T, P, ...) grid points
    primary_field_idx   : int — which field to sweep (default 0 = temperature)
    fixed_field_values  : dict or None — {field_name: value} for non-primary fields
    progress_cb         : callable(done, total) or None — progress callback

    Returns
    -------
    str — path to the XDMF file
    """
    if fixed_field_values is None:
        fixed_field_values = {}
        for i, f in enumerate(system.fields):
            if i != primary_field_idx:
                fixed_field_values[f.name] = f.initial_val

    pf = system.fields[primary_field_idx]

    # Composition range: union of all phase x-ranges.
    x_min = min(p.xmin for p in system.phases)
    x_max = max(p.xmax for p in system.phases)
    x_range = (x_min, x_max)
    field_range = (pf.min_val, pf.max_val)

    # Build mesh.
    nodes, triangles = _build_structured_tri_mesh(n_x, n_field, x_range,
                                                  field_range)

    # Compute phase fields.
    phase_id, G_min, phase_frac, eq_results = _compute_fields_on_mesh(
        system, nodes, n_x, n_field, field_range,
        primary_field_idx, fixed_field_values, progress_cb)

    # Extract boundaries.
    f_vals = np.linspace(field_range[0], field_range[1], n_field)
    boundaries = _extract_boundaries(eq_results, f_vals)

    # Ensure output directory exists.
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Derive file stem.
    if filepath.endswith('.hdf5'):
        stem = filepath[:-5]
    elif filepath.endswith('.h5'):
        stem = filepath[:-3]
    else:
        stem = filepath

    h5_path = stem + '.hdf5'
    _write_hdf5(h5_path, nodes, triangles,
                phase_id, G_min, phase_frac,
                boundaries, system, primary_field_idx)

    # Write XDMF.
    xdmf_path = stem + '.xdmf'
    h5_filename = os.path.basename(h5_path)
    _write_xdmf_2d(xdmf_path, h5_filename,
                    nodes.shape[0],
                    triangles.shape[0])

    # Emit companion ParaView controls plugin.
    ctrl_path = stem + '.ctrl.py'
    phase_names = [p.name for p in system.phases]
    diagram_title = system.title or 'Phase Diagram'
    _write_paraview_plugin(
        ctrl_path, phase_names, diagram_title)

    # Emit companion ParaView Python script.
    pv_script_path = stem + '.pv.py'
    _write_paraview_script_2d(
        pv_script_path,
        os.path.basename(xdmf_path),
        os.path.basename(ctrl_path),
        pf, x_range, phase_names, diagram_title)

    return xdmf_path


def export_3d_TPx(system, filepath, n_x=100, n_T=100, n_P=50,
                  progress_cb=None, precomputed_grid=None):
    """Export a 3D T-P-x phase diagram to HDF5 + XDMF.

    Requires a system with both temperature and pressure fields.

    Coordinates are normalised to [0, 1] on each axis so the volume
    renders as a cube in ParaView.  The physical ranges are stored as
    HDF5 attributes under /system and a companion ``.pv.py``
    script is emitted that configures the Axes Grid with proper titles,
    tick positions, and physical-value labels.

    Parameters
    ----------
    system      : System
    filepath    : str — file stem or .hdf5 path (derives .xdmf, .pv.py)
    n_x         : int — composition grid points
    n_T         : int — temperature grid points
    n_P         : int — pressure grid points
    progress_cb : callable(done, total) or None
    precomputed_grid : list[list[EqResult]] or None
        Cached grid from FullGridWorker (T-major: grid[i_T][i_P]).
        When supplied and dimensions match (n_T × n_P), the
        equilibrium results are reused instead of recomputed.

    Returns
    -------
    str — path to the XDMF file
    """
    T_field = system.T_field
    P_field = system.P_field
    if T_field is None or P_field is None:
        raise ValueError('export_3d_TPx requires a system with both '
                         'temperature and pressure fields.')

    x_min = min(p.xmin for p in system.phases)
    x_max = max(p.xmax for p in system.phases)

    x_vals = np.linspace(x_min, x_max, n_x)
    T_vals = np.linspace(T_field.min_val, T_field.max_val, n_T)
    P_vals = np.linspace(P_field.min_val, P_field.max_val, n_P)

    # Check whether the precomputed grid can be reused.
    use_cache = False
    if precomputed_grid is not None:
        if (len(precomputed_grid) == n_T
                and len(precomputed_grid[0]) == n_P):
            use_cache = True

    # Normalised coordinates [0, 1].
    x_norm = np.linspace(0.0, 1.0, n_x)
    T_norm = np.linspace(0.0, 1.0, n_T)
    P_norm = np.linspace(0.0, 1.0, n_P)

    total = n_T * n_P
    n_nodes = n_x * n_T * n_P

    # Allocate flat arrays — will be reshaped for HDF5.
    nodes = np.empty((n_nodes, 3), dtype=np.float64)
    phase_id = np.empty(n_nodes, dtype=np.int16)
    G_min = np.empty(n_nodes, dtype=np.float64)
    phase_frac = np.empty(n_nodes, dtype=np.float64)

    # Pre-build the node coordinate x-column (same for every row).
    x_col = np.column_stack([
        x_norm,
        np.zeros(n_x),
        np.zeros(n_x)])

    done = 0
    for kp, P in enumerate(P_vals):
        for jt, T in enumerate(T_vals):
            if use_cache:
                # Grid is T-descending, P-descending;
                # export is T-ascending, P-ascending.
                eq = precomputed_grid[n_T - 1 - jt][n_P - 1 - kp]
            else:
                eq = compute_equilibrium(system, T, P)

            base = (kp * n_T + jt) * n_x
            sl = slice(base, base + n_x)

            # Coordinates — update y, z columns.
            nodes[sl] = x_col
            nodes[sl, 1] = T_norm[jt]
            nodes[sl, 2] = P_norm[kp]

            # Classify all x values at once.
            pid, g, fr = _classify_nodes_vec(x_vals, eq)
            phase_id[sl] = pid
            G_min[sl] = g
            phase_frac[sl] = fr

            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    # Reshape field arrays to 3D for the structured mesh.
    shape_3d = (n_P, n_T, n_x)
    phase_id_3d = phase_id.reshape(shape_3d)
    G_min_3d = G_min.reshape(shape_3d)
    phase_frac_3d = phase_frac.reshape(shape_3d)

    # Ensure output directory exists.
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Derive file stem.
    if filepath.endswith('.hdf5'):
        stem = filepath[:-5]
    elif filepath.endswith('.h5'):
        stem = filepath[:-3]
    else:
        stem = filepath

    h5_path = stem + '.hdf5'

    # Axis metadata for the ParaView helper script.
    axes_meta = {
        'x_min': x_min, 'x_max': x_max,
        'T_min': T_field.min_val, 'T_max': T_field.max_val,
        'T_unit': T_field.unit,
        'P_min': P_field.min_val, 'P_max': P_field.max_val,
        'P_unit': P_field.unit,
    }

    with h5py.File(h5_path, 'w') as f:
        grp = f.create_group('mesh')
        grp.create_dataset('nodes', data=nodes, dtype='float64')

        grp = f.create_group('fields')
        grp.create_dataset('phase_id', data=phase_id_3d, dtype='int16')
        grp.create_dataset('G_min', data=G_min_3d, dtype='float64')
        grp.create_dataset('phase_frac', data=phase_frac_3d, dtype='float64')

        grp = f.create_group('phases')
        dt = h5py.string_dtype()
        grp.create_dataset('names',
                           data=[p.name for p in system.phases], dtype=dt)
        grp.create_dataset('types',
                           data=[p.phase_type for p in system.phases], dtype=dt)

        grp = f.create_group('system')
        grp.attrs['title'] = system.title or ''
        grp.attrs['components'] = system.components
        grp.attrs['energy_form'] = system.energy_form
        for k, v in axes_meta.items():
            grp.attrs[k] = v

    xdmf_path = stem + '.xdmf'
    h5_filename = os.path.basename(h5_path)
    _write_xdmf_3d(xdmf_path, h5_filename, n_x, n_T, n_P)

    # Emit companion ParaView controls plugin.
    ctrl_path = stem + '.ctrl.py'
    phase_names = [p.name for p in system.phases]
    diagram_title = system.title or 'Phase Diagram'
    _write_paraview_plugin(
        ctrl_path, phase_names, diagram_title)

    # Emit companion ParaView Python script.
    pv_script_path = stem + '.pv.py'
    _write_paraview_script(
        pv_script_path,
        os.path.basename(xdmf_path),
        os.path.basename(ctrl_path),
        axes_meta, phase_names, diagram_title)

    return xdmf_path


def _write_paraview_plugin(plugin_path,
                           phase_names, title):
    """Write a ParaView Python plugin (.ctrl.py).

    The plugin registers a source ("PDE Controls")
    with a Properties panel containing:
      - Opacity slider (0--100)
      - Per-phase visibility checkboxes
      - ColourBy dropdown (phase_id/G_min/phase_frac)

    It is a source (no input needed) that discovers
    per-phase Threshold sources named "PDE: <phase>"
    (created by the companion .pv.py) and controls
    their display properties as a batch operation.
    """
    # Build per-phase setter methods.
    phase_setters = []
    for i, name in enumerate(phase_names):
        phase_setters.append(f'''\
    @smproperty.intvector(
        name="Show {name}",
        default_values=1)
    @smdomain.xml(
        '<BooleanDomain name="bool"/>')
    def SetShow_{i}(self, val):
        self._show["{name}"] = bool(val)
        self.Modified()
''')
    setter_block = '\n'.join(phase_setters)

    phase_list_str = ', '.join(
        f'"{n}"' for n in phase_names)

    n_phases = len(phase_names)

    script = f'''\
"""
PDE Controls -- ParaView plugin for: {title}

Generated by PDE export.  Loaded automatically
by the companion .pv.py script.

Properties panel:
  Opacity       -- batch opacity (0--100)
  Show <phase>  -- per-phase visibility
  ColourBy      -- phase_id / G_min / phase_frac

Phases: [{phase_list_str}]
"""
from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
)
from vtkmodules.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
)
from paraview.util.vtkAlgorithm import (
    smproxy, smproperty, smdomain,
)


@smproxy.source(label="PDE Controls")
class PDEControlsSource(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType="vtkPolyData")
        self._opacity = 100
        self._colour_idx = 0
        self._prev_colour_idx = -1
        self._first_run = True
        self._show = {{}}
        self._prev_show = {{}}

    # ---- Batch opacity --------------------------

    @smproperty.intvector(
        name="Opacity", default_values=100)
    @smdomain.intrange(min=0, max=100)
    def SetOpacity(self, val):
        self._opacity = val
        self.Modified()

    # ---- Phase visibility -----------------------

{setter_block}
    # ---- Colour-by dropdown ---------------------

    @smproperty.intvector(
        name="ColourBy", default_values=0)
    @smdomain.xml(
        """<EnumerationDomain name="enum">
        <Entry value="0" text="phase_id"/>
        <Entry value="1" text="G_min"/>
        <Entry value="2" text="phase_frac"/>
    </EnumerationDomain>""")
    def SetColourBy(self, val):
        self._colour_idx = val
        self.Modified()

    # ---- Pipeline -------------------------------

    def RequestData(self, request,
                    inInfo, outInfo):
        import traceback

        out = vtkPolyData.GetData(outInfo, 0)
        # Output is intentionally empty --
        # this source is a control panel.

        try:
            from paraview.simple import (
                GetActiveView,
                GetSources, GetDisplayProperties,
                ColorBy, FindSource,
                GetColorTransferFunction)

            view = GetActiveView()
            if view is None:
                return 1

            # Hide our own empty output so it
            # does not affect the scalar bar.
            if self._first_run:
                self._first_run = False
                me = FindSource(
                    "PDE Controls")
                if me is not None:
                    d = GetDisplayProperties(
                        me, view=view)
                    if d is not None:
                        d.Visibility = 0

            fields = [
                "phase_id", "G_min",
                "phase_frac"]
            fld = fields[self._colour_idx]
            alpha = self._opacity / 100.0

            colour_changed = (
                self._colour_idx
                != self._prev_colour_idx)

            n_pde = 0
            for (name, _id), src in (
                    GetSources().items()):
                if not name.startswith("PDE: "):
                    continue
                n_pde += 1
                phase = name[5:]
                vis = self._show.get(
                    phase, True)
                prev_vis = (
                    self._prev_show.get(
                        phase, True))
                d = GetDisplayProperties(
                    src, view=view)
                if d is None:
                    continue
                # Only touch Visibility when it
                # actually changed — redundant
                # sets trigger re-entrant
                # pipeline updates.
                if vis != prev_vis:
                    d.Visibility = (
                        1 if vis else 0)
                if vis:
                    if colour_changed:
                        ColorBy(
                            d, ("POINTS", fld))
                    d.Opacity = alpha

            # Fix LUT range after a colour
            # change so the scalar bar shows
            # the full range.
            if colour_changed and n_pde > 0:
                lut = (
                    GetColorTransferFunction(
                        fld))
                if fld == "phase_id":
                    lut.RescaleTransferFunction(
                        0, max(n_pde - 1, 1))

            self._prev_colour_idx = (
                self._colour_idx)
            self._prev_show = dict(
                self._show)

        except Exception:
            traceback.print_exc()

        return 1
'''
    with open(plugin_path, 'w') as f:
        f.write(script)


def _write_paraview_script_2d(script_path,
                              xdmf_filename,
                              ctrl_filename,
                              primary_field,
                              x_range,
                              phase_names, title):
    """Write a ParaView Python helper for a 2D export.

    Coordinates are physical (x, field_value, 0) so
    no normalisation is needed.
    """
    x_min, x_max = x_range
    f_sym = primary_field.symbol
    f_unit = primary_field.unit
    f_min = primary_field.min_val
    f_max = primary_field.max_val

    phase_list_str = ', '.join(
        f'"{n}"' for n in phase_names)
    n_phases = len(phase_names)

    x_title = (
        f'Composition  '
        f'[{x_min:.4g} -- {x_max:.4g}]')
    y_title = (
        f'{f_sym}  '
        f'[{f_min:.4g} -- {f_max:.4g} {f_unit}]')

    script = f'''\
"""
ParaView helper for: {title}  (2D {f_sym}-x)

Generated by PDE export.  Run inside ParaView:
  Tools -> Python Shell -> Run Script

Axes:
  X = composition  [{x_min:.4g}, {x_max:.4g}]
  Y = {f_sym}  [{f_min:.4g}, {f_max:.4g}] {f_unit}

Phases: [{phase_list_str}]

Helper functions (call from the Python shell):
  set_opacity(0.6)
  set_opacity(0.6, "liquid")
  color_by("G_min")
  hide_phase("liquid")
  show_phase("liquid")
  show_all()
  pde_help()
"""
import os

from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# ---- Load XDMF ----
script_dir = os.path.dirname(
    os.path.abspath(__file__))
xdmf_path = os.path.join(
    script_dir, "{xdmf_filename}")

try:
    reader = Xdmf3ReaderS(
        FileName=[xdmf_path])
except NameError:
    reader = XDMFReader(
        FileNames=[xdmf_path])
reader.UpdatePipeline()

# ---- View setup ----
view = GetActiveViewOrCreate("RenderView")
view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle = "{x_title}"
view.AxesGrid.YTitle = "{y_title}"
view.AxesGrid.ZTitle = ""

# ---- Per-phase Threshold filters ----
PHASE_NAMES = [{phase_list_str}]
N_PHASES = {n_phases}

phase_thresholds = []
phase_displays = []
for pid in range(N_PHASES):
    th = Threshold(
        registrationName="PDE: "
        + PHASE_NAMES[pid],
        Input=reader)
    th.Scalars = ["POINTS", "phase_id"]
    th.LowerThreshold = pid
    th.UpperThreshold = pid
    th.UpdatePipeline()
    disp = Show(th, view)
    ColorBy(disp, ("POINTS", "phase_id"))
    disp.Opacity = 1.0
    phase_thresholds.append(th)
    phase_displays.append(disp)

Hide(reader, view)

if phase_displays:
    pd = phase_displays[0]
    pd.RescaleTransferFunctionToDataRange(
        True, False)
    pd.SetScalarBarVisibility(view, True)

Render()
ResetCamera()

# Look down the Z axis for a 2D view.
view.CameraPosition = [0.5, 0.5, 10]
view.CameraFocalPoint = [0.5, 0.5, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelProjection = 1
ResetCamera()

# ========================================
# Helper functions
# ========================================

def set_opacity(value, phase=None):
    """Set opacity for all or one phase."""
    for i, d in enumerate(phase_displays):
        if phase is None or \\
                PHASE_NAMES[i] == phase:
            d.Opacity = value
    Render()

def color_by(field="phase_id"):
    """Colour by: phase_id, G_min, phase_frac."""
    for d in phase_displays:
        ColorBy(d, ("POINTS", field))
        d.RescaleTransferFunctionToDataRange(
            True, False)
    Render()

def hide_phase(name):
    """Hide a phase by name."""
    for i, nm in enumerate(PHASE_NAMES):
        if nm == name:
            Hide(phase_thresholds[i], view)
    Render()

def show_phase(name):
    """Show a phase by name."""
    for i, nm in enumerate(PHASE_NAMES):
        if nm == name:
            Show(phase_thresholds[i], view)
    Render()

def show_all():
    """Show all phases."""
    for th in phase_thresholds:
        Show(th, view)
    Render()

def pde_help():
    """Print available helper functions."""
    print("PDE helper functions:")
    print("  set_opacity(0.6)")
    print("  set_opacity(0.6, 'liquid')")
    print("  color_by('G_min')")
    print("  hide_phase('liquid')")
    print("  show_phase('liquid')")
    print("  show_all()")
    print("Phases:", PHASE_NAMES)

print()
print("PDE phase diagram loaded (2D).")
print("Phases:", PHASE_NAMES)
print()
print("To load the controls plugin:")
print("  Manage Plugins -> Load New ->",
      "{ctrl_filename}")
print("  Then: Sources -> PDE Controls"
      " -> Apply")
print()
print("Type pde_help() for helper functions.")
'''
    with open(script_path, 'w') as f:
        f.write(script)


def _write_paraview_script(script_path,
                           xdmf_filename,
                           ctrl_filename,
                           axes_meta,
                           phase_names, title):
    """Write a ParaView Python helper script.

    When run inside ParaView (Tools -> Python Shell ->
    Run Script), it loads the XDMF via Xdmf3ReaderS,
    configures Axes Grid with physical-value labels,
    creates per-phase Threshold filters, loads the
    companion .ctrl.py plugin, and defines helper
    functions for the Python shell.
    """
    x_min = axes_meta['x_min']
    x_max = axes_meta['x_max']
    T_min = axes_meta['T_min']
    T_max = axes_meta['T_max']
    P_min = axes_meta['P_min']
    P_max = axes_meta['P_max']
    T_unit = axes_meta['T_unit']
    P_unit = axes_meta['P_unit']

    phase_list_str = ', '.join(
        f'"{n}"' for n in phase_names)

    x_title = f'x  [{x_min:.4g} -- {x_max:.4g}]'
    T_title = (
        f'T  [{T_min:.4g} -- {T_max:.4g}'
        f' {T_unit}]')
    P_title = (
        f'P  [{P_min:.4g} -- {P_max:.4g}'
        f' {P_unit}]')

    tick_pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_phases = len(phase_names)

    script = f'''\
"""
ParaView helper for: {title}

Generated by PDE export.  Run inside ParaView:
  Tools -> Python Shell -> Run Script

Coordinate mapping (normalised -> physical):
  x: [0,1] -> [{x_min:.4g}, {x_max:.4g}]
  T: [0,1] -> [{T_min:.4g}, {T_max:.4g}] {T_unit}
  P: [0,1] -> [{P_min:.4g}, {P_max:.4g}] {P_unit}

Phases: [{phase_list_str}]

Helper functions (call from the Python shell):
  set_opacity(0.6)          — set all phases
  set_opacity(0.6, "liquid") — set one phase
  color_by("G_min")         — phase_id, G_min,
                               or phase_frac
  show_phase("liquid")      — show a phase
  hide_phase("liquid")      — hide a phase
  show_all()                — show all phases
  pde_help()                — print this list
"""
import os

from paraview.simple import *

# Disable automatic camera reset on Show.
paraview.simple._DisableFirstRenderCameraReset()

# ---- Load XDMF ----
script_dir = os.path.dirname(
    os.path.abspath(__file__))
xdmf_path = os.path.join(
    script_dir, "{xdmf_filename}")

# ParaView 6.x uses Xdmf3ReaderS; older versions
# use XDMFReader.  Try both.
try:
    reader = Xdmf3ReaderS(
        FileName=[xdmf_path])
except NameError:
    reader = XDMFReader(
        FileNames=[xdmf_path])
reader.UpdatePipeline()

# ---- Axes Grid ----
view = GetActiveViewOrCreate("RenderView")
view.AxesGrid.Visibility = 1

view.AxesGrid.XTitle = "{x_title}"
view.AxesGrid.YTitle = "{T_title}"
view.AxesGrid.ZTitle = "{P_title}"

view.AxesGrid.XAxisUseCustomLabels = 1
view.AxesGrid.XAxisLabels = {tick_pos!r}
view.AxesGrid.YAxisUseCustomLabels = 1
view.AxesGrid.YAxisLabels = {tick_pos!r}
view.AxesGrid.ZAxisUseCustomLabels = 1
view.AxesGrid.ZAxisLabels = {tick_pos!r}

# ---- Per-phase Threshold filters ----
PHASE_NAMES = [{phase_list_str}]
N_PHASES = {n_phases}

phase_thresholds = []
phase_displays = []
for pid in range(N_PHASES):
    th = Threshold(
        registrationName="PDE: "
        + PHASE_NAMES[pid],
        Input=reader)
    th.Scalars = ["POINTS", "phase_id"]
    th.LowerThreshold = pid
    th.UpperThreshold = pid
    th.UpdatePipeline()
    disp = Show(th, view,
        "UnstructuredGridRepresentation")
    ColorBy(disp, ("POINTS", "phase_id"))
    disp.Opacity = 1.0
    phase_thresholds.append(th)
    phase_displays.append(disp)

Hide(reader, view)

if phase_displays:
    pd = phase_displays[0]
    pd.RescaleTransferFunctionToDataRange(
        True, False)
    pd.SetScalarBarVisibility(view, True)

Render()
ResetCamera()

# ============================================================
# Helper functions — call from the Python shell
# ============================================================

def set_opacity(value, phase=None):
    """Set opacity (0.0 -- 1.0) for all or one phase."""
    for i, d in enumerate(phase_displays):
        if phase is None or \\
                PHASE_NAMES[i] == phase:
            d.Opacity = value
    Render()

def color_by(field="phase_id"):
    """Colour all phases by a field.
    Options: 'phase_id', 'G_min', 'phase_frac'.
    """
    for d in phase_displays:
        ColorBy(d, ("POINTS", field))
        d.RescaleTransferFunctionToDataRange(
            True, False)
    Render()

def hide_phase(name):
    """Hide a phase by name."""
    for i, nm in enumerate(PHASE_NAMES):
        if nm == name:
            Hide(phase_thresholds[i], view)
    Render()

def show_phase(name):
    """Show a phase by name."""
    for i, nm in enumerate(PHASE_NAMES):
        if nm == name:
            Show(phase_thresholds[i], view,
                "UnstructuredGridRepresentation")
    Render()

def show_all():
    """Show all phases."""
    for th in phase_thresholds:
        Show(th, view,
            "UnstructuredGridRepresentation")
    Render()

def pde_help():
    """Print available helper functions."""
    print("PDE helper functions:")
    print("  set_opacity(0.6)")
    print("  set_opacity(0.6, 'liquid')")
    print("  color_by('G_min')")
    print("  hide_phase('liquid')")
    print("  show_phase('liquid')")
    print("  show_all()")
    print("Phases:", PHASE_NAMES)

print()
print("PDE phase diagram loaded.")
print('Phases: [{phase_list_str}]')
print()
print("To load the controls plugin:")
print("  Manage Plugins -> Load New ->",
      "{ctrl_filename}")
print("  Then: Sources -> PDE Controls"
      " -> Apply")
print()
print("Type pde_help() for available "
      "helper functions.")
'''
    with open(script_path, 'w') as f:
        f.write(script)
