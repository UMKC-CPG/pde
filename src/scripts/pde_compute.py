#!/usr/bin/env python3
"""
Equilibrium computation for PDE.

Public API
----------
  compute_equilibrium(system, field_values,
                      n_points=500) -> EqResult

Strategy
--------
  1. Evaluate G(x) for every phase on its composition
     grid, passing the field_values dict to each
     phase.gibbs() call.
  2. Collect all (x, G) points (tagged with their
     phase index) into one array.
  3. Compute the lower convex hull of that point
     cloud via scipy.spatial.ConvexHull.
  4. Walk hull vertices left-to-right.  Consecutive
     vertices from the same phase → single-phase
     region; different phases → two-phase region
     (common tangent).

This approach generalises naturally to higher-order
systems because ConvexHull works in any dimension.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class EqResult:
    """Equilibrium state at one set of field values.

    Attributes
    ----------
    field_values   : dict[str, float] — field name →
        value at which equilibrium was computed
        (e.g. {'temperature': 1200, 'pressure': 2})
    phase_curves   : list[(x, G, Phase)] — one entry
        per phase, same order as system.phases
    hull_x         : ndarray — composition coords of
        lower-hull vertices, sorted left to right
    hull_G         : ndarray — G at those vertices
    hull_phase_idx : ndarray[int] — phase indices
    regions        : list[dict] — left to right;
        keys: 'type', 'x0', 'x1', 'phases'
    """

    def __init__(self, field_values, phase_curves,
                 hull_x, hull_G, hull_phase_idx,
                 regions):
        self.field_values = dict(field_values)
        self.phase_curves = phase_curves
        self.hull_x = hull_x
        self.hull_G = hull_G
        self.hull_phase_idx = hull_phase_idx
        self.regions = regions

    # Convenience accessors for the two most common
    # fields — used extensively by the viz layer.

    @property
    def T(self):
        """Temperature value (shorthand)."""
        return self.field_values.get(
            'temperature', 0.0)

    @property
    def P(self):
        """Pressure value (shorthand)."""
        return self.field_values.get(
            'pressure', 0.0)

    @property
    def two_phase_regions(self):
        return [r for r in self.regions
                if r['type'] == 'two_phase']

    @property
    def single_phase_regions(self):
        return [r for r in self.regions
                if r['type'] == 'single']


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _extract_regions_binary(hull_x, hull_phase_idx,
                            all_x=None,
                            all_phase_idx=None):
    """Walk lower hull edges left-to-right for
    binary systems.

    Consecutive same-phase edges are merged into
    one single-phase region.  Zero-width regions
    (duplicate x values from coincident phase
    endpoints) are discarded.

    When the full composition grid (*all_x*,
    *all_phase_idx*) is provided, within-phase
    miscibility gaps are detected.  A same-phase
    hull edge that skips many grid points from
    that phase is a gap: the hull tangent line
    bridges a concave interior.  Such edges are
    labeled 'two_phase' with phases = (pi, pi).

    Parameters
    ----------
    hull_x, hull_phase_idx : ndarray
        Sorted lower-hull vertex data.
    all_x : ndarray or None
        Full composition grid (all phases).
    all_phase_idx : ndarray or None
        Phase index for each grid point.
    """
    raw = []
    for j in range(len(hull_x) - 1):
        x0 = hull_x[j]
        x1 = hull_x[j + 1]
        if x1 - x0 < 1e-12:
            continue
        pi0 = int(hull_phase_idx[j])
        pi1 = int(hull_phase_idx[j + 1])

        if pi0 != pi1:
            raw.append({
                'type': 'two_phase',
                'x0': x0, 'x1': x1,
                'phases': (pi0, pi1)})
            continue

        # Same phase on both sides.  Check for
        # a within-phase gap: if the hull edge
        # skips many grid points from this phase,
        # the tangent line bridges a concave
        # region → miscibility gap.
        is_gap = False
        if all_x is not None:
            mask = (
                (all_x > x0 + 1e-10)
                & (all_x < x1 - 1e-10)
                & (all_phase_idx == pi0))
            n_skipped = int(np.sum(mask))
            if n_skipped > 5:
                is_gap = True

        if is_gap:
            raw.append({
                'type': 'two_phase',
                'x0': x0, 'x1': x1,
                'phases': (pi0, pi0)})
        else:
            raw.append({
                'type': 'single',
                'x0': x0, 'x1': x1,
                'phases': (pi0,)})

    # Merge consecutive single-phase edges from
    # the same phase.
    merged = []
    for r in raw:
        if (r['type'] == 'single'
                and merged
                and merged[-1]['type'] == 'single'
                and merged[-1]['phases']
                == r['phases']):
            merged[-1]['x1'] = r['x1']
        else:
            merged.append(dict(r))

    return merged


def _extract_regions_nd(hull, lower_simplices,
                        all_phase_idx, points):
    """Extract phase regions from the lower hull
    of an N-D (ternary+) system.

    Groups lower-hull simplices by their phase-set
    signature.  Returns a list of dicts with:
      'phases' : frozenset of phase indices
      'simplices' : list of hull simplex indices
      'type' : 'single' | 'two_phase' | 'multi'
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for si in lower_simplices:
        verts = hull.simplices[si]
        phase_set = frozenset(
            int(all_phase_idx[v])
            for v in verts)
        groups[phase_set].append(si)

    regions = []
    for phase_set, simplices in groups.items():
        n_phases = len(phase_set)
        if n_phases == 1:
            rtype = 'single'
        elif n_phases == 2:
            rtype = 'two_phase'
        else:
            rtype = 'multi'
        regions.append({
            'type': rtype,
            'phases': tuple(sorted(phase_set)),
            'simplices': simplices,
        })

    return regions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_equilibrium(system, field_values,
                        n_points=500):
    """Compute equilibrium at the given field values.

    Parameters
    ----------
    system       : System
    field_values : dict[str, float] — e.g.
        {'temperature': 1200.0, 'pressure': 2.0}
    n_points     : int — composition grid density

    Returns
    -------
    EqResult

    Handles both binary (scalar x) and
    multi-component (vector x) systems.  The hull
    is computed in (N-1+1)-dimensional space where
    N = len(system.components).
    """
    fv = dict(field_values)
    N = len(system.components)

    # --- Step 1: evaluate G(x) per phase ----------
    phase_curves = []
    all_x_parts = []
    all_G_parts = []
    all_idx_parts = []

    for i, phase in enumerate(system.phases):
        x = phase.composition_grid(n_points)
        G = phase.gibbs(x, fv)
        phase_curves.append((x, G, phase))

        # Ensure x is 2-D for stacking.
        x_2d = (np.atleast_2d(x).T
                if x.ndim == 1 else x)
        all_x_parts.append(x_2d)
        all_G_parts.append(G)
        all_idx_parts.append(
            np.full(len(G), i, dtype=int))

    # all_x: (total_pts, N-1)
    all_x = np.vstack(all_x_parts)
    all_G = np.concatenate(all_G_parts)
    all_phase_idx = np.concatenate(
        all_idx_parts)

    # --- Step 2: lower convex hull ----------------
    # points: (total_pts, N) where last col is G.
    points = np.column_stack([all_x, all_G])
    # G is the last column: index N-1.
    g_col = points.shape[1] - 1

    try:
        hull = ConvexHull(points)
    except QhullError:
        if N == 2:
            sort_idx = np.argsort(all_x[:, 0])
            return EqResult(
                field_values=fv,
                phase_curves=phase_curves,
                hull_x=all_x[sort_idx, 0],
                hull_G=all_G[sort_idx],
                hull_phase_idx=(
                    all_phase_idx[sort_idx]),
                regions=[],
            )
        return EqResult(
            field_values=fv,
            phase_curves=phase_curves,
            hull_x=all_x,
            hull_G=all_G,
            hull_phase_idx=all_phase_idx,
            regions=[],
        )

    # Lower hull: simplices whose outward normal
    # has a negative G-component.
    lower_simplices = [
        i for i, eq in enumerate(
            hull.equations)
        if eq[g_col] < 0]

    lower_vertex_set = set()
    for si in lower_simplices:
        lower_vertex_set.update(
            hull.simplices[si])

    # --- Step 3: classify regions -----------------
    if N == 2:
        # Binary path: 1-D walk (unchanged).
        lv = np.array(sorted(
            lower_vertex_set,
            key=lambda v: points[v, 0]))
        hull_x = points[lv, 0]
        hull_G = points[lv, g_col]
        hull_phase_idx = all_phase_idx[lv]
        # Pass the full grid so the extractor
        # can detect within-phase miscibility
        # gaps (hull edges that skip many grid
        # points from the same phase).
        all_x_flat = all_x[:, 0]
        regions = _extract_regions_binary(
            hull_x, hull_phase_idx,
            all_x_flat, all_phase_idx)
    else:
        # Multi-component: simplex-based regions.
        lv = np.array(sorted(lower_vertex_set))
        hull_x = points[lv, :g_col]
        hull_G = points[lv, g_col]
        hull_phase_idx = all_phase_idx[lv]
        regions = _extract_regions_nd(
            hull, lower_simplices,
            all_phase_idx, points)

    return EqResult(
        field_values=fv,
        phase_curves=phase_curves,
        hull_x=hull_x,
        hull_G=hull_G,
        hull_phase_idx=hull_phase_idx,
        regions=regions,
    )
