#!/usr/bin/env python3
"""
Equilibrium computation for PDE.

Public API
----------
  compute_equilibrium(system, T, P=0.0, n_points=500) -> EqResult

Strategy
--------
  1. Evaluate G(x, T, P) for every phase on its composition grid.
  2. Collect all (x, G) points (tagged with their phase index) into one array.
  3. Compute the lower convex hull of that point cloud via scipy.spatial.ConvexHull.
     The lower hull is the set of hull facets whose outward normal has a
     negative y-component (equations[:, 1] < 0).
  4. Walk the hull vertices left-to-right in composition. Consecutive vertices
     from the same phase → single-phase region. Consecutive vertices from
     different phases → two-phase region (common tangent).

This approach generalises naturally to higher-order systems (ternary, etc.)
because ConvexHull operates in arbitrary dimensions.

Variable pressure
-----------------
The optional P argument is forwarded to every phase.gibbs() call. When P=0.0
(the default) the PV and ideal-gas terms in the energy models contribute zero,
so existing call sites that omit P are fully backward-compatible.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class EqResult:
    """Equilibrium state of a system at one temperature (and pressure).

    Attributes
    ----------
    T             : float
    P             : float — pressure at which equilibrium was computed
    phase_curves  : list of (x_array, G_array, Phase) — one entry per phase,
                    in the same order as system.phases; used to draw the G-x canvas.
    hull_x        : ndarray — composition coordinates of lower hull vertices,
                    sorted left to right.
    hull_G        : ndarray — G values at those vertices.
    hull_phase_idx: ndarray[int] — index into system.phases for each vertex.
    regions       : list[dict]  — ordered left to right, each dict has:
                      'type'  : 'single' or 'two_phase'
                      'x0'    : float  — left composition bound
                      'x1'    : float  — right composition bound
                      'phases': tuple[int]  — one entry for 'single',
                                             two entries for 'two_phase'
    """

    def __init__(self, T, phase_curves, hull_x, hull_G, hull_phase_idx, regions, P=0.0):
        self.T = T
        self.P = P
        self.phase_curves = phase_curves
        self.hull_x = hull_x
        self.hull_G = hull_G
        self.hull_phase_idx = hull_phase_idx
        self.regions = regions

    @property
    def two_phase_regions(self):
        return [r for r in self.regions if r['type'] == 'two_phase']

    @property
    def single_phase_regions(self):
        return [r for r in self.regions if r['type'] == 'single']


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _extract_regions(hull_x, hull_phase_idx):
    """Walk lower hull edges and classify each as single- or two-phase.

    Consecutive same-phase edges are merged into one single-phase region.
    Zero-width regions (duplicate x values from coincident phase endpoints)
    are discarded.
    """
    raw = []
    for j in range(len(hull_x) - 1):
        x0 = hull_x[j]
        x1 = hull_x[j + 1]
        if x1 - x0 < 1e-12:        # skip zero-width edges (coincident endpoints)
            continue
        pi0 = int(hull_phase_idx[j])
        pi1 = int(hull_phase_idx[j + 1])
        if pi0 != pi1:
            raw.append({'type': 'two_phase', 'x0': x0, 'x1': x1,
                        'phases': (pi0, pi1)})
        else:
            raw.append({'type': 'single', 'x0': x0, 'x1': x1,
                        'phases': (pi0,)})

    # Merge consecutive single-phase edges that belong to the same phase.
    merged = []
    for r in raw:
        if (r['type'] == 'single'
                and merged
                and merged[-1]['type'] == 'single'
                and merged[-1]['phases'] == r['phases']):
            merged[-1]['x1'] = r['x1']
        else:
            merged.append(dict(r))

    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_equilibrium(system, T, P=0.0, n_points=500):
    """Compute the equilibrium phase assemblage at temperature *T* and pressure *P*.

    Parameters
    ----------
    system   : System
    T        : float — temperature (same units as energy model coefficients)
    P        : float — pressure (same units as energy model V coefficients);
               default 0.0 keeps PV and ideal-gas terms inactive for backward
               compatibility with files that have no pressure block.
    n_points : int   — composition grid density for non-point phases

    Returns
    -------
    EqResult
    """
    # --- Step 1: evaluate G(x, T, P) for each phase ----------------------
    phase_curves = []
    all_x_parts = []
    all_G_parts = []
    all_idx_parts = []

    for i, phase in enumerate(system.phases):
        x = phase.composition_grid(n_points)
        G = phase.gibbs(x, T, P)
        phase_curves.append((x, G, phase))
        all_x_parts.append(x)
        all_G_parts.append(G)
        all_idx_parts.append(np.full(len(x), i, dtype=int))

    all_x = np.concatenate(all_x_parts)
    all_G = np.concatenate(all_G_parts)
    all_phase_idx = np.concatenate(all_idx_parts)

    # --- Step 2: lower convex hull ----------------------------------------
    points = np.column_stack([all_x, all_G])

    try:
        hull = ConvexHull(points)
    except QhullError:
        # Degenerate case (e.g. all points collinear) — fall back to the
        # raw point cloud sorted by x; region extraction will be empty.
        sort_idx = np.argsort(all_x)
        return EqResult(
            T=T,
            P=P,
            phase_curves=phase_curves,
            hull_x=all_x[sort_idx],
            hull_G=all_G[sort_idx],
            hull_phase_idx=all_phase_idx[sort_idx],
            regions=[],
        )

    # Lower hull: facets whose outward normal points downward (eq[i, 1] < 0).
    lower_vertex_set = set()
    for i, simplex in enumerate(hull.simplices):
        if hull.equations[i, 1] < 0:
            lower_vertex_set.update(simplex)

    lv = np.array(sorted(lower_vertex_set, key=lambda v: points[v, 0]))

    hull_x = points[lv, 0]
    hull_G = points[lv, 1]
    hull_phase_idx = all_phase_idx[lv]

    # --- Step 3: classify regions -----------------------------------------
    regions = _extract_regions(hull_x, hull_phase_idx)

    return EqResult(
        T=T,
        P=P,
        phase_curves=phase_curves,
        hull_x=hull_x,
        hull_G=hull_G,
        hull_phase_idx=hull_phase_idx,
        regions=regions,
    )
