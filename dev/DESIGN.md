# Design

> **Document hierarchy:** VISION → ARCHITECTURE → **DESIGN** → PSEUDOCODE →
> Code. For goals and principles, see `VISION.md`. For repository layout and
> module map, see `ARCHITECTURE.md`.

---

## 1. Core Data Structures

### 1.1 Field (`pde_phase.py`)

```python
@dataclass
class Field:
    name:        str      # 'temperature', 'pressure', …
    symbol:      str      # 'T', 'P', …
    unit:        str      # 'K', 'atm', …; '' if none
    min_val:     float
    max_val:     float
    initial_val: float
```

Role (primary sweep axis / secondary / fixed) is **not** stored on `Field`;
it belongs to the view configuration in `MainWindow`. `fields[0]` is temperature
by convention.

### 1.2 Phase (`pde_phase.py`)

```python
class Phase:
    name:         str
    phase_type:   str          # 'gas'|'liquid'|'solid'|'end_member'
    energy_model: EnergyModel
    xmin:         float
    xmax:         float
```

`Phase.gibbs(x, T, P)` delegates to `energy_model.gibbs(x, T, P)`. End-member
phases have `xmin == xmax`.

### 1.3 System (`pde_phase.py`)

```python
class System:
    components:  list[str]
    phases:      list[Phase]
    energy_form: str             # 'HS' or 'polynomial'
    fields:      list[Field]
    title:       str
```

Backward-compatible properties (`T_field`, `T_min/max/initial`, `P_field`,
`has_pressure`, `P_min/max/initial/unit`, `R_gas`, `P_ref`) delegate into
`self.fields` and `self.phases`.

### 1.4 EqResult (`pde_compute.py`)

```python
class EqResult:
    T:              float
    P:              float
    phase_curves:   list[(x_arr, G_arr, Phase)]
    hull_x:         ndarray        # lower hull, left→right
    hull_G:         ndarray
    hull_phase_idx: ndarray[int]
    regions:        list[dict]
    # region: {'type': 'single'|'two_phase',
    #          'x0': float, 'x1': float,
    #          'phases': tuple[int]}
```

---

## 2. Energy Model Hierarchy (`pde_energy.py`)

### 2.1 Abstract base

`EnergyModel` (ABC) defines `_gibbs_impl(x, field_values: dict)`. The public
`gibbs()` is currently a backward-compat shim that accepts both the old-style
`gibbs(x, T, P=0.0)` positional call and the new-style `gibbs(x, dict)` call.
See issue 12.6 for the retirement plan.

### 2.2 HSModel

```
G(x, T, P) = H(x) − T·S(x)  [+ P·V(x)]  [+ R·T·ln(P/P°)]
```

H, S, V are polynomials in ascending order: `[c0, c1, c2, …]` means c0 +
c1·x + c2·x² + …. Optional Poynting correction via V_coeffs. Optional ideal-gas
term via `ideal_gas=True`, `R_gas`, `P_ref`.

### 2.3 PolyModel

```
G(x, T, P) = Σᵢ cᵢ(T)·xⁱ  [+ P·V(x)]  [+ R·T·ln(P/P°)]
```

Each coefficient cᵢ(T) is itself a polynomial in T. Same ascending order
convention. Same optional Poynting and ideal-gas terms as HSModel.

### 2.4 PiecewisePatchModel

Uses H_orig(x) in the interior. Optional H_left(x) for
x ≤ x_cut_left and H_right(x) for x > x_cut_right. S(x)
is unchanged across the full range. The patch polynomials
are constructed to match G and dG/dx of a target phase at
the cut point, ensuring C¹ continuity.

#### Concavity limitation of endpoint patches

Early prototypes explored a stronger approach: cubic or
quintic Hermite H-patch/S-patch corrections at composition
endpoints (x = 0, x = 1) to enforce both G-value and
G-slope matching at transition temperatures (the "lens
must close to a point" requirement). The correction was
decomposed into h_corr(x) and s_corr(x) to keep
T-dependence linear within the HS framework.

This approach has a fundamental limitation. Any smooth
polynomial correction that enforces non-zero slope at the
endpoint and fades to zero at x_inner must have an
inflection point. The entropy slope term enters G as
−T · s_corr(x), so the resulting concavity grows linearly
with T. This creates a concave dimple inside the patch
region that causes spurious qhull artifacts (discontinuous
jumps in tie-line endpoints at some critical temperature).
Upgrading from cubic to quintic Hermite (C² at x_inner)
postpones the artifact to lower T but does not eliminate
it — the dimple remains in the interior of the patch.

Disabling the slope correction (h_d = s_d = 0) eliminates
the artifact but sacrifices the smooth lens-closing
property. The current PiecewisePatchModel avoids the
problem entirely by matching G and dG/dx at a single
interior cut point rather than at the composition
endpoints, which does not require a correction that fades
to zero and therefore avoids the inflection-point
conflict. The endpoint lens-closing problem remains open;
a future solution would likely require either a
non-polynomial correction shape or a pre-convexification
step before the global common-tangent construction.

### 2.5 CouplingTerm

```python
@dataclass
class CouplingTerm:
    response_coeffs: list[float]  # R(x) polynomial
    coupling_type:   str          # 'linear'|'ideal_gas'|'power'
    field_names:     list[str]
    params:          dict
```

Describes how one field (or pair of fields) enters G beyond the base H-S or
polynomial form.

### 2.6 VLE gas construction

`compute_vle_gas_hs(liq_H, liq_S, T_bp_A, T_bp_B, L_A, L_B)` derives the
gas-phase `(H_gas, S_gas)` satisfying all four VLE tangency conditions by
construction. The gas H and S are computed so that G_gas = G_liq and dG_gas/dx
= dG_liq/dx at x=0 (T=T_bp_A) and x=1 (T=T_bp_B). `HSModel.vle_params`
stores the original boiling-point parameters for round-tripping.

### 2.7 Coefficient convention

All coefficient lists use **ascending** order throughout the codebase: `[c0,
c1, c2, …]` = c0 + c1·x + c2·x² + …. This applies to H, S, V, polynomial
coefficients, patch polynomials, and response coefficients.

---

## 3. XML Input (`pde_input.py`)

### 3.1 XML → SystemSpec → System

`parse_system()` is a thin XML → `SystemSpec` translator.
It builds a list of `FieldSpec` and `PhaseSpec` objects
from the XML elements, assembles a `SystemSpec`, and
calls `spec.to_system()`.  All dependency resolution
(VLE gas derivation, patch-H construction) happens inside
`to_system()` via topological sort — the parser itself
has no multi-pass logic.

The XML uses a `<fields>` block with `<field name=…
symbol=… unit=… min=… max=… initial=… />` elements.
R_gas and P_ref live on the `<field name="pressure">`
element's extras.  A legacy fallback parser for old
`<temperature>`/`<pressure>` blocks still exists in the
code (`_parse_legacy_field_specs`) but is dead code
slated for removal — backward compatibility with prior
XML formats is not required.

### 3.2 VLE elements

VLE gas phases use a `<vle T_bp_A=… T_bp_B=… L_A=…
L_B=…/>` element inside `<phase>`.  The parser stores
these as `vle_params` inside the `PhaseSpec.model_params`
dict.  The liquid-phase reference is filled in by
convention (first liquid in declaration order) after all
phase specs are parsed.

### 3.3 Optional elements

`<title>`, `<V x0=… x1=…/>` inside `<energy>` (molar volume), `ideal_gas=
"true"` on `<phase>`, `R_gas`/`P_ref` on the pressure element.

---

## 4. Equilibrium Computation (`pde_compute.py`)

### 4.1 Lower convex hull

1. Evaluate `G(x, T, P)` on each phase's composition grid (default 500 points
   per phase, clipped to [xmin, xmax]).
2. Collect all `(x, G)` points into one array.
3. Compute `scipy.spatial.ConvexHull`; extract the lower hull (facets whose
   outward normal points downward, i.e., normal[1] < 0).
4. Walk hull vertices left-to-right; classify edges as single-phase or
   two-phase (common-tangent) regions.

The algorithm generalises to higher-order systems (ternary and beyond) because
`ConvexHull` operates in arbitrary dimensions.

### 4.2 Region classification

Each region in `EqResult.regions` is a dict:

- `'single'`: one phase is lowest in `[x0, x1]`.
- `'two_phase'`: a common tangent connects two phases; the mixture is a
  mechanical blend of the end-member compositions.

---

## 5. Visualization (`pde_viz.py`)

### 5.1 Layout

```
┌──────────────────────────────────────────────────────┐
│  GxCanvas (left)     │   SweepCanvas (right)          │
│  G(x) at current     │   field-x phase diagram        │
│  field values        │   (one canvas per field)        │
├──────────────────────────────────────────────────────┤
│  Mode combo  |  Field sliders (one per field)         │
└──────────────────────────────────────────────────────┘
```

### 5.2 GxCanvas

Draws G(x) curves + lower convex envelope + common tangent lines.

**Edit mode `'handles'`:** 3 diamond handles per HS phase. Vertical drag →
quadratic H fit via 3×3 Vandermonde solve. Horizontal drag (endpoint handles)
→ xmin/xmax update with 0.02 clamp margin.

**Ctrl+click → rigid shift:** moves the whole curve as a rigid body in both
x (translates composition range) and y (shifts G offset). Axes autoscaling is
frozen during the drag. Reparameterises H and S polynomials via the function
`_shift_poly_coeffs` (composition p(x) → p(x−δ)).

Emits `phase_edited(phase_name, PhaseData)` on drag release.

### 5.3 SweepCanvas

Draws the full phase diagram for one primary field in a single pass. All
regions are pre-drawn; a white cover rectangle hides the not-yet-revealed
portion. `update_reveal(value)` is O(1): adjusts the cover height.
`reset(precomputed)` redraws from new data after a secondary slider release
triggers a full recompute.

One SweepCanvas per field; created lazily on first mode switch.

### 5.4 Slider update cycle

Constants: `N_T_STEPS=200`, `N_P_STEPS=200`. One slider per
`system.fields[i]`, range 0…N_STEPS−1 mapped to field.min_val…max_val.

- **Primary slider:** O(1) lookup in `_precomputed[primary_idx]` →
  `GxCanvas.redraw(result)` + `SweepCanvas.update_reveal(value)`.
- **Secondary slider release:** full recompute of the sweep →
  `SweepCanvas.reset(new_precomputed)`.

### 5.5 MainWindow state

- `_precomputed[i]` — `list[EqResult]` for primary field `i`.
- `_field_arr[i]` — field values array for primary field `i`.
- `_primary_idx` — which field is the current primary sweep axis.
- `_sweep_canvases` — `dict[int, SweepCanvas]`, lazily created.
- `_field_sliders`, `_field_labels` — one per field.

Mode combo: one entry per field (`"{symbol}-x diagram"`); only shown when
`len(fields) > 1`.

### 5.6 FullGridWorker

Background QThread for full N_T × N_P grid of `EqResult` objects. Uses a
`threading.Event` for pause/resume. On completion: enables the 3-D
visualization button.

### 5.7 Drag state

`_DragState`: `phase_name`, `handle_idx`, `y_press_data`, `snapshot`,
`T_ref`, `P_ref`, `x_press_*`, `drag_axis`. `drag_axis='rigid'` is set at
press for Ctrl+click. `'vertical'` / `'horizontal'` is detected on first
motion for handle drags.

`_phase_orig_x` / `_phase_orig_y`: original x and y data stored at `redraw()`
time for rigid drag reference.

---

## 6. Graphical Builder (`pde_builder.py`)

### 6.1 Data containers

`PhaseData` is a pure-Python (no Qt) data container for one phase, mirroring
`Phase + EnergyModel`:

```python
@dataclass
class PhaseData:
    name:              str
    phase_type:        str
    xmin, xmax:        float
    ideal_gas:         bool
    hs_H, hs_S, hs_V: list         # HS form
    poly:              list         # polynomial form
    vle:               dict | None
    patch_left:        float | None
    patch_right:       float | None
    patch_left_phase:  str
    patch_right_phase: str
```

`SystemData` is the system-level mirror:

```python
@dataclass
class SystemData:
    title:        str
    components:   list[str]
    energy_form:  str
    T_min, T_max, T_initial:  float     # explicit T fields
    has_pressure:             bool
    P_min, P_max, P_initial:  float     # explicit P fields
    R_gas, P_ref, P_unit:     float/str
    phases:                   list[PhaseData]
```

### 6.2 Conversion paths (legacy — pending removal)

> **Note:** These paths exist in the current code but are
> slated for elimination.  D-6 replaces them with the
> Spec layer: `MainWindow` will hold a `SystemSpec`
> directly, the builder will receive and mutate it, and
> `from_system()` will be deleted.

- `SystemData.to_system()` — three-pass builder
  (non-VLE, VLE gas, patched).
- `SystemData.from_system(system)` — reconstructs
  `SystemData` from a live `System` via `isinstance`
  chains on `EnergyModel` subclasses.
- `SystemData.from_xml(xml)` — builds `SystemData`
  directly from XML.
- `SystemData.to_xml_str()` — serialises to XML.

### 6.3 Drag mechanics

`BuilderWindow(QDialog)` is non-modal; emits `system_applied(System)` on
Apply.

- **`apply_handle_drag()`** — 3×3 Vandermonde solve for H₀, H₁, H₂ from
  three handle (x, G) pairs.
- **`apply_xrange_drag()`** — updates xmin/xmax; clamped with 0.02 margin.
- **`apply_rigid_shift(pd, delta_G, delta_x=0.0)`** — vertical: adds
  `delta_G` to `hs_H[0]`. Horizontal: reparameterises H and S polynomials
  via `_shift_poly_coeffs` (p(x) → p(x−dx)), updates xmin/xmax.
- **`_shift_poly_coeffs(coeffs, dx)`** — ascending-order coefficients of
  p(x−dx) via numpy `poly1d` composition; handles arbitrary degree.
- **`BuilderWindow.update_phase_data(name, data)`** — syncs spinboxes after
  a canvas drag.

### 6.4 VLE editor

`PhaseEditorWidget` stack index 2 = VLE page (4 spinboxes: T_bp_A, T_bp_B,
L_A, L_B). Auto-switches when `phase_type='gas'`. `PhaseData.is_vle_gas`
property. `GxCanvas._draw_edit_overlay()` skips handles for phases with
`vle_params` set.

---

## 7. Consistency Checker (`pde_check.py`)

Non-blocking, Qt-free, model-agnostic (works through `Phase.gibbs()`).

| Check                          | What it tests                    |
|--------------------------------|----------------------------------|
| `check_phase_coverage`         | Full [0,1] composition covered   |
| `check_convexity`              | d²G/dx² < 0 (spinodal)          |
| `check_vle_terminal_tangency`  | G and dG/dx equality at T_bp     |
| `check_vle_phase_ordering`     | G_vapor < G_liquid above T_bp    |
| `check_end_member_gmatch`      | End-member crosses host G in T   |
| `check_vle_params_valid`       | L > 0; T_bp within T range       |

`ConsistencyWarning` carries `severity`, `message`, `detail`, `fix_delta`
(single H₀ shift), and `fix_hs` (minimum-norm 4×6 lstsq correction to H and
S jointly).

`run_all_checks()` skips numerical VLE checks for phases built via
`compute_vle_gas_hs()` (constraints are satisfied by construction).

---

## 8. Export (`pde_export.py`)

### 8.1 2-D export (`export_binary_Tx`)

Structured triangulated mesh over (composition, primary field). HDF5 datasets:

- `/mesh/nodes` — `(N_nodes, 3)` as `(x, field_value, 0)`.
- `/mesh/triangles` — `(N_tri, 3)` connectivity.
- `/fields/phase_id`, `/fields/G_min`, `/fields/phase_frac`.
- `/boundaries/…`, `/phases/…`, `/system/…`.

A companion `.xdmf` file makes the dataset directly openable in ParaView.

### 8.2 3-D export (`export_3d_TPx`)

Structured grid over (composition, temperature, pressure). Each dataset is
`(n_P, n_T, n_x)`. A companion `.pv.py` ParaView helper script is also
written alongside the HDF5 file.

---

## 9. 3-D Visualization (`pde_3d.py`)

`PhaseDiagram3D.from_grid(grid, T_arr, P_arr, system)` reorganises the full
T×P grid of `EqResult` objects into two-phase boundary surfaces.

`Viz3DWindow` wraps a PyVista `QtInteractor`. Axis convention:
- x-axis: composition (0 → 1)
- y-axis: pressure (fields[1])
- z-axis: temperature (fields[0])

Pyvista and pyvistaqt are lazy-imported; the rest of the application works
without them.

---

## 10. Design Strengths

Several architectural choices are deliberate and worth preserving
through the remediation phases. Each item below notes its current
status honestly.

**Separation of computation from UI.** `pde_compute.py` has no Qt
dependency. It can be called from scripts, tests, or export code
without starting a GUI. This boundary is clean and should remain so.

**VLE reparameterisation.** `compute_vle_gas_hs` guarantees all four
tangency conditions (G and dG/dx equality at both terminal boiling
points) by construction. The checker correctly skips redundant
numerical checks for VLE-built phases. This eliminates a whole class
of user error and works as designed.

**Consistency checker design.** `pde_check.py` is non-blocking,
Qt-free, and model-agnostic (works through `Phase.gibbs()`). Each
check function is independently callable. `ConsistencyWarning` carries
both a quick single-parameter fix and a minimum-norm joint H/S
correction. This composable design should be preserved.

**Clean XML schema.** The XML input uses a single
`<fields>` schema.  Backward compatibility with prior
XML formats is not required; the project has full
freedom to redesign the input format as needed.

**O(1) reveal animation.** `SweepCanvas` pre-draws all regions and
hides the unrevealed portion behind a white cover rectangle.
`set_cursor()` adjusts the cover height — O(1) regardless of diagram
complexity. This makes the primary slider feel responsive even on
large grids.

**Lower convex hull approach.** Using `scipy.spatial.ConvexHull` for
equilibrium computation is algorithmically sound and generalises to
higher-dimensional composition spaces in principle. The current
region-classification walk assumes a 1-D composition axis, so the
ternary generalisation is untested, but the algorithm choice itself
does not need to change when that work is done.

**Field abstraction (partial).** The `Field` dataclass
cleanly separates the identity of an intensive parameter
from its role in any particular view.  `SystemSpec` uses
`fields: list[FieldSpec]` from day one, resolving the
old `SystemData` gap (issue 12.3).  However, the
abstraction is still incomplete downstream: `EqResult`,
`compute_equilibrium`, `FullGridWorker`, and
`PhaseDiagram3D` all still hardcode temperature and
pressure (see issues 12.4, 12.6, 12.7).  Completing the
abstraction is Phase 1 work.

**Palette infrastructure (partial).** Nine named palettes exist in
`_PALETTES` and `ColorDialog` allows user selection. However,
`_color_map()` always reads the module-level `_COLOR_PALETTE` and
ignores the active selection; `reload_system()` resets to Muted (see
issue 12.9). The infrastructure is in place but the wiring is broken.

---

## 11. Target: Spec Layer

This section describes the target canonical form agreed upon during the
architectural review (see VISION principle 1). It resolves known issues
12.1–12.3 and 12.5 simultaneously.

### 11.1 Motivation

The current system has three parallel construction paths to `System`:
`parse_system()` (XML → runtime), `SystemData.to_system()` (builder →
runtime), and `SystemData.from_system()` (runtime → builder). Adding a new
energy model type requires changes in at least seven places (see issue 12.1).
`PhaseData` is a flat grab-bag whose fields grow with every new model type.
`SystemData` has not migrated to a `fields` list (issue 12.3), silently
dropping any field beyond T and P.

### 11.2 FieldSpec, PhaseSpec, SystemSpec

```python
@dataclass
class FieldSpec:
    name: str; symbol: str; unit: str
    min_val: float; max_val: float; initial_val: float
    extras: dict       # R_gas, P_ref, etc. — keyed by field

@dataclass
class PhaseSpec:
    name: str; phase_type: str
    xmin: float; xmax: float
    model_type:   str  # 'HS'|'polynomial'|'piecewise_patch'|…
    model_params: dict # all model-specific data; keys vary

@dataclass
class SystemSpec:
    title: str; components: list[str]; energy_form: str
    fields: list[FieldSpec]       # fields-first from day one
    phases: list[PhaseSpec]
```

`PhaseSpec` replaces `PhaseData`. Model-specific data lives entirely in
`model_params`; no flat per-model fields. Adding CALPHAD means one new
`model_type` string and new keys in `model_params` — `PhaseSpec` itself does
not change.

### 11.3 make_energy_model and factory dispatch

`PhaseSpec.make_energy_model(specs_by_name, built,
field_specs)` dispatches on `model_type` and constructs
the appropriate `EnergyModel` subclass.

- `specs_by_name` (`dict[str, PhaseSpec]`) — all phase
  specs in the system, for reading dependency-phase
  coefficient data (e.g. liquid H/S for VLE gas).
- `built` (`dict[str, EnergyModel]`) — models already
  constructed during the topological walk, for patch
  targets that need a built model's arrays.
- `field_specs` (`list[FieldSpec]`) — system fields,
  supplying R_gas/P_ref from the pressure field's
  extras and T_ref from the temperature field.

Cross-phase references resolved:

- VLE gas: `model_params['vle_params']['liquid_phase']`
  names the liquid whose H/S feed
  `compute_vle_gas_hs`.
- Patch: `model_params['patch_left_phase']` /
  `'patch_right_phase'` name the target phases for
  G/dG matching at cut points.

A topological sort (trivial for two-level dependency)
ensures dependencies are built before dependents.

### 11.4 Dependency-aware construction

The canonical construction path becomes:

```
raw input (XML, builder, direct construction)
    │
    ▼
SystemSpec + list[PhaseSpec]     ← single authoritative form
    │
    ├── to_xml_str()  → XML file       (serialisation)
    │
    ▼
SystemSpec.to_system()           ← only place EnergyModels built
    │  (topological sort → make_energy_model per phase)
    ▼
System                           ← derived; never round-tripped
```

`parse_system()` becomes a thin XML → SystemSpec translator that then calls
`to_system()` internally. `from_system()` is eliminated entirely. The builder
receives the live `SystemSpec` directly.

### 11.5 Builder migration (resolved)

> **Status: resolved.** D-5, D-6, D-8, and D-9 were
> merged into a single migration.  `PhaseData` and
> `SystemData` are deleted.

`PhaseSpec` exposes permanent convenience properties
(`H_coeffs`, `S_coeffs`, `V_coeffs`, `ideal_gas`,
`poly_coeffs`, `vle_params`, `patch_left_x`, etc.)
that delegate into `model_params`.  These are the
API used by the builder UI, the fitting functions
(`apply_handle_drag`, etc.), and the G(x) canvas
edit overlay.

`BuilderWindow` takes a `SystemSpec` directly.  On
Apply it calls `spec.to_system()` and emits the
`(System, SystemSpec)` pair.  `MainWindow` stores
the `SystemSpec`; no `from_system()` round-trip.

`SystemSpec.to_xml_str()` and `parse_system_spec()`
provide serialisation round-tripping.
`_G_from_phase_spec` replaces the former
`_G_from_phase_data` in the viz layer.

---

## 12. Known Design Issues

Issues 12.1–12.5 are interdependent. They all stem from the builder
maintaining a separate mutable data model that must mirror the runtime model.
Issues 12.6–12.10 are independent and can be addressed in any order. Each
issue notes which remediation phase addresses it (see `TODO.md`).

### 12.1 Dual data model (Phase vs. PhaseData) — Phase 0

> **Status: resolved.** `PhaseData` and `SystemData`
> are deleted.  The builder and viz work directly
> with `PhaseSpec`/`SystemSpec`.  See §11.5.

The original problem: every phase lived in two
separate object graphs (runtime `Phase` + `EnergyModel`
and builder `PhaseData` in `SystemData`).  Three
conversion paths connected them.  Every new energy
model feature required changes in at least seven
places.

**Resolution:** `PhaseSpec` with `model_type` +
`model_params` dict replaces `PhaseData`;
`make_energy_model()` is the single factory.
See §11.2–11.3.

### 12.2 Patch-H computation duplicated — Phase 0

> **Status: resolved.** `compute_left_patch_H` and
> `compute_right_patch_H` now live in `pde_energy.py`
> (D-7).  The `pde_input.py` copies are removed;
> `pde_builder.py` still has its own copies pending
> the D-5/D-6 migration.

The original problem: `_compute_left_patch_H` existed
in both `pde_input.py` and `pde_builder.py` with
different signatures and degenerate-case handling.

**Resolution:** shared helpers in `pde_energy.py`;
the coefficient-list form is canonical.

### 12.3 SystemData not migrated to fields list — Phase 0

> **Status: resolved at the Spec level.** `SystemSpec`
> uses `fields: list[FieldSpec]` (D-1) and the parse
> path builds it correctly (D-4).  `SystemData` in the
> builder still has the flat structure — it will be
> replaced entirely by D-6.

The original problem: `SystemData` had flat
`T_min/max/initial`, `has_pressure`,
`P_min/max/initial`, `R_gas`, `P_ref`, `P_unit`
attributes.  A third field was silently dropped.

**Resolution:** `SystemSpec` uses `fields:
list[FieldSpec]` from day one.  See §11.2.

### 12.4 EqResult / compute_equilibrium — Phase 1

> **Status: resolved.** `EqResult` stores
> `field_values: dict`.  `.T` and `.P` are computed
> properties.  `compute_equilibrium(system,
> field_values)` takes a dict.
> `_extract_field_values` eliminated.
> — D-10, D-11, D-13.

### 12.5 `_G_from_phase_data` duplication — Phase 0

> **Status: resolved.** `_G_from_phase_data` replaced
> by `_G_from_phase_spec` which reads coefficients
> directly from `PhaseSpec` convenience properties.
> The lightweight evaluator is retained for drag
> performance (avoids constructing a temporary
> `EnergyModel` per mouse-move event).

### 12.6 Backward-compat gibbs() shim — Phase 1

> **Status: resolved.** Shim removed.
> `EnergyModel.gibbs(x, field_values)` is the only
> public API.  `Phase.gibbs()`, `pde_check`, and all
> callers updated to pass dicts.  — D-12.

### 12.7 3-D visualization hardcoded to T × P — Phase 1

> **Status: resolved.** `FullGridWorker` takes two
> `Field` objects + value arrays.  `PhaseDiagram3D`
> and `Viz3DWindow` derive axis labels from
> `Field.symbol` and `Field.unit`.  Single-field
> systems already hide the 3-D button.  — D-14, D-15.

### 12.8 pde.py entry point has legacy cruft — Phase 2

Unused imports (`h5py`, `math`, `random`). `ScriptSettings` class reimplements
`argparse` across four methods.

**Resolution:** replace `ScriptSettings` with a plain function using
`argparse.ArgumentParser`. Remove unused imports. Self-contained, zero risk.

### 12.9 Color assignment ignores active palette — Phase 2

> **Status: resolved.** `_init_system_state` preserves
> existing user-chosen colors for phases that survive
> a reload; only new phases get fresh defaults.
> Two-phase color/hatch also preserved.  — C-2.

### 12.10 Unit system — Phase 2

> Resolved.  `<units>` block implemented; R_gas
> validation active.  — C-3.

`Field.unit` is display-only.  Coefficients are
dimensionless floats.  Mismatched units (e.g.
R_gas in J vs. energy in kJ) previously produced
wrong diagrams with no error.

**Resolution:** a `<units>` element inside `<system>`
declares the unit system as a human-readable record
with optional parse-time validation:

```xml
<system>
  ...
  <units energy="kJ/mol" temperature="K"
         pressure="atm"/>
</system>
```

Attributes (`energy`, `temperature`, `pressure`)
are all optional.  When both `energy` and
`temperature` are declared and a pressure field
carries an `R_gas` value, the parser checks R_gas
against a lookup table of known values per unit
pair and issues a `warnings.warn()` if the
relative error exceeds 1%.

Known R_gas values (energy_unit, K):

| Units        | R_gas              |
|--------------|--------------------|
| J/mol, K     | 8.314              |
| kJ/mol, K    | 0.008314           |
| cal/mol, K   | 1.987              |
| kcal/mol, K  | 0.001987           |
| eV/atom, K   | 8.617 x 10^-5      |

CALPHAD systems (`energy_form='calphad'`) default
to SI (`J/mol`, `K`, `Pa`) when no `<units>` block
is present.  If a `<units>` block is present with
non-SI values, the parser warns.

`SystemSpec.units` stores the dict; `to_xml_str()`
emits `<units>` when non-empty.  All 11 demo files
now carry explicit `<units>` blocks.

---

## 13. Multi-Component Generalisation

The current implementation is binary (two components,
scalar composition x). The architecture is designed so
that generalisation to ternary and higher-order systems
does not require algorithmic changes to the equilibrium
engine — only data-structure and visualisation changes.

### 13.1 What is already general

- `System.components` is a plain list; `n_components`
  exists but nothing enforces `len == 2`.
- `scipy.spatial.ConvexHull` (Qhull) works in arbitrary
  dimensions. For N components the hull is computed in
  (N−1 + 1)-dimensional space (composition simplex + G).
- The Gibbs phase rule falls out of the hull geometry:
  at most N coexisting phases at fixed field values.

### 13.2 What needs work (hardwired to scalar x)

- **pde_energy.py:** `polyval(x, coeffs)` is 1-D.
  Ternary+ requires multivariate polynomials (e.g.,
  Redlich–Kister–Muggianu pairwise binary terms plus
  higher-order excess terms). New `EnergyModel`
  subclass(es) and XML schema needed.
- **pde_phase.py:** `xmin`/`xmax` scalar bounds and
  `linspace` composition grid must become simplicial
  meshes (triangular for ternary, tetrahedral for
  quaternary).
- **pde_compute.py:** `ConvexHull` call is fine.
  Lower-hull filter index changes
  (`equations[:, 1] < 0` → `equations[:, N−1] < 0`).
  `_extract_regions()` walks 1-D hull vertices and must
  be rewritten to identify simplex faces for ternary+.
- **pde_input.py:** `composition_range` (scalar
  `xmin`/`xmax`) and energy coefficient XML schema both
  need multivariate equivalents.
- **pde_viz.py:** complete redesign for each
  dimensionality level (Gibbs triangle for ternary,
  linked 2-D slices for higher).

### 13.3 Composition-space geometry by component count

| Components | Composition space    | Hull dim | Max coexisting |
|------------|----------------------|----------|----------------|
| Binary     | line (1-D)           | 2-D      | 2 phases       |
| Ternary    | triangle (2-D)       | 3-D      | 3 phases       |
| Quaternary | tetrahedron (3-D)    | 4-D      | 4 phases       |

### 13.4 Visualisation strategy

For ternary systems, the composition axis becomes a 2-D
Gibbs simplex. Structured grid + masking (Cartesian
sampling, mask outside the simplex) is simplest but wastes
~50% of grid points. Unstructured triangular meshes are
more efficient and were chosen as the target approach for
HDF5 + XDMF export.

For quaternary and beyond, the composition space is 3-D+
and a flat screen is inadequate. VR/volumetric rendering
inside the composition tetrahedron is a natural target.
`vedo` (VTK wrapper with OpenVR support) has a
prototype-level hook in the codebase already.

### 13.5 Composition representation

Binary convention: `x` is a scalar — the mole
fraction of component B (components[1]).
`x_A = 1 - x`.

Multi-component generalisation: composition is an
`(N-1)`-dimensional vector of independent mole
fractions `[x_2, x_3, ..., x_N]`.  The first
component's fraction is implied:
`x_1 = 1 - sum(x_i for i in 2..N)`.

At the API boundary, `x` changes from a 1-D array
of shape `(n_pts,)` to a 2-D array of shape
`(n_pts, N-1)`.  For binary systems `N-1 = 1`, so
a column vector `(n_pts, 1)` is the canonical
form; the scalar convention is a squeeze of this
for backward compatibility.

Key decision: **keep the scalar squeeze for
binary**.  Energy models receive `x` as `(n,)` when
`N = 2` and `(n, N-1)` when `N >= 3`.  This avoids
a mass migration of all binary-model code.  The
`Phase.gibbs()` wrapper handles the reshape:

```python
def gibbs(self, x, field_values):
    x = np.atleast_2d(x)
    if self.n_components == 2:
        x = x.ravel()    # (n,) for binary
    return self.energy_model.gibbs(x, field_values)
```

### 13.6 Energy model generalisation

#### 13.6.1 Native multi-component model

For ternary+ native (non-CALPHAD) systems, the
natural formalism is the Redlich-Kister-Muggianu
(RKM) pairwise expansion:

```
G(x, T) = sum_i x_i * G_i^0(T)           (end-member reference)
         + sum_{i<j} x_i * x_j
           * sum_k L_{ij,k}(T)
           * (x_i - x_j)^k               (binary interaction)
         + sum_{i<j<k} x_i * x_j * x_k
           * L_{ijk}(T)                   (ternary interaction)
```

This is implemented as a new `RKMModel(EnergyModel)`
subclass.  The constructor takes:

```python
class RKMModel(EnergyModel):
    def __init__(self, n_components,
                 ref_G,        # list of T-poly per component
                 binary_L,     # dict[(i,j)] -> list of T-polys per order k
                 ternary_L=None):  # dict[(i,j,k)] -> T-poly
```

`ref_G[i]` is a T-polynomial (ascending order) for
the Gibbs energy of pure component `i`.
`binary_L[(i,j)][k]` is a T-polynomial for the
k-th order Redlich-Kister parameter between
components `i` and `j`.

For binary systems (`n_components == 2`), this
reduces to the familiar sub-regular solution
model:

```
G(x, T) = (1-x)*G_A(T) + x*G_B(T)
         + x*(1-x) * sum_k L_k(T) * (1-2x)^k
```

This overlaps with but does not replace HSModel.
HS remains the simple pedagogical model; RKM is
the thermodynamically rigorous multi-component
extension.

#### 13.6.2 CALPHAD multi-component

`CALPHADModel` already wraps `pycalphad.calculate()`
which handles N-component systems natively.  The
changes needed:

- `components` becomes a list of N elements
  (currently hardcoded as binary `[comp_A, comp_B]`)
- Site-fraction construction generalises from
  2 columns to N columns per mixing sublattice
- The `_gibbs_impl` signature accepts
  `(n_pts, N-1)` composition arrays and maps them
  to site fractions

The internal changes are modest because pycalphad
does the heavy lifting.

### 13.7 Data structure changes

#### pde_phase.py

```python
class Phase:
    # xmin, xmax (scalar) → composition_bounds:
    #   binary: (xmin, xmax)
    #   ternary+: convex polygon on the simplex,
    #     or None for the full simplex
    # is_point → is_point (xmin == xmax for binary;
    #   single-point composition for ternary)
    composition_bounds: ...

    def composition_grid(self, n_points=500):
        # binary: np.linspace(xmin, xmax, n)
        # ternary: triangular mesh on simplex
        # quaternary: tetrahedral mesh
```

For the spec layer, `PhaseSpec.xmin`/`xmax` remain
for binary.  Multi-component phases add a
`composition_region` key to `model_params` (or a
new PhaseSpec attribute).

#### SystemSpec

No structural changes — `components` is already a
list of arbitrary length.  `n_components` is
`len(components)`.

### 13.8 Equilibrium computation

#### 13.8.1 Hull construction

`compute_equilibrium()` builds the convex hull in
(N-1+1)-dimensional space:

```python
# Binary (current):
points = np.column_stack([x, G])       # (n, 2)

# Multi-component:
# x is (n, N-1), G is (n,)
points = np.column_stack([x, G])       # (n, N)
hull = ConvexHull(points)
```

The lower-hull filter generalises:

```python
# Binary: equations[:, 1] < 0
#   (G is the second coordinate)
# N-component: equations[:, N-1] < 0
#   (G is the last coordinate)
lower = hull.equations[:, N-1] < 0
```

#### 13.8.2 Region extraction

This is the hardest algorithmic change.

**Binary (current):** walks sorted hull vertices
left-to-right.  Each vertex belongs to one phase.
Consecutive vertices from different phases define
a two-phase region; same-phase runs define
single-phase regions.

**Ternary:** the lower hull is a 2-D triangulated
surface.  Each simplex (triangle) in the hull uses
vertices from 1, 2, or 3 distinct phases:

- 1 phase: interior of a single-phase region
- 2 phases: edge of a two-phase (tie-line) region
- 3 phases: invariant tie-triangle (three-phase
  equilibrium, the ternary analogue of a eutectic)

Region extraction groups adjacent simplices by
their phase-set signature.  The algorithm:

1. Label each lower-hull simplex with its
   phase-set (frozenset of phase indices).
2. Build an adjacency graph of simplices.
3. Connected components with the same phase-set
   form a region.
4. Region boundaries are extracted from the
   unshared edges of each component.

The output replaces the current flat list of
`{'type': 'single'|'two_phase', 'x0': ...,
'x1': ..., 'phases': ...}` dicts with a richer
structure:

```python
@dataclass
class EqRegion:
    phase_set: frozenset[int]  # phase indices
    vertices: ndarray          # (m, N-1) boundary
    simplices: list[int]       # hull simplex indices
    # type is implicit:
    #   len(phase_set) == 1 → single-phase
    #   len(phase_set) == 2 → two-phase (tie-lines)
    #   len(phase_set) == N → invariant
```

For binary backward compatibility, the 1-D walk
algorithm is preserved when `N == 2`.

### 13.9 XML schema

Multi-component `<components>`:

```xml
<system>
  <components>Al Cu Mg</components>
  <energy_form>calphad</energy_form>
  <tdb>Al-Cu-Mg.tdb</tdb>
</system>
```

For native RKM models, a new energy block format:

```xml
<energy model="rkm">
  <!-- End-member reference G(T) for this phase -->
  <ref component="Al">
    <G a0="0" a1="0"/>
  </ref>
  <ref component="Cu">
    <G a0="5000" a1="-3.2"/>
  </ref>
  <ref component="Mg">
    <G a0="2600" a1="0"/>
  </ref>

  <!-- Binary interaction L_{Al,Cu}(T) -->
  <binary components="Al Cu">
    <L order="0" a0="-12000" a1="3.0"/>
    <L order="1" a0="-1100" a1="0.5"/>
  </binary>
  <binary components="Al Mg">
    <L order="0" a0="-1800" a1="2.0"/>
  </binary>

  <!-- Optional ternary interaction -->
  <ternary components="Al Cu Mg">
    <L a0="500"/>
  </ternary>
</energy>
```

Composition range for ternary phases replaces
scalar `xmin`/`xmax` with a convex polygon (or
is omitted for the full simplex):

```xml
<composition_region>
  <vertex x_Cu="0.0" x_Mg="0.0"/>
  <vertex x_Cu="0.5" x_Mg="0.0"/>
  <vertex x_Cu="0.0" x_Mg="0.5"/>
</composition_region>
```

### 13.10 Implementation roadmap

The multi-component generalisation decomposes into
four ordered implementation steps, matching the
existing TODO items C-8 through C-11:

1. **C-8: RKMModel + XML schema.**  The energy
   model subclass and input format.  Can be tested
   in binary mode first (RKM reduces to sub-regular
   solution) before exercising ternary.

2. **C-9: Simplicial composition grid.**  Replace
   `Phase.composition_grid()` with a simplex mesh
   generator.  `pde_phase.py` changes: new
   `composition_bounds` representation, `is_point`
   generalisation.

3. **C-10: N-D hull region extraction.**  Rewrite
   `_extract_regions()` in `pde_compute.py`.
   The `EqRegion` dataclass replaces the current
   dict-based region format.  Binary backward
   compat via the 1-D walk when `N == 2`.

4. **C-11: Ternary visualisation.**  Gibbs triangle
   canvas in `pde_viz.py` or a new module.  HDF5
   export with unstructured triangular mesh.

Each step is independently testable.  C-8 can be
validated against known binary sub-regular solution
results.  C-9+C-10 can be validated against
pycalphad's own equilibrium solver for ternary
CALPHAD systems.  C-11 is purely visual.

**CALPHAD ternary** is the easiest path to a
working ternary diagram because CALPHADModel
already delegates to pycalphad, which handles
N-component systems.  The implementation order for
a ternary milestone is: C-9 → C-10 → C-11 → C-8,
deferring the native RKM model until the grid,
hull, and visualisation infrastructure exist.

---

## 14. ParaView Integration Architecture

The export path produces four files per phase diagram:

| File              | Purpose                          |
|-------------------|----------------------------------|
| `*.hdf5`          | Precomputed equilibrium data     |
| `*.xdmf`          | XDMF wrapper for ParaView       |
| `*.pv.py`         | Scene setup script (loader)      |
| `*.pv.ctrl.py`    | Python plugin (controls panel)   |

### 14.1 Two-speed workflow

Interactive exploration uses PyVista embedded in the
PySide6 application (tight loop: slider → recompute →
update VTK mesh, all in-process). Batch/high-resolution
analysis uses ParaView with the precomputed HDF5 files.
Both paths consume the same HDF5 schema, which is the
critical shared layer.

```
┌─────────────────────────────┐
│   PDE compute engine         │
│   (pde_compute + energy)     │
└──────┬──────────┬────────────┘
       │          │
  live (in-process)  batch (disk)
       │          │
┌──────▼──────┐  ┌▼──────────────┐
│ PyVista/Qt  │  │ HDF5 + XDMF   │
│ (explore)   │  │ (archival)     │
└─────────────┘  └──────┬─────────┘
                        │
                 ┌──────▼──────┐
                 │  ParaView   │
                 │  (analysis) │
                 └─────────────┘
```

### 14.2 HDF5 schema

At each grid point (composition × field values), the hull
is precomputed and stored as integer phase labels plus
boundary compositions. ParaView renders labeled regions
and boundary surfaces directly. Datasets:

- `/mesh/nodes`, `/mesh/triangles` — geometry
- `/fields/phase_id`, `/fields/G_min`,
  `/fields/phase_frac` — per-node equilibrium data
- `/boundaries/…`, `/phases/…`, `/system/…` — metadata

### 14.3 ParaView plugin mechanism

The `.pv.ctrl.py` plugin uses `VTKPythonAlgorithmBase`
with `@smproperty` / `@smdomain` decorators to create
native Properties-panel controls (opacity slider,
colour-by dropdown, per-phase visibility checkboxes)
without requiring Python Qt bindings. Key compatibility
lessons from ParaView 6.0 development:

- ParaView 6.0 does not ship PySide6/PySide2. Use
  `paraview.qt` for Qt access, or the `@smproperty`
  decorator system for native controls.
- Import `VTKPythonAlgorithmBase` from
  `vtkmodules.util.vtkAlgorithm`, not from
  `paraview.util.vtkAlgorithm`.
- Inside `RequestData`, use `GetDisplayProperties()`
  instead of `Show()` and never call `Render()` — both
  cause re-entrant pipeline updates that crash ParaView.
- A source-based plugin (`nInputPorts=0`) avoids the
  need for a dummy input selection in the pipeline.
- `PropertyGroup` via `@smproperty.xml` class decorator
  does not work in ParaView 6.0; use flat properties.

### 14.4 Session save/load (future)

A saved session file should contain the system definition
(reproduced as XML), the precomputed grid (G values, phase
indices, region boundaries at each field-value grid
point), and metadata (PDE version, timestamp, field ranges
and resolutions). Loading a saved session skips all
computation and opens the visualisation directly from the
stored grid. This enables distributing completed
calculations to students and caching expensive
multi-component or CALPHAD computations.

---

## 15. XML Coupling-Term Schema (future)

When new field types are added beyond T and P, the XML
schema will need explicit coupling-term elements. The
existing HS shorthand (`<H>`, `<S>`, `<V>` tags) is kept
as a recognised abbreviation. New systems can use explicit
coupling terms:

```xml
<energy>
  <base>
    <poly x0="0.5" x1="-1.0" x2="1.0"/>
  </base>
  <coupling field="temperature" type="linear">
    <response x0="-0.005"/>
  </coupling>
  <coupling field="pressure" type="linear">
    <response x0="0.001"/>
  </coupling>
  <coupling fields="temperature pressure"
           type="ideal_gas" P_ref="1.0">
    <response x0="8.314e-3"/>
  </coupling>
</energy>
```

A magnetic-field example:

```xml
<fields>
  <field name="temperature" symbol="T"
         unit="K" min="0" max="1000"
         initial="300"/>
  <field name="magnetic_field" symbol="H"
         unit="T" min="0" max="10"
         initial="0"/>
</fields>

<!-- inside a phase: -->
<coupling field="magnetic_field" type="linear">
  <response x0="-0.002" x1="0.001"/>
</coupling>
```

---

## 16. CALPHAD Integration

CALPHAD databases (TDB files) provide assessed
thermodynamic data for multicomponent systems using
Redlich-Kister polynomials for excess Gibbs energy,
sublattice models for ordered phases, and magnetic
contributions (Hillert-Jarl model near the Curie
temperature — a non-polynomial coupling term that is
a concrete example of the `'custom'` coupling type
in §2.5).

### 16.1 Architecture

The integration uses `pycalphad` (open source) for
TDB parsing and single-phase G evaluation.  The key
design decisions:

1. **TDB at system level.**  The `<system>` block
   carries a `<tdb>` element with the file path.
   `SystemSpec.to_system()` loads the `Database`
   once and shares it across all phase constructors.

2. **Components = element names.**  When
   `energy_form='calphad'`, the `<components>`
   values are real element symbols (e.g. `Al Mg`).
   The parser uppercases them for pycalphad and
   each `CALPHADModel` appends `VA` internally.

3. **All-CALPHAD systems.**  `energy_form='calphad'`
   means every phase comes from the same TDB.
   Mixing CALPHAD and native phases within one
   system is not supported in v1.

4. **SI units.**  pycalphad uses J/mol, K, Pa.
   When `energy_form='calphad'`, all field ranges
   and display values are SI.

5. **Lazy import.**  `pycalphad` is imported only
   inside `SystemSpec.to_system()` and
   `CALPHADModel.__init__`.  The application works
   without it installed; loading a CALPHAD input
   file triggers the import (with a clear error
   if missing).

### 16.2 CALPHADModel

```python
class CALPHADModel(EnergyModel):
    """G(x, T, P) from pycalphad Database + phase.

    Wraps pc.calculate() for single-phase Gibbs
    energy evaluation.  Handles sublattice models,
    Redlich-Kister mixing, and magnetic contributions
    via the pycalphad internals.
    """
    def __init__(self, db, phase_name,
                 components, P_default=101325.0):
        # db: pycalphad.Database (shared)
        # phase_name: 'LIQUID', 'FCC_A1', etc.
        # components: ['AL', 'MG'] (no VA)

    def _gibbs_impl(self, x, field_values):
        # x is mole fraction of components[-1]
        # Build (n, 2) points array
        # Call pc.calculate(db, comps+['VA'],
        #     phase_name, T=T, P=P, points=pts)
        # Return GM.values.flatten()
```

### 16.3 XML Schema

```xml
<system>
  <components>Al Mg</components>
  <energy_form>calphad</energy_form>
  <tdb>Al-Mg.tdb</tdb>
</system>

<phase name="liquid" type="liquid">
  <energy model="calphad" phase="LIQUID"/>
</phase>
```

The `phase` attribute on `<energy>` is the TDB
phase identifier (e.g. `LIQUID`, `FCC_A1`,
`HCP_A3`).  The TDB path is resolved relative to
the input XML file's directory.

### 16.4 PhaseSpec model_params

```
'calphad':
    calphad_phase : str        — TDB phase name
    components    : list[str]  — element names
```

No cross-phase dependencies; build order is
independent.  The `couplings` property returns
an empty list (CALPHAD G(x,T,P) is opaque to
the coupling inspector).

### 16.5 Limitations (v1)

- Binary systems only (PDE limitation, not
  pycalphad).
- No mixing of CALPHAD and native phases.
- No automatic unit conversion; SI is required.
- No sublattice-aware composition mapping;
  x is a simple mole fraction.
- `couplings` returns `[]` — the 3-D and export
  modules work but field-coupling inspection
  is unavailable.
