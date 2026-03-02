# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PDE** (Phase Diagram Explorer) is a Python tool for defining phase energy curves and visualizing the resulting phase diagrams. The intended output format is HDF5 + XDMF.

## Environment Setup

Source `.pde/pderc` (bash) before building or running to set required environment variables:

```bash
source .pde/pderc
```

This sets:
- `PDE_DIR` — repo root / install prefix (e.g. `$HOME/CPG/cpg-repo/pde`)
- `PDE_RC` — directory containing `pderc.py` (e.g. `$PDE_DIR/.pde`)
- `PDE_BIN` — install destination for scripts (`$PDE_DIR/bin`)
- Adds `PDE_BIN` to `PATH`

## Build and Install

CMake installs the Python scripts only — no compilation.

```bash
cd build/release && cmake ../..
make install   # installs pde.py and companion modules to $PDE_DIR/bin/
```

## Running the Script

```bash
pde.py                        # reads pde.in.xml, opens interactive UI
pde.py -i myinput.xml -o myoutput
```

Each run appends the command and timestamp to a local `command` file.

## Configuration

`pderc.py` (a Python module) provides default parameters via `parameters_and_defaults() -> dict`:
- `infile`: `pde.in.xml`
- `outfile`: `pde.out`

Load order: `$PDE_RC/pderc.py` is loaded first; a `pderc.py` in the current working directory takes precedence. `.pde/pderc.py` in this repo is the reference template.

## Architecture

```
src/scripts/pde.py          Main script (installed to $PDE_DIR/bin/)
src/scripts/pde_input.py    XML input parser → System object
src/scripts/pde_energy.py   Energy model classes (HSModel, PolyModel)
src/scripts/pde_phase.py    Phase and System data structures
src/scripts/pde_compute.py  Equilibrium computation (convex hull) → EqResult
src/scripts/pde_viz.py      PySide6 + matplotlib interactive UI
src/scripts/pde_builder.py  Graphical input builder (QDialog) + data model
src/scripts/pde_3d.py       3D T-P-x visualization (PhaseDiagram3D + Viz3DWindow)
src/scripts/pde.1.py        Standalone vedo + PySide6 prototype (not installed)
.pde/pderc                  Bash environment setup script
.pde/pderc.py               Python resource control file template
jobs/demo/                  Demo input files (see Demo Jobs below)
```

### Pipeline

```
XML input
  └─ pde_input.parse_system()  →  System
       └─ pde_compute.compute_equilibrium(system, T, P=0.0)  →  EqResult
            └─ pde_viz.launch_ui(system)  →  PySide6 window
```

### `pde.py` internals

- `ScriptSettings.__init__()` — loads `pderc.py`, parses CLI args (`-i`/`-o`), reconciles them, logs invocation to `command`.
- `ScriptSettings.read_input_file()` — delegates to `pde_input.parse_system()`; stores the result as `self.system`.
- `main()` — if input file exists, calls `settings.read_input_file()` then `start_program(settings)`; if the default `pde.in.xml` is absent, calls `pde_viz.launch_ui_empty()` which opens the main window with a default system and auto-opens the builder; if an explicit `-i` file is missing, prints an error and exits.
- `start_program()` — calls `pde_viz.launch_ui(settings.system)`.

### `pde_input.py`

Parses the XML input file using `lxml.etree`. The system-level `<energy_form>` tag (`'HS'` or `'polynomial'`) selects which energy model parser is used for all phases; mixing forms within one file is not supported. Returns a `System` object.

Optional XML elements parsed:
- `<title>` — optional top-level element; sets the window title bar text. If absent, the title is derived from the input filename with path and extensions (`.xml`, `.in`) stripped (e.g. `pde.in.xml` → `pde`, `isomorphous.xml` → `isomorphous`).
- `<pressure>` block — sets pressure fields on the System (see `pde_phase.py`).
  - `<unit>` child (e.g. `<unit>atm</unit>`) — optional pressure unit string displayed in slider labels; omitting it leaves labels as bare numbers.
- `<V x0=... x1=.../>` inside `<energy>` — optional molar volume polynomial for the PV term.
- `ideal_gas="true"` attribute on `<phase>` — enables the ideal-gas chemical potential term (see `pde_energy.py`).

### `pde_energy.py`

Abstract base `EnergyModel` with two concrete subclasses. All models accept an optional `P` argument (default `0.0`; backward-compatible with files that have no pressure block):

- **`HSModel`**: `G(x,T,P) = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P°)]`
- **`PolyModel`**: `G(x,T,P) = Σᵢ cᵢ(T)·xⁱ [+ P·V(x)] [+ R·T·ln(P/P°)]`

Both use ascending-order coefficient conventions: `[c0, c1, c2, ...]` means `c0 + c1·x + c2·x² + ...`

Pressure terms (mutually exclusive for a pure gas — combining both double-counts):
- `V_coeffs` present → adds `P·V(x)` (Poynting correction for condensed phases).
- `ideal_gas=True` → adds `R_gas·T·ln(P/P_ref)` (ideal-gas chemical potential).
- `R_gas`, `P_ref` are passed down from the system-level `<pressure>` block by the parser.

### `pde_phase.py`

- **`Phase`** — name, phase_type (`'gas'`, `'liquid'`, `'solid'`, `'end_member'`), energy model, xmin/xmax. Key methods: `gibbs(x, T, P=0.0)`, `composition_grid(n_points=500)`, `is_point` (property, True when xmin==xmax).
- **`System`** — components list, all phases, energy_form, T_min/T_max/T_initial, plus pressure fields: `has_pressure`, `P_min`, `P_max`, `P_initial`, `R_gas`, `P_ref`, `P_unit` (str, `''` if unspecified), and `title` (str, `''` if unset — viz derives title from filename in that case). Convenience properties: `gas_phases`, `liquid_phases`, `solid_phases`, `end_members`.

### `pde_compute.py`

`compute_equilibrium(system, T, P=0.0, n_points=500)` evaluates G(x,T,P) for every phase, then computes the lower convex hull of the combined (x, G) point cloud via `scipy.spatial.ConvexHull`. Consecutive hull vertices from the same phase form a single-phase region; from different phases, a two-phase region. Returns an `EqResult`.

`EqResult` fields: `T`, `P`, `phase_curves`, `hull_x`, `hull_G`, `hull_phase_idx`, `regions`, plus properties `two_phase_regions` and `single_phase_regions`.

### `pde_viz.py`

Interactive PySide6 + matplotlib window with full pressure support:

**Pre-computation** — at startup, `launch_ui(system)` calls:
- `precompute_Tx_diagram(system, P_initial)` — 200 T steps (T_max → T_min) at fixed P.
- The P-x diagram is **not** pre-computed at startup; it is computed lazily the first time the user switches to "Fixed T (P-x)" mode (at the current T slider value).
- `precompute_diagram` is kept as a backward-compatible alias for `precompute_Tx_diagram`.

**Canvases:**
- **`GxCanvas`** (left): G(x) curves per phase + lower convex envelope + common tangent lines. Redrawn on every primary slider move.
- **`TxCanvas`** (right, Fixed P mode): T-x phase diagram revealed incrementally top-down; white cover rectangle shrinks as T slider moves down (O(1) updates).
- **`PxCanvas`** (right, Fixed T mode): P-x phase diagram, symmetric to `TxCanvas` but revealed bottom-up as P slider moves down. Created lazily on first mode switch.

**Layout:**
All controls sit in a single **top row** above the canvas area:
`Builder…` — `3D View…` (if pressure) — `[Fixed P | Fixed T]` (if pressure) — `Reveal all` — `Colors…` — `[Pre-compute | Progress | Status]` (if pressure).
Below the canvas: T slider row; then P slider row (when `has_pressure`).

**Controls:**
- **Mode selector** (only when `system.has_pressure`): "Fixed P (T-x)" / "Fixed T (P-x)" radio buttons swap the right canvas and reassign primary/secondary roles to the sliders.
- **T slider**: integer kelvin ticks; always visible.
- **P slider**: shown only when `system.has_pressure`; maps 0…N_P_STEPS-1 ticks linearly to P_min…P_max. Endpoint labels and live label include `system.P_unit` (e.g. "0.5 atm" / "5 atm" / "P = 2 atm"); bare numbers when `P_unit=''`.
- **"Reveal all" checkbox**: when checked, hides the cover rectangle on both canvases so the full phase diagram is visible regardless of slider position. Unchecking restores the cover to the current slider position. State is preserved across diagram regens triggered by the secondary slider.
- **"Pre-compute full T-P-x" button** (only when `system.has_pressure`): toggles background computation of the full N_T_STEPS × N_P_STEPS grid. Label cycles: "Pre-compute full T-P-x" → "Pause Computation" → "Restart Computation" → (repeats) → "Full T-P-x cached" (disabled when done). Progress bar and status label update via signals. Once cached, moving the secondary slider is instantaneous (index lookup instead of recompute). While paused or in-progress, the secondary slider falls back to on-demand recompute (unchanged from no-grid behavior).
- **"Colors…" button**: opens `ColorDialog` — a non-modal dialog for per-phase color swatches, palette presets (`_PALETTES`), two-phase region color, and hatch style (`_HATCH_OPTIONS`). Changes apply live to all canvases. The palette dropdown defaults to `'Muted'`.
- **"3D View…" button** (only when `system.has_pressure`): disabled until the full T-P-x grid is cached; clicking opens `Viz3DWindow` (non-modal). Becomes disabled again when resolution changes (grid cleared) or `reload_system()` fires (grid cleared + old window closed). Raising the button when the window is already visible brings it to front.

**Legend positioning:**
- Default location is `'upper left'` on all three canvases.
- Right-clicking any canvas shows a context menu ("Legend position") listing all nine standard matplotlib locations; the current selection is checked. Choosing one moves the legend immediately. Each canvas stores its own `_legend_loc`; `GxCanvas` rerenders via `redraw()`, `TxCanvas`/`PxCanvas` call `legend.set_loc()` directly.

**Two-phase region rendering:**
- Drawn as one thin `broken_barh` strip per T/P step with `linewidth=0` to suppress visible bar-boundary lines.
- When hatch is active: `edgecolors='black'` (hatch lines are drawn in the edge color by matplotlib; hatch line width comes from `rcParams['hatch.linewidth']`, not the patch `linewidth`).
- When hatch is `''` (None): `edgecolors='none'` for a clean solid fill.
- Legend swatch mirrors the same `edgecolor`/`linewidth` logic.

**`FullGridWorker` internals:**
- `threading.Event` (`_run_event`, initially set) + `_abort` flag.
- `run()` calls `_run_event.wait()` then checks `_abort` before each `compute_equilibrium` call — blocks immediately on pause, negligible overhead when running.
- `pause()` clears the event; `resume()` sets it; `abort()` sets `_abort=True` then sets the event (unblocks a paused worker).
- `MainWindow._worker_state`: `'idle'` → `'running'` → `'paused'` ↔ `'running'` → `'done'`.
- `MainWindow.closeEvent` calls `abort()` + `wait()` before window teardown; also closes `_builder` and `_viz3d_window` if open.

**Slider interaction logic:**
- Primary slider (`valueChanged`) → fast O(1) update: nearest pre-computed result looked up by index.
- Secondary slider (`sliderReleased`) → recomputes the opposite sweep at the new secondary value (or performs an instant index lookup if the full grid is cached). The cover is initialized at the current primary slider position, not reset to `T_initial`/`P_initial`.
- `actionTriggered` signal connected to `_on_T_action` / `_on_P_action`: handles discrete slider actions (bar-click, arrow keys, Home/End) that do not fire `sliderReleased`. Uses `QTimer.singleShot(0, ...)` to defer into the event loop so `valueChanged` has already updated the slider value before the released handler reads it. `isSliderDown()` guards against double-firing during thumb drags.
- Mode switch → cover is initialized at the current primary slider position.

**`MainWindow` refactor for builder support:**
- `__init__` delegates to `_init_system_state()` (sets all non-widget state) and `_build_central_widget()` (creates all Qt widgets/layouts/signals; calls `setCentralWidget()` so it can be safely called again on reload).
- `reload_system(system)` — public slot called by the builder on Apply: aborts any running worker, closes the color dialog, closes `_viz3d_window` if open, pre-computes a new T-x diagram, calls `_init_system_state()` + `_build_central_widget()`; re-activates handle-drag edit mode on the fresh canvas if the builder is still visible.
- `_open_builder()` — opens (or raises) the builder window; creates a new `BuilderWindow` each time the old one is not visible; connects its `system_applied` signal to `reload_system`; activates `GxCanvas` handle-drag edit mode and wires `phase_edited` → `_on_phase_edited`.
- `_on_builder_closed()` — deactivates `GxCanvas` edit mode and disconnects `phase_edited` from `_on_phase_edited` when the builder window closes.
- `_on_phase_edited(name, new_pd)` — calls `BuilderWindow.update_phase_data()` to sync spinboxes, then builds a temporary `System` with the edited phase and recomputes equilibrium at the current T/P, then redraws `GxCanvas` (live hull update after every drag without waiting for Apply).
- **"Builder…" button** is always present in the top row (before the final stretch).
- `launch_ui_empty()` / `_make_default_system()` — entry point when no input file is found; creates a single-liquid-phase default system and auto-opens the builder.

**Interactive G(x) handle-drag editing:**
- `_G_from_phase_data(pd, x, T, P=0.0, R_gas=0.0, P_ref=1.0)` — module-level helper; evaluates full `G = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P₀)]` from a live `PhaseData` object; mirrors `HSModel.gibbs()`; pressure params default to zero for backward compatibility.
- `_DragState` — `@dataclass`: `phase_name`, `handle_idx` (0/1/2 = left/mid/right), `y_press_data`, `snapshot` (Phase-5 undo), `T_ref`/`P_ref` (Phase-8); plus `x_press_data`, `x_press_px`, `y_press_px`, `drag_axis` (None/'vertical'/'horizontal') for Phase-3 direction detection.
- `GxCanvas` extended:
  - `phase_edited = Signal(str, object)` class attribute.
  - `set_edit_mode(mode, live_phase_data=None)` — `'off'` deactivates and disconnects mpl event handlers; `'handles'` activates; initialises `_live_phase_data` from `SystemData.from_system()` if not supplied; guards double-connection with `if self._press_cid is None`.
  - `redraw()` overrides G curves from `_live_phase_data` (with full pressure terms) when `_edit_mode != 'off'`; calls `_draw_edit_overlay()` after drawing.
  - `_draw_edit_overlay(result)` — draws 3 diamond handle artists per HS phase at xmin, xmid, xmax (zorder=10); handle G positions computed with full pressure terms; fills `_handle_info` dict.
  - `_on_press()` — hit-tests handles within 12 px; creates `_DragState` including pixel coordinates for direction detection.
  - `_on_motion()` — determines drag direction on first motion (≥3 px threshold; endpoint handles only go horizontal); **vertical**: calls `apply_handle_drag()` live (3-point quadratic H fit, curvature updates immediately); **horizontal**: extends/contracts the phase line and moves endpoint + midpoint handles.
  - `_on_release()` — routes to `apply_xrange_drag()` (horizontal) or `apply_handle_drag()` (vertical); updates `_live_phase_data`; emits `phase_edited`.
- Edit mode is only active for HS-form systems (PolyModel phases render handles but fitting functions return unchanged `PhaseData`).

### `pde_3d.py`

3-D T-P-x phase diagram visualization. New dependencies: `pyvista`, `pyvistaqt` (lazy-imported in `Viz3DWindow.__init__` so the rest of the app works when they are absent).

**`PhaseDiagram3D`** dataclass — `T_arr`/`P_arr` (ascending), `system`, `two_phase_surfaces` (list of dicts with `'label'`, `'phases'`, `'x_left'`, `'x_right'`):
- `from_grid(grid, T_arr, P_arr, system)` — walks `grid[i_T][i_P]` (descending, from `FullGridWorker`); remaps to ascending via `j_T = N_T-1-i_T`; fills per-region NaN arrays keyed by phase-index tuple.
- `to_pyvista_surfaces()` — returns `list[pyvista.StructuredGrid]`, two per two-phase region (x_left, x_right); axis: x=composition, y=T (K), z=P; `dimensions=[N_P, N_T, 1]`.
- `to_pyvista_volume()` / `to_xdmf()` — stubs; raise `NotImplementedError`.

**`Viz3DWindow(QMainWindow)`** — non-modal:
- Top row: per-region visibility checkboxes, opacity slider (0–100, default 60), disabled Export… stub, Close.
- `_actors` dict: key `f"{label}|{side}"` → VTK actor; `SetVisibility()` / `GetProperty().SetOpacity()`.
- `set_scale(xscale=T_span, yscale=1.0, zscale=T_span/P_span)` normalizes the three axes to equal visual length (composition ∈ [0,1], T in K, P in user units differ wildly otherwise).
- `show_bounds()` uses `xtitle`/`ytitle`/`ztitle` keyword arguments (pyvista API).
- `closeEvent` calls `self._plotter.close()` to release VTK resources.

### `pde_builder.py`

Graphical input builder. Non-modal `QDialog`; emits `system_applied(System)` on Apply.

**Data model (pure Python, no Qt):**
- `PhaseData` — one phase: name, phase_type, xmin/xmax, ideal_gas, hs_H/S/V lists, poly list.
- `SystemData` — full system fields (mirrors `System` + phase list). Methods:
  - `to_system() → System` — build live objects using `pde_energy`/`pde_phase`.
  - `to_xml_str() → str` — serialize to lxml XML parseable by `pde_input`.
  - `from_system(cls, system)` — round-trip from live System.
  - `from_xml(cls, path)` — delegate to `pde_input.parse_system()` then `from_system()`.

**UI widgets:**
- `_FloatSpinBox` — QDoubleSpinBox ±1e9, 6 decimals, step 0.001.
- `CoeffRowWidget` — labelled row of float spinboxes with dynamic +/− buttons; `get_coeffs()`/`set_coeffs()`. Optional `coeff_name` parameter (e.g. `'H'`) adds Unicode subscript headers (H₀, H₁, H₂, …) above each spinbox using a two-row `QGridLayout` (row 0 = subscript labels, row 1 = spinboxes). Label and +/− buttons are bottom-aligned (`Qt.AlignBottom`) to sit flush with the spinboxes.
- `PolyPhaseCoeffWidget` — 2D grid for polynomial coefficients (x-power rows × T-power columns); rows and columns dynamically resizable.
- `PhaseEditorWidget(QFrame)` — one phase editor: header row (name, type, x range, Ideal gas checkbox, remove button) + QStackedWidget:
  - HS page: H/S/V rows with subscript headers + dynamic hint label (`_hint_lbl`) updated by `_update_eq_label()` — shows `G(x,T) = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P₀)]` and updates live when the V-enable or Ideal-gas checkbox toggles.
  - Poly page: static hint label (`G(x,T) = c₀(T) + c₁(T)·x + …`) + `PolyPhaseCoeffWidget`.
  - `get_phase_data()`/`set_phase_data()`/`set_energy_form()`.
- `BuilderWindow(QDialog)` — top-level builder: system group (title, components, form), temperature group, pressure group, scroll area of `PhaseEditorWidget`s, Load XML / Save XML / Apply / Close buttons.

**Canvas ↔ builder sync:**
- `apply_handle_drag(phase_data, drag_handle_idx, handles_x, handles_G, T, energy_form, P=0.0, R_gas=0.0, P_ref=1.0) → PhaseData` — Phase-4: 3×3 Vandermonde solve for H₀, H₁, H₂ so that `G(xᵢ) = handles_G[i]` at all three handle positions; correctly inverts full G including pressure terms (`H = G + T·S − P·V − R·T·ln(P/P₀)`); falls back to uniform H₀ shift on `LinAlgError`. Phase-8 TODO: two-temperature H+S decomposition. Returns unchanged `PhaseData` for non-HS forms.
- `apply_xrange_drag(phase_data, handle_idx, new_x) → PhaseData` — Phase-3: deep copy with updated `xmin` (handle 0) or `xmax` (handle 2); clamped to [0, xmax−0.02] or [xmin+0.02, 1].
- `BuilderWindow.update_phase_data(name, data)` — walks `_phase_editors` to find the editor whose name matches, then calls `set_phase_data(data)` on it; called by `MainWindow._on_phase_edited()` to keep spinbox values in sync after a canvas drag.

### `pde.1.py`

A minimal standalone prototype that opens a `vedo` window with a PySide6 color-picker button. Not wired into the main pipeline.

## Demo Jobs

All in `jobs/demo/`:

| File | Description |
|------|-------------|
| `pde.in.xml` | Symmetric eutectic binary (HS form) |
| `pde-lg.in.xml` | Liquid-gas (HS form) |
| `azeotrope.in.xml` | Minimum boiling azeotrope (HS form) |
| `vle-pressure.in.xml` | Vapor-liquid equilibrium with variable pressure (P=0.5–5 atm) |
| `isomorphous.xml` | Isomorphous (complete solid solution) system |
| `asymmetric-eutectic.xml` | Asymmetric eutectic binary |
| `eutectic-with-compound.xml` | Eutectic with intermetallic compound |
| `eutectoid.xml` | Eutectoid transformation |
| `polynomial-asymmetric.xml` | Asymmetric system using polynomial energy form |

## Dependencies

`lxml`, `numpy`, `h5py`, `scipy`, `PySide6`, `matplotlib` — plus `argparse`, `abc` (stdlib).
`pde_3d.py` additionally requires `pyvista` and `pyvistaqt` (lazy-imported; app works without them).
`pde.1.py` additionally requires `vedo`.
