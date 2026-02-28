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
`[Fixed P | Fixed T]` (if pressure) — `Reveal all` — `Colors…` — `[Pre-compute | Progress | Status]` (if pressure).
Below the canvas: T slider row; then P slider row (when `has_pressure`).

**Controls:**
- **Mode selector** (only when `system.has_pressure`): "Fixed P (T-x)" / "Fixed T (P-x)" radio buttons swap the right canvas and reassign primary/secondary roles to the sliders.
- **T slider**: integer kelvin ticks; always visible.
- **P slider**: shown only when `system.has_pressure`; maps 0…N_P_STEPS-1 ticks linearly to P_min…P_max. Endpoint labels and live label include `system.P_unit` (e.g. "0.5 atm" / "5 atm" / "P = 2 atm"); bare numbers when `P_unit=''`.
- **"Reveal all" checkbox**: when checked, hides the cover rectangle on both canvases so the full phase diagram is visible regardless of slider position. Unchecking restores the cover to the current slider position. State is preserved across diagram regens triggered by the secondary slider.
- **"Pre-compute full T-P-x" button** (only when `system.has_pressure`): toggles background computation of the full N_T_STEPS × N_P_STEPS grid. Label cycles: "Pre-compute full T-P-x" → "Pause Computation" → "Restart Computation" → (repeats) → "Full T-P-x cached" (disabled when done). Progress bar and status label update via signals. Once cached, moving the secondary slider is instantaneous (index lookup instead of recompute). While paused or in-progress, the secondary slider falls back to on-demand recompute (unchanged from no-grid behavior).
- **"Colors…" button**: opens `ColorDialog` — a non-modal dialog for per-phase color swatches, palette presets (`_PALETTES`), two-phase region color, and hatch style (`_HATCH_OPTIONS`). Changes apply live to all canvases. The palette dropdown defaults to `'Muted'`.

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
- `MainWindow.closeEvent` calls `abort()` + `wait()` before window teardown.

**Slider interaction logic:**
- Primary slider (`valueChanged`) → fast O(1) update: nearest pre-computed result looked up by index.
- Secondary slider (`sliderReleased`) → recomputes the opposite sweep at the new secondary value (or performs an instant index lookup if the full grid is cached). The cover is initialized at the current primary slider position, not reset to `T_initial`/`P_initial`.
- `actionTriggered` signal connected to `_on_T_action` / `_on_P_action`: handles discrete slider actions (bar-click, arrow keys, Home/End) that do not fire `sliderReleased`. Uses `QTimer.singleShot(0, ...)` to defer into the event loop so `valueChanged` has already updated the slider value before the released handler reads it. `isSliderDown()` guards against double-firing during thumb drags.
- Mode switch → cover is initialized at the current primary slider position.

**`MainWindow` refactor for builder support:**
- `__init__` delegates to `_init_system_state()` (sets all non-widget state) and `_build_central_widget()` (creates all Qt widgets/layouts/signals; calls `setCentralWidget()` so it can be safely called again on reload).
- `reload_system(system)` — public slot called by the builder on Apply: aborts any running worker, closes the color dialog, pre-computes a new T-x diagram, calls `_init_system_state()` + `_build_central_widget()`.
- `_open_builder()` — opens (or raises) the builder window; creates a new `BuilderWindow` each time the old one is not visible; connects its `system_applied` signal to `reload_system`.
- **"Builder…" button** is always present in the top row (before the final stretch).
- `launch_ui_empty()` / `_make_default_system()` — entry point when no input file is found; creates a single-liquid-phase default system and auto-opens the builder.

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
- `CoeffRowWidget` — labelled row of float spinboxes with dynamic +/− buttons; `get_coeffs()`/`set_coeffs()`. Optional `coeff_name` parameter (e.g. `'H'`) adds Unicode subscript headers (H₀, H₁, H₂, …) above each spinbox using a two-row `QGridLayout` (row 0 = subscript labels, row 1 = spinboxes).
- `PolyPhaseCoeffWidget` — 2D grid for polynomial coefficients (x-power rows × T-power columns); rows and columns dynamically resizable.
- `PhaseEditorWidget(QFrame)` — one phase editor: header row (name, type, x range, ideal_gas, remove button) + QStackedWidget (HS page: H/S/V rows with subscript headers and italic polynomial hint label; poly page: PolyPhaseCoeffWidget); `get_phase_data()`/`set_phase_data()`/`set_energy_form()`.
- `BuilderWindow(QDialog)` — top-level builder: system group (title, components, form), temperature group, pressure group, scroll area of `PhaseEditorWidget`s, Load XML / Save XML / Apply / Close buttons.

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
`pde.1.py` additionally requires `vedo`.
