# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PDE** (Phase Diagram Energy) is a Python tool for defining phase energy curves and visualizing the resulting phase diagrams. The intended output format is HDF5 + XDMF.

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
- `main()` — calls `settings.read_input_file()` then `start_program(settings)`.
- `start_program()` — calls `pde_viz.launch_ui(settings.system)`.

### `pde_input.py`

Parses the XML input file using `lxml.etree`. The system-level `<energy_form>` tag (`'HS'` or `'polynomial'`) selects which energy model parser is used for all phases; mixing forms within one file is not supported. Returns a `System` object.

Optional XML elements parsed:
- `<pressure>` block — sets `has_pressure`, `P_min`, `P_max`, `P_initial`, `R_gas`, `P_ref` on the System.
- `<V x0=... x1=.../>` inside `<energy>` — optional molar volume polynomial for the PV term.
- `ideal_gas="true"` attribute on `<phase>` — enables the R·T·ln(P/P°) ideal-gas correction.

### `pde_energy.py`

Abstract base `EnergyModel` with two concrete subclasses. All models accept an optional `P` argument (default `0.0`; backward-compatible with files that have no pressure block):

- **`HSModel`**: `G(x,T,P) = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P°)]`
- **`PolyModel`**: `G(x,T,P) = Σᵢ cᵢ(T)·xⁱ [+ P·V(x)] [+ R·T·ln(P/P°)]`

Both use ascending-order coefficient conventions: `[c0, c1, c2, ...]` means `c0 + c1·x + c2·x² + ...`

Pressure terms:
- `V_coeffs` present → adds `P·V(x)` (Poynting correction for condensed phases).
- `ideal_gas=True` → adds `R_gas·T·ln(P/P_ref)` (ideal-gas chemical potential). Do **not** combine with `V_coeffs` for a pure ideal gas — that double-counts.
- `R_gas`, `P_ref` are passed down from the system-level `<pressure>` block by the parser.

### `pde_phase.py`

- **`Phase`** — name, phase_type (`'gas'`, `'liquid'`, `'solid'`, `'end_member'`), energy model, xmin/xmax. Key methods: `gibbs(x, T, P=0.0)`, `composition_grid(n_points=500)`, `is_point` (property, True when xmin==xmax).
- **`System`** — components list, all phases, energy_form, T_min/T_max/T_initial, plus pressure fields: `has_pressure`, `P_min`, `P_max`, `P_initial`, `R_gas`, `P_ref`. Convenience properties: `gas_phases`, `liquid_phases`, `solid_phases`, `end_members`.

### `pde_compute.py`

`compute_equilibrium(system, T, P=0.0, n_points=500)` evaluates G(x,T,P) for every phase, then computes the lower convex hull of the combined (x, G) point cloud via `scipy.spatial.ConvexHull`. Consecutive hull vertices from the same phase form a single-phase region; from different phases, a two-phase region. Returns an `EqResult`.

`EqResult` fields: `T`, `P`, `phase_curves`, `hull_x`, `hull_G`, `hull_phase_idx`, `regions`, plus properties `two_phase_regions` and `single_phase_regions`.

### `pde_viz.py`

Interactive PySide6 + matplotlib window with full pressure support:

**Pre-computation** — at startup, `launch_ui(system)` calls:
- `precompute_Tx_diagram(system, P_initial)` — 200 T steps (T_max → T_min) at fixed P.
- `precompute_Px_diagram(system, T_initial)` — 200 P steps (P_max → P_min) at fixed T (only when `system.has_pressure`).
- `precompute_diagram` is kept as a backward-compatible alias for `precompute_Tx_diagram`.

**Canvases:**
- **`GxCanvas`** (left): G(x) curves per phase + lower convex envelope + common tangent lines. Redrawn on every primary slider move.
- **`TxCanvas`** (right, Fixed P mode): T-x phase diagram revealed incrementally top-down; white cover rectangle shrinks as T slider moves down (O(1) updates).
- **`PxCanvas`** (right, Fixed T mode): P-x phase diagram, symmetric to `TxCanvas` but revealed bottom-up as P slider moves down.

**Controls:**
- **Mode selector** (only when `system.has_pressure`): "Fixed P (T-x)" / "Fixed T (P-x)" radio buttons swap the right canvas and reassign primary/secondary roles to the sliders.
- **T slider**: integer kelvin ticks; always visible.
- **P slider**: shown only when `system.has_pressure`; maps 0…N_P_STEPS-1 ticks linearly to P_min…P_max.
- **"Pre-compute full T-P-x" button** (only when `system.has_pressure`): launches `FullGridWorker` (a `QThread`) to compute the full N_T_STEPS × N_P_STEPS grid in the background. Progress bar and status label update via signals. Once cached, moving the secondary slider is instantaneous (index lookup instead of recompute).

**Slider interaction logic:**
- Primary slider (`valueChanged`) → fast O(1) update: nearest pre-computed result looked up by index.
- Secondary slider (`sliderReleased`) → recomputes the opposite sweep at the new secondary value (or performs an instant index lookup if the full grid is cached).
- Mode switch → resets the newly-primary canvas to `T_initial`/`P_initial` and re-reveals up to the current slider position.

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
