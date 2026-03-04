# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PDE** (Phase Diagram Explorer) is a Python tool for defining phase energy curves and visualizing the resulting phase diagrams. The intended output format is HDF5 + XDMF.

For full architecture detail see `docs/architecture.md`.

## Environment Setup

```bash
source .pde/pderc      # sets PDE_DIR, PDE_RC, PDE_BIN; adds PDE_BIN to PATH
```

## Build and Install

CMake installs the Python scripts only — no compilation.

```bash
cd build/release && cmake ../..
make install            # installs pde.py and companion modules to $PDE_DIR/bin/
```

## Running

```bash
pde.py                        # reads pde.in.xml, opens interactive UI
pde.py -i myinput.xml -o myoutput
```

Each run appends the command and timestamp to a local `command` file.

## Configuration

`pderc.py` provides defaults via `parameters_and_defaults() → dict` (`infile: pde.in.xml`, `outfile: pde.out`). The repo template is `.pde/pderc.py`; a `pderc.py` in the working directory overrides it.

## Architecture

```
src/scripts/pde.py          Main script (installed to $PDE_DIR/bin/)
src/scripts/pde_input.py    XML input parser → System object
src/scripts/pde_energy.py   Energy model classes (HSModel, PolyModel, CouplingTerm)
src/scripts/pde_phase.py    Field, Phase, System data structures
src/scripts/pde_compute.py  Equilibrium computation (convex hull) → EqResult
src/scripts/pde_viz.py      PySide6 + matplotlib interactive UI
src/scripts/pde_builder.py  Graphical input builder (QDialog) + PhaseData/SystemData
src/scripts/pde_check.py    Consistency checker (warnings panel in BuilderWindow)
src/scripts/pde_3d.py       3D T-P-x visualization (PhaseDiagram3D + Viz3DWindow)
src/scripts/pde.1.py        Standalone vedo + PySide6 prototype (not installed)
.pde/pderc                  Bash environment setup script
.pde/pderc.py               Python resource control file template
jobs/demo/                  Demo input files
```

### Pipeline

```
XML input → pde_input.parse_system() → System
                 ↓
    pde_compute.compute_equilibrium(system, T, P) → EqResult
                 ↓
    pde_viz.launch_ui(system) → PySide6 window
```

## Key Data Structures

**`Field`** (`pde_phase.py`) — `name`, `symbol`, `unit`, `min_val`, `max_val`, `initial_val`. Role (primary/secondary sweep axis) is NOT stored on Field — it belongs to view config.

**`System`** — `components`, `phases`, `energy_form`, `fields: list[Field]` (fields[0] = temperature by convention), `title`. Backward-compat properties: `T_field`, `T_min/max/initial`, `P_field`, `has_pressure`, `P_min/max/initial/unit`, `gas/liquid/solid_phases`, `end_members`.

**`EnergyModel`** (ABC) — `gibbs()` shim accepts both `gibbs(x, T, P=0.0)` (old) and `gibbs(x, field_values: dict)` (new). Subclasses implement `_gibbs_impl(x, field_values: dict)`. Coefficient convention: ascending order `[c0, c1, c2, …]`.
- **`HSModel`**: `G = H(x) − T·S(x) [+ P·V(x)] [+ R·T·ln(P/P°)]`
- **`PolyModel`**: `G = Σᵢ cᵢ(T)·xⁱ [+ P·V(x)] [+ R·T·ln(P/P°)]`

**`PhaseData` / `SystemData`** (`pde_builder.py`) — pure-Python mirrors of Phase/System for the builder. `SystemData.to_system()`, `to_xml_str()`, `from_system()`, `from_xml()`.

## XML Input

Two accepted schemas (both produce identical `System` objects):

- **New** (builder output): `<fields>` block with `<field name=… symbol=… unit=… min=… max=… initial=… />` elements.
- **Legacy** (demo files): separate `<temperature>` and optional `<pressure>` blocks.

Optional: `<title>`, `<V x0=… x1=…/>` inside `<energy>` (molar volume), `ideal_gas="true"` on `<phase>`, `R_gas`/`P_ref` on pressure element.

VLE gas phases: `<vle T_bp_A=… T_bp_B=… L_A=… L_B=…/>` element inside `<phase>`; parser does a two-pass build so the liquid H/S are available when constructing the gas phase.

## Visualization (`pde_viz.py`)

- **`GxCanvas`** (left): G(x) curves + lower convex envelope. Supports `'handles'` edit mode for HS phases (3 draggable diamonds; vertical drag fits H coefficients, horizontal drag adjusts x range). **Ctrl+click** anywhere on a curve → rigid shift: drag moves the whole curve as a rigid body in both x (translates composition range) and y (shifts G offset); axes autoscaling is frozen during the drag.
- **`SweepCanvas`** (right): unified T-x / P-x / field-x diagram. One per field; created lazily on first mode switch. Y axis derived from `primary_field`.
- **Sliders**: one per `system.fields[i]`, range 0…N_STEPS−1 mapped to `field.min_val…field.max_val`. Primary slider → O(1) lookup; secondary slider release → recompute sweep.
- **`FullGridWorker`**: background QThread for full N_T×N_P grid; pause/resume via `threading.Event`.
- **`MainWindow` state**: `_precomputed[i]`, `_field_arr[i]`, `_primary_idx`, `_sweep_canvases: dict[int, SweepCanvas]`.

## Builder (`pde_builder.py`)

- **`BuilderWindow(QDialog)`** — non-modal; emits `system_applied(System)` on Apply.
- **`apply_handle_drag()`** — 3×3 Vandermonde solve for H₀, H₁, H₂ from 3 handle (x, G) pairs.
- **`apply_xrange_drag()`** — updates xmin/xmax; clamped with 0.02 margin.
- **`apply_rigid_shift(phase_data, delta_G, delta_x=0.0)`** — vertical shift adds `delta_G` to `hs_H[0]`; horizontal shift reparameterises H and S polynomials via `_shift_poly_coeffs` (composition p(x) → p(x−δ)) so the curve shape is preserved at the new x domain, then clamps and updates xmin/xmax.
- **`_shift_poly_coeffs(coeffs, dx)`** — returns ascending-order coefficients of p(x−dx) via numpy `poly1d` composition; handles arbitrary degree.
- **`BuilderWindow.update_phase_data(name, data)`** — syncs spinboxes after a canvas drag.

## Demo Jobs (`jobs/demo/`)

| File | Description |
|------|-------------|
| `pde.in.xml` | Symmetric eutectic binary (HS form) |
| `azeotrope.in.xml` | Minimum boiling azeotrope (HS form) |
| `vle-pressure.in.xml` | Vapor-liquid equilibrium with variable pressure (P=0.5–5 atm) |
| `isomorphous.xml` | Isomorphous (complete solid solution) |
| `asymmetric-eutectic.xml` | Asymmetric eutectic binary |
| `eutectic-with-compound.xml` | Eutectic with intermetallic compound |
| `eutectoid.xml` | Eutectoid transformation |
| `polynomial-asymmetric.xml` | Asymmetric system using polynomial energy form |
| `vle-pressure+alpha+beta.in.xml` | VLE + two partial-range solid phases (alpha/beta); tests rigid horizontal shift |

## Dependencies

`lxml`, `numpy`, `h5py`, `scipy`, `PySide6`, `matplotlib` — plus `argparse`, `abc` (stdlib).
`pde_3d.py` additionally requires `pyvista` and `pyvistaqt` (lazy-imported; app works without them).
`pde.1.py` additionally requires `vedo`.
