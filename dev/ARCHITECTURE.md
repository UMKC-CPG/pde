# Architecture

> **Document hierarchy:** VISION → **ARCHITECTURE** → DESIGN → PSEUDOCODE →
> Code. For goals and principles, see `VISION.md`.

---

## 1. Repository Layout

```
pde/
  dev/
    VISION.md              Goals and principles
    ARCHITECTURE.md        This document
    DESIGN.md              Algorithmic design
    PSEUDOCODE.md          Algorithm specifications
    TODO.md                Task list by level
  src/
    scripts/
      pde.py               Entry point (→ $PDE_DIR/bin/)
      pde_phase.py         Spec + runtime data model
      pde_energy.py        Energy models + VLE + patches
      pde_input.py         XML parser → System
      pde_compute.py       Equilibrium → EqResult
      pde_viz.py           PySide6 + matplotlib UI
      pde_builder.py       Graphical input builder
      pde_check.py         Consistency checker
      pde_3d.py            3-D visualization (pyvista)
      pde_export.py        HDF5 + XDMF export
  tests/                   Test suite (pytest)
  jobs/demo/               Demo input files
  .pde/
    pderc                  Bash environment setup
    pderc.py               Python resource control template
  CLAUDE.md                AI assistant guidance
```

---

## 2. Module Map

### Data model
- **pde_phase.py** — Spec layer (`FieldSpec`,
  `PhaseSpec`, `SystemSpec`) and runtime layer
  (`Field`, `Phase`, `System`).  The spec layer is
  the single canonical input form; `SystemSpec.
  to_system()` is the only place `EnergyModel`
  instances are built.  Topological helpers
  (`_dependency_edges`, `_topo_sort`) live here too.

### Energy
- **pde_energy.py** — `EnergyModel` (ABC), `HSModel`,
  `PolyModel`, `PiecewisePatchModel`,
  `CALPHADModel`, `CouplingTerm`.  VLE helper
  `compute_vle_gas_hs`.  Patch-H helpers.  All
  polynomial coefficient machinery lives here.
  `CALPHADModel` wraps `pycalphad.calculate()` for
  assessed TDB data (see DESIGN §16).

### Input / serialisation
- **pde_input.py** — XML parser.  Translates XML into
  `SystemSpec` (list of `FieldSpec` + `PhaseSpec`),
  then calls `spec.to_system()`.  VLE and patch
  dependencies are resolved by the topological sort
  inside `to_system()`, not by the parser itself.

### Computation
- **pde_compute.py** — `compute_equilibrium()` via scipy `ConvexHull`. Produces
  `EqResult` (hull vertices, regions, phase curves).

### Visualization
- **pde_viz.py** — `GxCanvas` (G(x) curves + envelope), `SweepCanvas` (field-x
  phase diagram), `MainWindow`, `FullGridWorker` (background T×P grid).
  Interactive display with drag-editing.

### Builder
- **pde_builder.py** — `BuilderWindow(QDialog)` +
  fitting functions (`apply_handle_drag`, etc.).
  Works directly with `PhaseSpec`/`SystemSpec` —
  no intermediate data model.

### Checking
- **pde_check.py** — Six check functions + `run_all_checks` orchestrator.
  `ConsistencyWarning` carries fix suggestions.

### 3-D and export
- **pde_3d.py** — `PhaseDiagram3D` and `Viz3DWindow` classes. Wraps a PyVista
  `QtInteractor`. Lazy-imported; the application works without it.
- **pde_export.py** — `export_binary_Tx` (2-D mesh),
  `export_3d_TPx` (3-D grid). Produces four files per
  export: `.hdf5` (data), `.xdmf` (ParaView wrapper),
  `.pv.py` (scene setup script), `.pv.ctrl.py`
  (VTKPythonAlgorithmBase plugin for native ParaView
  controls). See DESIGN §14 for architecture details.

### Entry point
- **pde.py** — CLI argument parsing via `ScriptSettings`, env setup, calls
  `parse_system()` + `launch_ui()`.

---

## 3. Dependency Graph

```
pde.py (entry point)
  +-- pde_input.py
  |     +-- pde_phase.py
  |     +-- pde_energy.py
  +-- pde_viz.py
  |     +-- pde_compute.py
  |     |     +-- pde_phase.py
  |     +-- pde_builder.py
  |     |     +-- pde_phase.py
  |     |     +-- pde_energy.py
  |     |     +-- pde_input.py  (lazy)
  |     +-- pde_check.py
  |     |     +-- pde_phase.py
  |     +-- pde_3d.py           (optional, lazy)
  |     |     +-- pde_compute.py
  |     +-- pde_export.py
  |           +-- pde_compute.py
  +-- pde_phase.py
        +-- pde_energy.py       (lazy, in make_energy_model)
```

---

## 4. Build System

```bash
source .pde/pderc
cd build/release && cmake ../..
make install       # copies scripts to $PDE_DIR/bin/
```

**Runtime:** lxml, numpy, h5py, scipy, PySide6,
matplotlib (all required).
**Optional:** pyvista, pyvistaqt (3-D visualization),
vedo (prototype only).
**Optional:** pycalphad (CALPHAD model support; see
DESIGN §16).
**Dev:** pytest.

### 4.1 CALPHAD / TDB Database Sources

CALPHAD integration (VISION goal 6) will use
`pycalphad` to read TDB (Thermo-Calc Database) files.
TDB is a plain-text de facto open standard — not
proprietary to Thermo-Calc. The following open sources
provide assessed thermodynamic data:

| Source | URL | Access |
|--------|-----|--------|
| TDBDB (Brown) | avdwgroup.engin.brown.edu | Free search → links |
| NIMS CPDDB | cpddb.nims.go.jp | Free download |
| pycalphad examples | github.com/pycalphad/pycalphad | MIT license |
| CALPHAD journal | via TDBDB links | Paper supplements |
| OpenCalphad | opencalphad.com | Open |
| SGTE pure elements | Dinsdale 1991; Graz mirror | Published |

TDBDB indexes 766+ TDB entries from 528 publications.
SGTE pure-element data (G(T) for 78 elements) is
embedded in most TDB files and provides the
experimentally assessed end-member energies that make
phase-diagram lenses close correctly. ESPEI (MIT) can
build new TDB files from experimental data via Bayesian
parameter optimization.

---

## 5. Pipeline

```
XML file
   │
   ▼
pde_input.parse_system()
   │  XML → SystemSpec (FieldSpecs + PhaseSpecs)
   │    → spec.to_system()  (topo sort, build models)
   │      → System (components, phases, fields, title)
   │
   ├──► pde_viz.launch_ui(system)
   │       ├── precompute: compute_equilibrium
   │       │     per field value → list[EqResult]
   │       ├── GxCanvas      — G(x) curves + hull
   │       ├── SweepCanvas   — field-x diagram
   │       ├── BuilderWindow — live editor
   │       └── Viz3DWindow   — 3-D surface
   │
   └──► pde_export — HDF5 + XDMF mesh
```

### Per-slider update cycle

```
slider move → _field_sliders[i].valueChanged
  ├─ primary slider: O(1) lookup in _precomputed
  │    → GxCanvas.redraw(result)
  │    → SweepCanvas.update_reveal(value)
  └─ secondary slider release: full recompute
       → SweepCanvas.reset(new_precomputed)
```

---

## 6. Development Checkpoints

Development follows a four-phase remediation plan. See `TODO.md` for the full
task list organized by phase:

- **Phase 0:** Canonical form redesign — introduce SystemSpec, PhaseSpec, and
  FieldSpec as the single authoritative spec layer.
- **Phase 1:** Complete the field abstraction — EqResult field_values dict,
  compute_equilibrium dict API, retire the gibbs() backward-compat shim.
- **Phase 2:** Cleanup — replace pde.py entry point, fix palette persistence,
  add unit-system metadata to the XML schema.
- **Phase 3:** Test suite — energy model correctness, round-trip invariants,
  equilibrium invariants, and regression baselines.
