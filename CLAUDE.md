# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Document Hierarchy

This project uses a five-level document chain. All design documents live in
`dev/`. Read them in order when starting work:

1. `dev/VISION.md` — goals, principles, non-negotiables
2. `dev/ARCHITECTURE.md` — layout, modules, dependencies
3. `dev/DESIGN.md` — algorithms, data structures, math
4. `dev/PSEUDOCODE.md` — language-agnostic algorithm specs
5. Source code in `src/`

`dev/TODO.md` tracks tasks organized by level. Use `/focus` to start a session
and `/refine` to check consistency across the chain.

## Level Awareness

During development conversations, the programmer may shift between levels of
the design chain without explicitly noticing. For example, a discussion about a
code fix may drift into questioning an algorithm's design, or a design
discussion may surface a conflict with a core principle.

When you notice the conversation has moved to a different level than where it
started, say so briefly. For example: "This sounds like it's becoming an
ARCHITECTURE question — should we capture it there before continuing with the
code?" The goal is awareness, not interruption. Let the programmer decide
whether to switch context, propagate the change to the appropriate document, or
stay focused and defer.

Do not enforce rigid boundaries. The levels exist to organize thinking, not to
prevent it. A developer who is on a productive train of thought should not be
stopped — but when the thought resolves, help them recognize which documents it
touches so nothing is left inconsistent.

## Coding Style

- Lines MUST NOT exceed 80 characters — hard limit. At the same time, every
  line MUST be filled to at least 70 characters whenever content allows. Target
  the 70–80 character band; both the floor and the ceiling are mandatory.
- All program code must include rich, expressive documentation so that students
  can easily follow the source. Use clear, self-documenting variable names —
  avoid cryptic abbreviations. Prefer concise but meaningful names (e.g.,
  `elec_mom` not `em`; `nuc_pot` not `np`). Slightly-too-long names are far
  better than opaque short ones.

## Project Overview

**PDE** (Phase Diagram Explorer) is a Python tool for defining phase energy
curves and visualizing the resulting phase diagrams. Output formats include
interactive PySide6/matplotlib display and HDF5 + XDMF files for ParaView. For
full architecture detail see `dev/ARCHITECTURE.md`.

## Environment Setup

```bash
source .pde/pderc          # sets PDE_DIR, PDE_RC, PDE_BIN
```

## Build and Install

CMake installs the Python scripts only — no compilation.

```bash
cd build/release && cmake ../..
make install               # installs scripts to $PDE_DIR/bin/
```

## Running

```bash
pde.py                             # reads pde.in.xml, opens UI
pde.py -i myinput.xml -o myoutput
```

Each run appends the command and timestamp to a local `command` file.

## Configuration

`pderc.py` provides defaults via `parameters_and_defaults()` (`infile:
pde.in.xml`, `outfile: pde.out`). The repo template is `.pde/pderc.py`; a
`pderc.py` in the working directory overrides it.

## Quick Architecture Reference

```
src/scripts/
  pde.py              Entry point (installed to $PDE_DIR/bin/)
  pde_phase.py        Field, Phase, System data structures
  pde_energy.py       Energy models (HSModel, PolyModel, CALPHADModel, …)
  pde_input.py        XML parser → System
  pde_compute.py      Equilibrium via convex hull → EqResult
  pde_viz.py          PySide6 + matplotlib interactive UI
  pde_builder.py      Graphical input builder (QDialog)
  pde_check.py        Thermodynamic consistency checker
  pde_3d.py           3-D T-P-x visualization (pyvista)
  pde_export.py       HDF5 + XDMF export
```

### Pipeline

```
XML → parse_system() → System
  → compute_equilibrium(system, T, P) → EqResult
  → launch_ui(system) → PySide6 window
```

## Dependencies

`lxml`, `numpy`, `h5py`, `scipy`, `PySide6`, `matplotlib` — plus `argparse` and
`abc` from the stdlib. `pde_3d.py` additionally requires `pyvista` and
`pyvistaqt` (lazy-imported; the application works without them). CALPHAD support
(`energy_form='calphad'`) requires `pycalphad` (also lazy-imported).

## Testing

```bash
pytest tests/ -v
```

## Demo Jobs (`jobs/demo/`)

| File                              | Description                       |
|-----------------------------------|-----------------------------------|
| `pde.in.xml`                      | Symmetric eutectic (HS form)      |
| `azeotrope.in.xml`               | Minimum boiling azeotrope (HS)    |
| `vle-pressure.in.xml`            | VLE with variable pressure        |
| `isomorphous.xml`                | Complete solid solution            |
| `asymmetric-eutectic.xml`        | Asymmetric eutectic binary        |
| `eutectic-with-compound.xml`     | Eutectic with intermetallic       |
| `eutectoid.xml`                  | Eutectoid transformation          |
| `polynomial-asymmetric.xml`      | Asymmetric, polynomial energy     |
| `vle-pressure+alpha+beta.in.xml` | VLE + alpha/beta solid phases     |
| `calphad-al-mg.in.xml`          | Al-Mg eutectic (CALPHAD/TDB)      |
| `calphad-al-cu-mg.in.xml`      | Al-Cu-Mg ternary (CALPHAD/TDB)    |
