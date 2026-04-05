# Vision

## Purpose

PDE (Phase Diagram Explorer) computes and visualizes
binary phase diagrams from user-defined free-energy
curves. It takes thermodynamic model parameters as input,
computes equilibrium via the lower convex hull of Gibbs
energy vs. composition, and produces interactive phase
diagrams. Output formats include interactive
PySide6/matplotlib display and HDF5 + XDMF structured
meshes for ParaView.

PDE serves a dual mission: **pedagogical** (a
self-explanatory tool for introducing students to phase
diagrams, G(x) energy curves, and thermodynamic
reasoning) and **research** (a flexible platform for
exploring phase behaviour in systems beyond standard
T-P-x thermodynamics). The design goal is a system that
is easy to use in the common case and extensible to the
uncommon case without requiring a rewrite.

## Goals

1. **Interactive exploration.** Users define energy curves (H-S, polynomial, or
   piecewise-patch forms) and immediately see the resulting G(x) curves and
   phase diagram. Dragging handles or sliders updates the diagram in real time.

2. **Multi-field generality.** The system supports an arbitrary number of
   intensive thermodynamic fields (temperature, pressure, magnetic field,
   chemical potential, etc.). Adding a new field requires no changes to the
   computation, checking, or export code.

3. **Graphical construction.** A builder dialog allows users to create and edit
   phase diagrams without writing XML by hand. The builder is the primary
   authoring interface.

4. **Thermodynamic consistency.** A checker validates energy models against
   known thermodynamic constraints (coverage, convexity, VLE tangency, and
   end-member matching) and reports warnings with suggested fixes.

5. **Persistent output.** Phase diagrams export to HDF5 + XDMF structured
   meshes suitable for ParaView visualization and post-processing.

6. **Extensible energy models.** New energy model types
   (e.g., CALPHAD) can be added by defining a new
   model_type string and model_params keys, without
   modifying the spec layer or the builder framework.
   CALPHAD integration via `pycalphad` + TDB databases
   is the primary extensibility target. The native HS
   and polynomial models lack the flexibility to match
   both end-member values/slopes and interior mixing
   behaviour simultaneously (see DESIGN §2.4 concavity
   limitation); assessed CALPHAD databases solve this
   with experimentally grounded Gibbs energy functions.
   No additional design principles beyond model-type
   dispatch (principle 4) are required — a
   `CALPHADModel` subclass wrapping `pycalphad` fits
   the existing `EnergyModel` interface. See DESIGN
   §16 for integration details and the reference
   memory `ref_tdb_sources.md` for database sources.

7. **Multi-component generalisation.** The architecture
   should support ternary and higher-order systems.
   Composition moves from a scalar to a simplex; the
   convex-hull equilibrium engine generalises
   naturally. Visualisation strategies scale from
   Gibbs triangles (ternary) to VR volume rendering
   (quaternary+). See DESIGN §13.

8. **Higher-dimensional field exploration.** Beyond
   temperature and pressure, users should be able to
   add intensive fields (chemical potential, electric
   field, magnetic field) and explore the resulting
   high-dimensional phase space through linked/sliced
   views and ParaView batch analysis. See DESIGN §14.

## Design Principles

1. **Single canonical path.** Every `System` object is derived through exactly
   one code path: raw input → SystemSpec → System. No round-tripping and no
   parallel construction paths. (Motivates DESIGN §11.)

2. **Field-agnostic computation.** Temperature and pressure are Field instances
   with conventional names, not structurally privileged. Computation, checking,
   and export operate on generic field_values dicts. (Motivates DESIGN §12,
   issues 12.4 and 12.6.)

3. **Separation of spec from runtime.** The spec layer (SystemSpec, PhaseSpec,
   FieldSpec) is mutable, serializable, and UI-friendly. The runtime layer
   (System, Phase, EnergyModel) is derived, immutable in practice, and
   optimized for computation. (Motivates DESIGN §11.2.)

4. **Model-type dispatch.** Energy model specifics live in model_params dicts
   keyed by model_type strings. Adding a model type requires one factory case
   and new param keys — the spec and builder frameworks do not change.
   (Motivates DESIGN §11.3.)

5. **Dependency-aware construction.** Cross-phase
   dependencies (VLE gas needing liquid H/S, patches
   needing target-phase G) are declared in model_params
   and resolved by topological ordering during System
   construction. (Motivates DESIGN §11.4.)

6. **Pedagogy first, then power.** Named presets (e.g.,
   "standard thermodynamics", "VLE with pressure")
   should cover the common cases effortlessly. The full
   generality is available underneath but not forced on
   the user. Additions should be additive, not
   invasive — a new field type or coupling term should
   be expressible by adding data, not by scattering new
   `if has_field_X` branches through existing code.

7. **Joint interface design.** The field generalisation
   (goals 2, 8) and the multi-component generalisation
   (goal 7) both affect `EnergyModel.gibbs()`. They
   must be designed together at the interface level so
   that `gibbs(x: ndarray, field_values: dict)` covers
   both generalisations in a single revision. `x` is
   already an ndarray; for multi-component it becomes
   2-D. The `field_values` dict handles arbitrary
   fields. No second round of breaking changes.
