# Task List

> **Document hierarchy:** Tasks are organized by the level of the design chain
> they affect. Each item should cite the relevant document section.

---

## VISION

*No pending VISION items.*

---

## ARCHITECTURE

- [x] A-1. Update the dependency graph (ARCHITECTURE §3)
  and module map (§2) to reflect the SystemSpec
  construction flow and the pde_phase → pde_energy
  lazy import.  — Resolved 2026-04-07.
- [x] A-2. Remove the "Legacy design doc" note from
  the repository layout; root `DESIGN.md` deleted.
  — Resolved 2026-04-07.

---

## DESIGN

### Phase 0 — Canonical form redesign (DESIGN §11, §12.1–12.3, 12.5)

- [x] D-1. Define `FieldSpec`, `PhaseSpec`, `SystemSpec` dataclasses in
  `pde_phase.py`. `PhaseSpec` uses `model_type: str` + `model_params:
  dict`. `SystemSpec` uses `fields: list[FieldSpec]`. Pseudocode added
  as §7; existing construction algorithm renumbered to §7.2.
  (DESIGN §11.2) — Resolved 2026-04-02.
- [x] D-2. Implement `PhaseSpec.make_energy_model(specs_by_name, built)`
  with dispatch on `model_type`. Cover `'HS'`, `'polynomial'`,
  `'piecewise_patch'`. Resolve VLE and patch dependencies via the
  `specs_by_name` dict. (DESIGN §11.3, PSEUDOCODE §7)
  — Resolved 2026-04-07.
- [x] D-3. Implement `SystemSpec.to_system()` using topological sort +
  `make_energy_model()` per phase. This replaces the three-pass logic in
  both `parse_system()` and `SystemData.to_system()`. (DESIGN §11.4)
  — Resolved 2026-04-07.
- [x] D-4. Rewrite `parse_system()` as XML → SystemSpec → System. The
  parser becomes a thin translator that calls `SystemSpec.to_system()`
  internally. (DESIGN §11.4)
  — Resolved 2026-04-07.
- [x] D-5/D-6/D-8/D-9. Merged into a single
  migration: `PhaseData`/`SystemData` deleted.
  `PhaseSpec` gains permanent convenience properties
  (H_coeffs, S_coeffs, V_coeffs, etc.) delegating
  into model_params.  `BuilderWindow` now takes a
  `SystemSpec` directly.  `MainWindow` stores the
  live `SystemSpec`; no `from_system()` round-trip.
  `SystemSpec.to_xml_str()` and
  `parse_system_spec()` provide serialisation.
  `_G_from_phase_data` replaced by
  `_G_from_phase_spec`.  (DESIGN §11.5, §12.5)
  — Resolved 2026-04-07.
- [x] D-7. Move patch-H computation helpers to `pde_energy.py`. Both
  `pde_input` and `pde_builder` call the shared coefficient-list form.
  (DESIGN §12.2) — Resolved 2026-04-07.

### Phase 1 — Field abstraction (DESIGN §12.4, 12.6, 12.7)

- [x] D-10. `EqResult` stores `field_values: dict`;
  `.T` and `.P` are computed properties.
  — Resolved 2026-04-07.
- [x] D-11. `compute_equilibrium(system,
  field_values)` — dict forwarded to
  `phase.gibbs(x, fv)`.  All call sites updated.
  — Resolved 2026-04-07.
- [x] D-12. Removed the `gibbs()` backward-compat
  shim.  `EnergyModel.gibbs(x, field_values)` is
  the only public API.  `Phase.gibbs()`,
  `pde_check`, all updated.
  — Resolved 2026-04-07.
- [x] D-13. `_extract_field_values` eliminated;
  replaced by direct `r.field_values[field.name]`
  access at all 8 call sites.
  — Resolved 2026-04-07.
- [x] D-14. `FullGridWorker` takes two `Field`
  objects + value arrays instead of named T/P.
  — Resolved 2026-04-07.
- [x] D-15. `PhaseDiagram3D` and `Viz3DWindow` take
  two `Field` objects.  Axis labels from
  `Field.symbol`/`unit`.  3-D button already hidden
  for single-field systems.
  — Resolved 2026-04-07.

---

## PSEUDOCODE

- [x] P-1. Update pseudocode §1 and §6 to use
  `field_values: dict` instead of positional T, P.
  EqResult now takes `field_values` instead of
  separate T and P.  — Resolved 2026-04-07.
- [x] P-2. Pseudocode §1 and §6 use `field_values:
  dict`; no gibbs() shim references remain.
  — Resolved 2026-04-07.

---

## CODE

### Phase 2 — Cleanup (DESIGN §12.8, 12.9, 12.10)

- [x] C-2. `_init_system_state` preserves existing
  user-chosen colors across reloads; only assigns
  fresh defaults for new phases.  Two-phase color
  and hatch also preserved.  (DESIGN §12.9)
  — Resolved 2026-04-07.
- [x] C-3. `<units>` element in `<system>` declares
  energy, temperature, and pressure units.  R_gas
  validated against a lookup table at parse time;
  CALPHAD systems default to SI.  All 11 demo files
  carry explicit `<units>` blocks.  Builder UI:
  editable combo boxes for energy, temperature,
  and pressure units; locked to SI for CALPHAD.
  (DESIGN §12.10) — Resolved 2026-04-07.

### Phase 3 — Test suite (VISION goal 4, DESIGN §7)

- [x] C-4. Energy model unit tests in
  `tests/test_energy_models.py`: HSModel (5 tests),
  PolyModel (3), PiecewisePatchModel (3), VLE
  tangency (2), CALPHADModel (10: sublattice
  layout, SER reference states, melting-point
  crossover, vectorised shape, finite G,
  default pressure, couplings).
  — Resolved 2026-04-07.
- [x] C-5. Round-trip invariants in
  `tests/test_round_trip.py`: spec metadata and
  G(x) values survive parse → to_xml_str → re-parse
  for all 10 demo files.  — Resolved 2026-04-07.
- [x] C-6. Equilibrium invariants in
  `tests/test_equilibrium.py`: hull-on-curves,
  hull-envelope convexity, common-tangent condition
  for all 10 demo files.  — Resolved 2026-04-07.
- [x] C-7. Regression baselines in
  `tests/test_regression.py`: 9 tie-line endpoint
  checks + 10 no-tie-at-initial-T checks across all
  demo files.  — Resolved 2026-04-07.

### CALPHAD integration (VISION goal 6, DESIGN §16)

- [x] C-16. Builder UI support for CALPHAD:
  `PhaseEditorWidget` gains a CALPHAD stack page
  (index 3) with TDB phase name input.
  `BuilderWindow` adds 'calphad' to the energy form
  combo, a TDB file picker group (visible only for
  CALPHAD), and component injection in
  `_collect_system_spec()`.  Round-trip verified
  with the Al-Mg demo.  — Resolved 2026-04-07.
- [x] C-15. `CALPHADModel` in `pde_energy.py` wraps
  `pycalphad.calculate()` with sublattice-aware
  site-fraction mapping.  `SystemSpec.to_system()`
  loads TDB once; parser handles
  `energy_form='calphad'` + `<tdb>` element.
  Demo Al-Mg TDB and input XML in `jobs/demo/`.
  All 89 tests pass.  (DESIGN §16)
  — Resolved 2026-04-07.

### Multi-component (VISION goal 7, DESIGN §13)

- [x] C-9. Simplicial composition grid:
  `_simplex_grid()` generates uniform (N-1)-simplex
  meshes.  `Phase` gains `n_components`; binary
  path (linspace) unchanged.  `CALPHADModel`
  generalised for N-component site-fraction
  construction.  — Resolved 2026-04-07.
- [x] C-10. N-D hull construction + region extraction:
  `compute_equilibrium()` handles (n, N) points
  with generalised lower-hull filter.  Binary 1-D
  walk preserved.  Ternary+ uses simplex-based
  `_extract_regions_nd()` grouping facets by
  phase-set.  — Resolved 2026-04-07.
- [ ] C-11. Ternary visualisation: Gibbs triangle
  composition axis in `pde_viz.py` or new module.
- [ ] C-8. Multivariate energy models: Redlich-Kister-
  Muggianu pairwise expansion for ternary+.  New
  `RKMModel` subclass and XML schema.  (Deferred:
  CALPHAD ternary works without it.)

### Future — ParaView plugin (DESIGN §14)

- [ ] C-12. Integrate ParaView plugin generation into
  `pde_export.py`: `GetDisplayProperties()` instead of
  `Show()`, no `Render()` in `RequestData`, self-hide
  on first run to prevent scalar-bar contamination.
- [ ] C-13. Re-enable `LoadPlugin()` in the generated
  `.pv.py` script once plugin stability is confirmed.
- [ ] C-14. Add per-phase visibility checkboxes to the
  generated `.pv.ctrl.py` plugin.

---

## ARCHIVE

- [x] V-1. CALPHAD extensibility (goal 6) reviewed.
  No additional design principles needed beyond
  model-type dispatch (principle 4). Decision: pursue
  medium-term path — `CALPHADModel` wrapping
  `pycalphad` + TDB databases. Native HS/polynomial
  models cannot match end-member value+slope while
  keeping interior behaviour correct (DESIGN §2.4).
  Rationale captured in VISION goal 6 annotation.
  Resolved 2026-04-02.
