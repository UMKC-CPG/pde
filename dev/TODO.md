# Task List

> **Document hierarchy:** Tasks are organized by the level of the design chain
> they affect. Each item should cite the relevant document section.

---

## VISION

*No pending VISION items.*

---

## ARCHITECTURE

- [ ] A-1. After Phase 0, update the dependency graph (ARCHITECTURE §3)
  to reflect the elimination of `SystemData.from_system()` and the new
  SystemSpec construction flow.
- [ ] A-2. After Phase 0, remove the "Legacy design doc" note from the
  repository layout once the root `DESIGN.md` is archived or deleted.

---

## DESIGN

### Phase 0 — Canonical form redesign (DESIGN §11, §12.1–12.3, 12.5)

- [x] D-1. Define `FieldSpec`, `PhaseSpec`, `SystemSpec` dataclasses in
  `pde_phase.py`. `PhaseSpec` uses `model_type: str` + `model_params:
  dict`. `SystemSpec` uses `fields: list[FieldSpec]`. Pseudocode added
  as §7; existing construction algorithm renumbered to §7.2.
  (DESIGN §11.2) — Resolved 2026-04-02.
- [ ] D-2. Implement `PhaseSpec.make_energy_model(specs_by_name, built)`
  with dispatch on `model_type`. Cover `'HS'`, `'polynomial'`,
  `'piecewise_patch'`. Resolve VLE and patch dependencies via the
  `specs_by_name` dict. (DESIGN §11.3, PSEUDOCODE §7)
- [ ] D-3. Implement `SystemSpec.to_system()` using topological sort +
  `make_energy_model()` per phase. This replaces the three-pass logic in
  both `parse_system()` and `SystemData.to_system()`. (DESIGN §11.4)
- [ ] D-4. Rewrite `parse_system()` as XML → SystemSpec → System. The
  parser becomes a thin translator that calls `SystemSpec.to_system()`
  internally. (DESIGN §11.4)
- [ ] D-5. Add shim properties to `PhaseSpec` (`hs_H`, `hs_S`, `hs_V`,
  `poly`, `vle`, etc.) delegating into `model_params` so `BuilderWindow`
  keeps working without UI changes. (DESIGN §11.5)
- [ ] D-6. Update `MainWindow` to hold a `SystemSpec` alongside `System`.
  Builder receives the live `SystemSpec` directly. Eliminate
  `SystemData.from_system()`. (DESIGN §11.5)
- [ ] D-7. Move patch-H computation helpers to `pde_energy.py`. Both
  `pde_input` and `pde_builder` call the shared coefficient-list form.
  (DESIGN §12.2)
- [ ] D-8. Implement `SystemSpec.to_xml_str()` and `SystemSpec.from_xml()`
  for serialisation round-tripping. (DESIGN §11.4)
- [ ] D-9. Verify that `_G_from_phase_data` duplication (issue 12.5) is
  dissolved: canvas drag now calls `spec.make_energy_model().gibbs()`.
  (DESIGN §12.5)

### Phase 1 — Field abstraction (DESIGN §12.4, 12.6, 12.7)

- [ ] D-10. Give `EqResult` a `field_values: dict[str, float]`. Add `.T`
  and `.P` as computed properties for backward compat. (DESIGN §12.4)
- [ ] D-11. Update `compute_equilibrium` signature to `(system,
  field_values: dict)`. Forward the dict to every `phase.gibbs()` call.
  (DESIGN §12.4)
- [ ] D-12. Remove the `gibbs()` backward-compat shim. The only public API
  becomes `gibbs(x, field_values: dict)`. Update `Phase.gibbs()`,
  `compute_equilibrium()`, and `pde_check`. (DESIGN §12.6)
- [ ] D-13. Eliminate `_extract_field_values`. SweepCanvas uses
  `[r.field_values[field.name] for r in precomputed]`. (DESIGN §12.4)
- [ ] D-14. Generalise `FullGridWorker` to take `field0_values`,
  `field1_values`, and two field indices instead of named T/P arrays.
  (DESIGN §12.7)
- [ ] D-15. Generalise `PhaseDiagram3D` and `Viz3DWindow` to take two
  `Field` objects. Axis labels from `Field.symbol`. Hide the 3-D button
  for single-field systems. (DESIGN §12.7)

---

## PSEUDOCODE

- [ ] P-1. After Phase 0, update pseudocode §6 (sweep precomputation) to
  use `field_values: dict` instead of positional T, P.
- [ ] P-2. After Phase 1, verify all pseudocode sections reflect the
  dict-based API and the absence of the gibbs() shim.

---

## CODE

### Phase 2 — Cleanup (DESIGN §12.8, 12.9, 12.10)

- [ ] C-1. Replace `ScriptSettings` in `pde.py` with a plain
  `argparse.ArgumentParser` function. Remove unused imports (`h5py`,
  `math`, `random`). (DESIGN §12.8)
- [ ] C-2. Make `MainWindow` track active palette as instance variable.
  Pass it to `_color_map()` so `reload_system()` does not reset the
  user's palette choice. (DESIGN §12.9)
- [ ] C-3. Add `<units>` block to XML schema and builder UI as a
  human-readable record. Optionally validate R_gas against known values
  per unit combination. (DESIGN §12.10)

### Phase 3 — Test suite (VISION goal 4, DESIGN §7)

- [ ] C-4. Energy model unit tests: known-good G values at specific
  (x, T, P) inputs for HSModel, PolyModel, PiecewisePatchModel. VLE
  tangency conditions for `compute_vle_gas_hs` output.
- [ ] C-5. Round-trip invariants: `SystemSpec.from_xml(xml)` →
  `to_xml_str()` reproduces equivalent XML.
- [ ] C-6. Equilibrium invariants: hull points lie on phase curves;
  single-phase regions have d²G/dx² > 0; two-phase boundaries are
  common-tangent pairs.
- [ ] C-7. Regression baselines: for each demo job, record tie-line
  endpoints at representative field values as expected outputs.

### Future — Multi-component (VISION goal 7, DESIGN §13)

- [ ] C-8. Multivariate energy models: Redlich–Kister–
  Muggianu pairwise expansion for ternary+. New
  `EnergyModel` subclass and XML schema.
- [ ] C-9. Simplicial composition grid: replace scalar
  `xmin`/`xmax` with triangular (ternary) or
  tetrahedral (quaternary) meshes in `pde_phase.py`.
- [ ] C-10. N-D hull region extraction: rewrite
  `_extract_regions()` to identify simplex faces for
  ternary+. Update lower-hull filter index.
- [ ] C-11. Ternary visualisation: Gibbs triangle
  composition axis in `pde_viz.py` or new module.

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
