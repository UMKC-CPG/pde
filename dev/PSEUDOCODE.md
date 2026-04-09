# Pseudocode

> **Document hierarchy:** VISION → ARCHITECTURE → DESIGN → **PSEUDOCODE** →
> Code. For the design rationale behind these algorithms, see `DESIGN.md`.

---

## 1. Equilibrium Computation (DESIGN §4)

```
function compute_equilibrium(system,
                             field_values: dict):
    # field_values maps field names to current
    # values, e.g. {'temperature': 1200.0,
    #               'pressure': 2.0}

    all_x  ← []
    all_G  ← []
    curves ← []

    for each phase in system.phases:
        x ← linspace(phase.xmin, phase.xmax,
                      n_points)
        G ← phase.gibbs(x, field_values)
        append (x, G) to all_x, all_G
        append (x, G, phase) to curves

    points ← stack(all_x, all_G) as (N, 2)
    hull   ← ConvexHull(points)

    # Extract lower hull: facets with
    # downward-pointing normal
    lower_vertices ← {}
    for each simplex in hull.simplices:
        normal ← hull.equations[simplex].normal
        if normal[1] < 0:
            add simplex vertices to lower_vertices

    sort lower_vertices by x coordinate
    hull_x, hull_G ← points[lower_vertices]

    # Classify regions by walking hull vertices
    regions ← []
    for consecutive pairs (i, i+1) in
            lower_vertices:
        phase_i ← phase owning point i
        phase_j ← phase owning point j
        if phase_i == phase_j:
            append single-phase region [x_i, x_j]
        else:
            append two-phase region [x_i, x_j]

    return EqResult(field_values, curves,
                    hull_x, hull_G, regions)
```

---

## 2. VLE Gas Construction (DESIGN §2.6)

Given liquid H(x) and S(x) polynomials, boiling points T_bp_A (at x=0) and
T_bp_B (at x=1), and latent heats L_A and L_B, derive gas H(x) and S(x)
satisfying four tangency conditions.

```
function compute_vle_gas_hs(liq_H, liq_S, T_bp_A, T_bp_B,
                            L_A, L_B):
    # At x=0, T=T_bp_A: G_gas = G_liq and dG/dx equal
    # At x=1, T=T_bp_B: G_gas = G_liq and dG/dx equal
    #
    # G = H(x) - T*S(x), so four equations in H_gas, S_gas
    # coefficients.

    G_liq_A  ← eval(liq_H, 0) - T_bp_A * eval(liq_S, 0)
    dG_liq_A ← eval(liq_H', 0) - T_bp_A * eval(liq_S', 0)
    G_liq_B  ← eval(liq_H, 1) - T_bp_B * eval(liq_S, 1)
    dG_liq_B ← eval(liq_H', 1) - T_bp_B * eval(liq_S', 1)

    # Gas G at boiling points = liquid G + latent heat
    G_gas_A  ← G_liq_A + L_A
    dG_gas_A ← dG_liq_A
    G_gas_B  ← G_liq_B + L_B
    dG_gas_B ← dG_liq_B

    # Solve 4×4 system for H_gas[0..1], S_gas[0..1]:
    #   H0 - T_bp_A * S0                  = G_gas_A
    #   H1 - T_bp_A * S1                  = dG_gas_A
    #   H0 + H1 - T_bp_B * (S0 + S1)     = G_gas_B
    #   H1 - T_bp_B * S1                  = dG_gas_B

    solve linear system → H_gas = [H0, H1]
                          S_gas = [S0, S1]
    return (H_gas, S_gas)
```

---

## 3. Patch-H Computation (DESIGN §2.4)

Compute a replacement polynomial for the left or right tail of a phase's
enthalpy so that G and dG/dx match a target phase at the cut point.

```
function compute_left_patch_H(H, S, H_target, S_target,
                               xmin, x_cut, T):
    # Evaluate target phase G and dG/dx at the cut point
    G_target  ← eval(H_target, x_cut)
                  - T * eval(S_target, x_cut)
    dG_target ← eval(H_target', x_cut)
                  - T * eval(S_target', x_cut)

    # The patch polynomial must satisfy:
    #   H_patch(x_cut) - T * S(x_cut) = G_target
    #   H_patch'(x_cut) = dG_target + T * S'(x_cut)
    # Using a linear H_patch = [a0, a1]:

    dH ← dG_target + T * eval(S', x_cut)
    H_at_cut ← G_target + T * eval(S, x_cut)

    a1 ← dH
    a0 ← H_at_cut - a1 * x_cut

    return [a0, a1]
```

The right-patch version is analogous, with the roles of xmin/xmax and the roles
of left/right reversed.

---

## 4. Handle Drag — Vandermonde Solve (DESIGN §6.3)

```
function apply_handle_drag(phase_data, handles):
    # handles = [(x0, G0), (x1, G1), (x2, G2)]
    # at current temperature T

    # G = H(x) - T*S(x), so H(x) = G(x) + T*S(x)
    # With 3 points and quadratic H = H0 + H1*x + H2*x²,
    # build the Vandermonde system:

    for i in 0..2:
        y[i] ← handles[i].G + T * eval(S, handles[i].x)

    V ← [[1, x0, x0²],
          [1, x1, x1²],
          [1, x2, x2²]]

    [H0, H1, H2] ← solve(V, y)
    phase_data.hs_H ← [H0, H1, H2]
```

---

## 5. Rigid Shift — Polynomial Reparameterisation (DESIGN §6.3)

```
function apply_rigid_shift(phase_data, delta_G, delta_x):
    if delta_G != 0:
        phase_data.hs_H[0] += delta_G       # vertical shift

    if delta_x != 0:
        # Reparameterise: p(x) → p(x - dx)
        phase_data.hs_H ← shift_poly(hs_H, delta_x)
        phase_data.hs_S ← shift_poly(hs_S, delta_x)
        phase_data.xmin += delta_x
        phase_data.xmax += delta_x
        clamp xmin, xmax to [0, 1]

function shift_poly_coeffs(coeffs, dx):
    # Given ascending coeffs [c0, c1, c2, …] of p(x),
    # return ascending coeffs of p(x - dx).
    p ← poly1d(reverse(coeffs))       # numpy descending
    q ← compose p with (x - dx)       # p(x - dx)
    return reverse(q.coeffs)          # back to ascending
```

---

## 6. Sweep Precomputation (DESIGN §5.4)

```
function precompute_sweep_diagram(system,
        primary_idx, fixed_values, n_steps):
    field  ← system.fields[primary_idx]
    values ← linspace(field.min_val,
                      field.max_val, n_steps)
    results ← []

    for v in values:
        fv ← copy(fixed_values)
        fv[field.name] ← v
        results.append(
            compute_equilibrium(system, fv))

    return results, values
```

---

## 7. Spec-Layer Dataclasses (DESIGN §11.2)

```
dataclass FieldSpec:
    name:        str    # 'temperature', 'pressure', …
    symbol:      str    # 'T', 'P', …
    unit:        str    # 'K', 'atm', …
    min_val:     float
    max_val:     float
    initial_val: float
    extras:      dict   # default {}
        # Field-level constants consumed by energy
        # models at build time. Example for pressure:
        #   'R_gas':  float  (gas constant)
        #   'P_ref':  float  (reference pressure)

dataclass PhaseSpec:
    name:         str
    phase_type:   str   # 'gas'|'liquid'|'solid'|'end_member'
    xmin:         float # default 0.0
    xmax:         float # default 1.0
    model_type:   str   # 'HS'|'polynomial'|'piecewise_patch'
    model_params: dict  # default {}
        # All model-specific data. Keys by model_type:
        #
        # 'HS':
        #   H_coeffs:   list[float]  (ascending)
        #   S_coeffs:   list[float]  (ascending)
        #   V_coeffs:   list[float] | None
        #   ideal_gas:  bool
        #   vle_params: dict | None
        #     { liquid_phase: str,
        #       T_bp_A: float, T_bp_B: float,
        #       L_A: float, L_B: float }
        #
        # 'polynomial':
        #   poly_coeffs: list[list[float]]
        #   V_coeffs:    list[float] | None
        #   ideal_gas:   bool
        #
        # 'piecewise_patch':
        #   H_coeffs:          list[float]
        #   S_coeffs:          list[float]
        #   patch_left_phase:  str | None
        #   patch_left_x:      float | None
        #   patch_right_phase: str | None
        #   patch_right_x:     float | None

dataclass SystemSpec:
    title:       str
    components:  list[str]
    energy_form: str            # 'HS' | 'polynomial'
    fields:      list[FieldSpec]
    phases:      list[PhaseSpec]

# Contracts:
#  - FieldSpec mirrors Field 1:1, plus extras for
#    field-level constants energy models need.
#  - PhaseSpec replaces PhaseData; all model-specific
#    data lives in model_params (no flat per-model
#    fields).
#  - SystemSpec replaces SystemData; uses fields list
#    from day one (no hardwired T/P).
#  - Cross-phase refs (liquid_phase, patch_left_phase,
#    etc.) store phase *names*, resolved at build time
#    by topological sort (§7.2).
#  - VLE data lives inside the gas phase's
#    model_params['vle_params']; the liquid_phase key
#    triggers derivation via compute_vle_gas_hs.
```

---

## 7.2 Spec-Layer Construction (DESIGN §11.4)

```
function build_system_from_spec(spec: SystemSpec):
    # Topological sort: phases with no dependencies first,
    # then phases whose dependencies are already built.
    built ← {}                    # name → EnergyModel
    specs_by_name ← {ps.name: ps for ps in spec.phases}
    order ← topo_sort(spec.phases, dependency_edges)

    for ps in order:
        model ← ps.make_energy_model(
            specs_by_name, built, spec.fields)
        built[ps.name] ← model

    phases ← []
    for ps in spec.phases:
        phases.append(Phase(ps.name, ps.phase_type,
                            built[ps.name],
                            ps.xmin, ps.xmax))

    fields ← [Field(fs.name, fs.symbol, fs.unit,
                     fs.min_val, fs.max_val,
                     fs.initial_val)
               for fs in spec.fields]

    return System(spec.components, phases,
                  spec.energy_form, fields, spec.title)

function dependency_edges(phase_specs):
    edges ← []
    for ps in phase_specs:
        vle ← ps.model_params.get('vle_params')
        if vle is not None and vle.liquid_phase:
            edges.append(vle.liquid_phase → ps)
        if ps.model_params.get('patch_left_phase'):
            edges.append(
                ps.model_params['patch_left_phase'] → ps)
        if ps.model_params.get('patch_right_phase'):
            edges.append(
                ps.model_params['patch_right_phase'] → ps)
    return edges
```

---

## 7.3 make_energy_model Factory (DESIGN §11.3)

```
function PhaseSpec.make_energy_model(specs_by_name,
                                     built,
                                     field_specs):
    # Extract field-level constants from system fields.
    R_gas, P_ref ← pressure field extras (default 0.0, 1.0)
    T_ref ← temperature field initial_val (default 0.0)

    match self.model_type:

    case 'HS':
        vle ← model_params.get('vle_params')
        if vle is not None:
            # VLE gas: derive H/S from liquid phase.
            liq ← specs_by_name[vle.liquid_phase]
            H_gas, S_gas ← compute_vle_gas_hs(
                liq.H_coeffs, liq.S_coeffs,
                vle.T_bp_A, vle.T_bp_B,
                vle.L_A, vle.L_B)
            model ← HSModel(H_gas, S_gas,
                V, ideal_gas, R_gas, P_ref)
            model.vle_params ← boiling-point data
            return model
        # Plain HS: coefficients used directly.
        return HSModel(H_coeffs, S_coeffs,
            V, ideal_gas, R_gas, P_ref)

    case 'polynomial':
        return PolyModel(poly_coeffs,
            V, ideal_gas, R_gas, P_ref)

    case 'piecewise_patch':
        H_left, H_right ← None, None
        if patch_left_phase and patch_left_x:
            target ← built[patch_left_phase]
            H_left ← compute_left_patch_H(
                H, S, target.H, target.S,
                xmin, patch_left_x, T_ref)
        if patch_right_phase and patch_right_x:
            target ← built[patch_right_phase]
            H_right ← compute_right_patch_H(
                H, S, target.H, target.S,
                xmax, patch_right_x, T_ref)
        return PiecewisePatchModel(H, S,
            H_left, patch_left_x,
            H_right, patch_right_x,
            V, ideal_gas, R_gas, P_ref)

    else: error "Unknown model_type"
```
