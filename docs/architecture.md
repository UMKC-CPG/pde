# PDE Architecture Design

*Working document — updated as design decisions are made.*

---

## 1. Vision and Guiding Principles

PDE (Phase Diagram Explorer) serves a **dual mission**:

- **Pedagogical** — a self-explanatory tool for introducing students to phase
  diagrams, G(x) energy curves, and thermodynamic reasoning.  The UI must make
  the physics visible and legible without hiding complexity behind automation.
- **Research** — a flexible platform for exploring phase behaviour in systems
  beyond standard T-P-x thermodynamics, including non-classical thermodynamic
  parameters (magnetic field, electric field) and non-physical analogues
  (statistical-physics models of economic or population systems).

These two missions are complementary, not competing.  The design goal is a
system that is *easy to use in the common case* (standard T-P-x binary phase
diagrams) and *extensible to the uncommon case* without requiring a rewrite.

**Guiding principles derived from that vision:**

1. **T and P are not special.**  They are two instances of a general concept —
   a sweepable thermodynamic field.  The architecture must not hardcode their
   presence or absence.

2. **The convex-hull computation is domain-agnostic.**  As long as the user can
   supply G(x, {fields}) for each phase, the common-tangent / phase-coexistence
   machinery works identically regardless of what the fields represent.

3. **Additions should be additive, not invasive.**  A new field type or a new
   energy coupling term should be expressible by adding data/configuration, not
   by scattering new `if has_field_X` branches through existing code.

4. **Pedagogy first, then power.**  Named presets (e.g. "standard
   thermodynamics", "VLE with pressure") should cover the common cases
   effortlessly.  The full generality is available underneath but not forced
   on the user.

---

## 2. Current Architecture — What Works, What Doesn't

### What works well

- `EnergyModel` ABC with `gibbs(x, T, P=0.0)` — clean, array-friendly.
- `Phase` / `System` dataclasses — simple and easy to reason about.
- Convex-hull computation in `pde_compute` — completely domain-agnostic today
  (it only sees arrays of x and G values).
- `pde_builder` `SystemData`/`PhaseData` round-trip model.

### Brittleness sources

- `System` carries `has_pressure`, `P_min`, `P_max`, `P_initial`, `P_unit`,
  `R_gas`, `P_ref` as flat attributes.  Every new field would add more flat
  attributes.
- `pde_viz` contains numerous `if system.has_pressure:` branches that will
  metastasize as more optional fields are added.
- `EnergyModel.gibbs(x, T, P=0.0)` — T and P are positional; adding H, E, or
  custom fields requires changing every call site.
- `SystemData`/`PhaseData` in `pde_builder` mirror `System`/`Phase` with
  pressure-specific attributes baked in.
- The XML schema uses dedicated `<temperature>` and `<pressure>` blocks;
  there is no generic mechanism for other fields.

---

## 3. The Field Abstraction

### 3.1 What a Field is

A **Field** is an intensive thermodynamic parameter (or its analogue) that:

- Has a numeric range the user can sweep interactively.
- Appears in the energy function of one or more phases via a coupling term.
- Can serve as an axis in a phase diagram.

Standard examples:

| Field | Symbol | Conjugate extensive quantity |
|-------|--------|------------------------------|
| Temperature | T | −Entropy (−S) |
| Pressure | P | Volume (V) |
| Magnetic field strength | H | −Magnetisation (−M) |
| Electric field strength | E | −Polarisation (−P_el) |
| "Resource pressure" (economic model) | μ_r | "Utility density" |

### 3.2 Proposed Field data model

```python
@dataclass
class Field:
    name:        str    # unique identifier, e.g. 'temperature', 'pressure'
    symbol:      str    # display symbol, e.g. 'T', 'P', 'H'
    unit:        str    # display unit string, e.g. 'K', 'atm', 'T', ''
    min_val:     float
    max_val:     float
    initial_val: float
```

**Role is not stored on the Field.**  Which field is the *primary sweep axis*
(the one that generates the phase diagram at fixed values of the others) and
which is the *secondary axis* (giving the 3-D T-P-x-style space) is a property
of the current *view configuration*, not of the field itself.  A system with
fields T and H would allow the user to sweep T at fixed H, or sweep H at fixed T.

### 3.3 System with Fields

`System` loses its flat pressure attributes and gains a `fields` list:

```python
class System:
    components:  list[str]
    phases:      list[Phase]
    energy_form: str            # 'HS' | 'polynomial' | 'general' (future)
    fields:      list[Field]    # ordered; fields[0] is T by convention
    title:       str
```

Backward-compatible convenience properties on `System`:

```python
@property
def T_field(self) -> Field:      return self.fields[0]
@property
def T_min(self) -> float:        return self.fields[0].min_val
@property
def T_max(self) -> float:        return self.fields[0].max_val
@property
def T_initial(self) -> float:    return self.fields[0].initial_val
@property
def has_pressure(self) -> bool:  return any(f.name == 'pressure' for f in self.fields)
# ... etc.
```

This makes the migration from old to new code incremental — existing code that
reads `system.T_min` or `system.has_pressure` continues to work.

---

## 4. The Generalised Energy Model

### 4.1 Mathematical structure

The Gibbs energy of a phase is decomposed as:

```
G(x, {λ}) = G_base(x)  +  Σᵢ  Rᵢ(x) · fᵢ({λ})
```

where:

- `G_base(x)` — the "intrinsic" composition-dependent energy (e.g. H(x) in the
  HS model).
- `Rᵢ(x)` — a **response function**: a polynomial in composition x representing
  the conjugate extensive quantity (e.g. −S(x), V(x), −M(x)).
- `fᵢ({λ})` — a **field function**: a scalar function of one or more field
  values (e.g. T, P, T·ln(P/P_ref)).

### 4.2 Standard couplings

| Name | Rᵢ(x) | fᵢ({λ}) | Fields used |
|------|--------|---------|-------------|
| Entropic | −S(x) | T | T |
| Poynting (PV) | V(x) | P | P |
| Ideal gas | R_gas | T · ln(P/P_ref) | T, P |
| Magnetic | −M(x) | H | H |
| Electric | −P_el(x) | E | E |
| General linear | R(x) | λ | any single field |

Note that the **ideal gas term** is a cross-coupling: its field function depends
on *both* T and P simultaneously.  The decomposition must therefore allow
`fᵢ` to be a function of a *set* of fields, not just a single field.

### 4.3 Proposed CouplingTerm data model

```python
@dataclass
class CouplingTerm:
    response_coeffs: list[float]   # R(x) polynomial, ascending order
    coupling_type:   str           # 'linear' | 'ideal_gas' | 'power' | 'custom'
    field_names:     list[str]     # which Field.name values this term uses
    params:          dict          # extra parameters, e.g. {'P_ref': 1.0}
```

**Coupling types** (initially supported):

- `'linear'`  — `f = λ` (single field).  Covers −S·T, V·P, −M·H, −P_el·E.
- `'ideal_gas'` — `f = T · ln(P / params['P_ref'])`.  Covers the ideal-gas
  chemical-potential term.  Fields: `['temperature', 'pressure']`.
- `'power'`   — `f = λⱼ` for a single field to a given power.  Covers the
  PolyModel T-coefficients (each term is xⁱ·Tʲ).
- `'custom'`  — arbitrary Python callable; for research use.  Validation and
  UI support are explicitly out of scope until the rest of the framework is
  stable.

### 4.4 Generalised EnergyModel interface

```python
class EnergyModel(ABC):
    @abstractmethod
    def gibbs(self, x, field_values: dict[str, float]) -> np.ndarray:
        """Return G(x, {fields}) as a numpy array."""
```

The current positional `(x, T, P=0.0)` signature is replaced by a dict.
A **shim** ensures backward compatibility during the migration:

```python
def gibbs(self, x, T=None, P=0.0, field_values=None, **kwargs):
    if field_values is None:
        field_values = {'temperature': float(T), 'pressure': float(P)}
        field_values.update(kwargs)
    return self._gibbs_impl(x, field_values)
```

### 4.5 Current models as special cases

**HSModel** becomes an `EnergyModel` with:
```
G_base  = H(x)  (H_coeffs polynomial)
term[0] = CouplingTerm(response=-S_coeffs, type='linear', fields=['temperature'])
term[1] = CouplingTerm(response=V_coeffs,  type='linear', fields=['pressure'])    # optional
term[2] = CouplingTerm(response=[R_gas],   type='ideal_gas', fields=[...])        # optional
```

**PolyModel** becomes an `EnergyModel` with one `'power'` coupling term per
(x-power, T-power) pair:
```
G = Σᵢⱼ aᵢⱼ · xⁱ · Tʲ
```
Each term: `response=[0,…,0,1]` (monomial xⁱ), `type='power'`, `fields=['temperature']`,
`params={'power': j}`.

Both can be implemented as thin wrappers around the existing numpy code with no
change to the numerical core.

---

## 5. XML Schema Evolution

### 5.1 Generalised fields block

Replace the dedicated `<temperature>` and `<pressure>` top-level blocks with a
general `<fields>` section:

```xml
<fields>
  <field name="temperature" symbol="T" unit="K"
         min="250" max="500" initial="400"/>
  <field name="pressure" symbol="P" unit="atm"
         min="0.5" max="5.0" initial="1.0"/>
</fields>
```

### 5.2 Generalised coupling terms in phase energy

The HS `<H>` / `<S>` / `<V>` shorthand is kept as a **recognised abbreviation**
for backward compatibility.  New systems can use explicit coupling terms:

```xml
<energy>
  <base>
    <poly x0="0.5" x1="-1.0" x2="1.0"/>   <!-- G_base = H(x) -->
  </base>
  <coupling field="temperature" type="linear">
    <response x0="-0.005"/>                <!-- -S(x) -->
  </coupling>
  <coupling field="pressure" type="linear">
    <response x0="0.001"/>                 <!-- V(x) -->
  </coupling>
  <coupling fields="temperature pressure" type="ideal_gas" P_ref="1.0">
    <response x0="8.314e-3"/>              <!-- R_gas -->
  </coupling>
</energy>
```

A magnetic example:

```xml
<fields>
  <field name="temperature"     symbol="T" unit="K"  min="0"   max="1000" initial="300"/>
  <field name="magnetic_field"  symbol="H" unit="T"  min="0"   max="10"   initial="0"/>
</fields>

<!-- inside a phase: -->
<coupling field="magnetic_field" type="linear">
  <response x0="-0.002" x1="0.001"/>      <!-- -M(x) -->
</coupling>
```

### 5.3 Backward compatibility

The parser recognises the legacy schema (top-level `<temperature>` and
`<pressure>` blocks, `<H>`, `<S>`, `<V>` tags inside `<energy>`) and maps it
to the generalised data model transparently.  Old input files continue to work
without modification.

---

## 6. Visualisation Architecture

### 6.1 Field-agnostic canvases

The current `TxCanvas` and `PxCanvas` are replaced by a single **`SweepCanvas`**
parameterised by:

- `primary_field`: the `Field` whose range forms the vertical axis.
- `fixed_fields`: a dict `{field_name: value}` for all other fields held constant.

The "Fixed P (T-x)" / "Fixed T (P-x)" mode selector generalises to:
*"which field is the primary sweep axis?"*  With only one non-composition field
(T), no selector appears.  With two fields (T and P, or T and H, etc.), the
selector lists both options.  With three or more fields, additional sliders
appear for the non-sweep fields.

### 6.2 Slider generalisation

Each field in `system.fields` beyond composition gets:
- A labelled slider with min/max/initial from the `Field`.
- A live value label showing `{symbol} = {value} {unit}`.

The primary-sweep slider drives O(1) lookups in the pre-computed grid; secondary
(and additional) sliders trigger recomputation.

### 6.3 G(x) canvas

`GxCanvas.redraw()` calls `phase.gibbs(x, field_values)` where `field_values`
is built from the current state of all sliders.  No field-specific branches are
needed.

### 6.4 3-D view

`Viz3DWindow` generalises to any two-field system.  The axis convention becomes:

```
x-axis: composition
y-axis: primary field (currently T)
z-axis: secondary field (currently P)
```

Labels and units are taken from the `Field` objects.

---

## 7. Multi-Component Path

The composition variable `x` currently is a scalar (binary system).  Ternary
and higher systems require `x` to be a vector on the composition simplex.

Key changes required (deferred, but the design must not prevent them):

- `Phase.gibbs(x, field_values)` — `x` becomes an ndarray of shape `(n_components-1,)`
  or a row of a `(n_points, n_components-1)` grid.
- `pde_compute` — `scipy.spatial.ConvexHull` generalises naturally to
  `(x₁, x₂, …, G)` in `n_components + 1` dimensions.  The "lower hull" criterion
  remains unchanged.
- Visualisation — ternary diagrams (Gibbs triangle) for `GxCanvas`; isothermal
  sections, liquidus projections for `SweepCanvas`.
- `EnergyModel` — response functions become polynomials in multiple composition
  variables (e.g. Redlich-Kister / Legendre expansions on the simplex).

The `Field` generalisation is **independent** of the multi-component
generalisation; both affect `EnergyModel.gibbs()`, and they must be designed
together at the interface level to avoid a second round of breaking changes.

**Target interface:**
```python
def gibbs(self, x: np.ndarray, field_values: dict[str, float]) -> np.ndarray:
    ...
```
`x` is already a numpy array today; the only change is that it becomes 2-D for
multi-component systems.  The `field_values` dict is the field generalisation.
Together these two changes cover both generalisations with a single interface
revision.

---

## 8. CALPHAD Integration

CALPHAD databases (TDB files) provide assessed thermodynamic data for
multicomponent systems using:

- **Redlich-Kister polynomials** for excess Gibbs energy.
- **Sublattice models** for ordered phases (e.g. intermetallics).
- **Magnetic contributions** (Hillert-Jarl model near the Curie temperature) —
  a non-polynomial coupling term that is a concrete example of the `'custom'`
  coupling type described in §4.3.

The **`pycalphad`** library (open source) handles TDB parsing and phase
equilibrium calculations.  The natural integration point is a `CALPHADModel`
subclass of `EnergyModel` that wraps a `pycalphad` phase object:

```python
class CALPHADModel(EnergyModel):
    def __init__(self, pycalphad_phase, ...): ...
    def gibbs(self, x, field_values): ...   # delegates to pycalphad
```

This requires no changes to `pde_compute` or the visualisation layer — only a
new `EnergyModel` subclass and a TDB import path in `pde_input`.

*CALPHAD integration is a medium-term goal.  The design must not preclude it,
but implementation is deferred until the Field generalisation is stable.*

---

## 9. Persistence and Archiving

Pre-computed grids (the full T-P-x data produced by `FullGridWorker`) are
currently held only in memory.  For pedagogical use (distributing completed
calculations to students) and for expensive multi-component / CALPHAD
computations, these must be serialisable to disk.

Proposed format: **HDF5 + XDMF** (already the planned output format per the
project overview).  The `PhaseDiagram3D.to_xdmf()` stub is the intended entry
point.

A saved session file should contain:
1. The system definition (reproduced as the XML input or its generalised
   equivalent).
2. The pre-computed grid: `G` values, phase indices, and region boundaries at
   each `(T, P, …)` grid point.
3. Metadata: PDE version, computation timestamp, field ranges and resolutions.

Loading a saved session skips all computation and opens the visualisation
directly from the stored grid.

---

## 10. Thermodynamic Consistency Checking

Rather than enforcing individual constraints (e.g. the VLE slope-matching
condition discussed during design) as one-off fixes in the builder, a dedicated
**consistency checker** layer is planned.

Design principles:

- **Explain, don't just flag.**  Each violation is accompanied by a description
  of the physical meaning and, where possible, a suggestion for correction.
  This serves the pedagogical mission.
- **Non-blocking.**  Violations are warnings, not errors.  The user can proceed
  with an inconsistent system and observe the artefacts (finite two-phase regions
  at pure-component compositions, etc.).
- **Composable.**  Each check is a standalone function:
  ```python
  def check_vle_terminal_tangency(system) -> list[ConsistencyWarning]: ...
  def check_convexity(system, T, P) -> list[ConsistencyWarning]: ...
  ```
  New checks can be added without touching existing ones.
- **Integrated into the builder.**  Warnings are displayed in a status area
  within `BuilderWindow` after each Apply.

Known checks to implement:

| Check | Condition |
|-------|-----------|
| VLE terminal tangency | At each pure-component boiling point, G^liquid and G^vapor must be tangent (G-match *and* slope-match). |
| End-member G-match | At each melting/transition temperature, G^phase(x_end) must equal G^end_member. |
| Convexity | d²G/dx² > 0 for each phase over its composition range at the initial T. |
| Phase coverage | The composition domain [0, 1] should be covered by at least one phase at all T. |

---

## 11. Migration Roadmap

The changes above are substantial.  The migration is designed to be
**incremental and non-breaking** at each step.

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | This design document | Done |
| 1 | **Field generalisation — data model** | Next |
|   | Replace `System` flat pressure attrs with `fields: list[Field]` |  |
|   | Add backward-compatible properties (`has_pressure`, `T_min`, etc.) |  |
|   | Update `pde_input` parser; update `pde_builder` `SystemData` |  |
|   | Update XML schema; keep legacy schema working |  |
| 2 | **EnergyModel interface** | Follows Phase 1 |
|   | Generalise `gibbs(x, T, P=0.0)` → `gibbs(x, field_values)` |  |
|   | Add shim for backward compatibility |  |
|   | Refactor `HSModel` and `PolyModel` to use `CouplingTerm` internally |  |
| 3 | **Visualisation generalisation** | Follows Phase 2 |
|   | Replace `TxCanvas`/`PxCanvas` with `SweepCanvas(primary_field)` |  |
|   | Replace `has_pressure` branches with field-count logic |  |
|   | Generalise sliders to be field-driven |  |
| 4 | **Consistency checker** | Parallel with Phase 3 |
|   | `pde_check.py` module; surface warnings in builder |  |
| 5 | **Multi-component** | After Phase 3 |
|   | Generalise composition to vector; update convex hull; add ternary canvas |  |
| 6 | **Persistence** | After Phase 3 |
|   | HDF5+XDMF save/load for pre-computed grids |  |
| 7 | **CALPHAD** | After Phase 5 |
|   | `CALPHADModel` wrapping `pycalphad`; TDB import in `pde_input` |  |

Each phase ends with a working, tested program.  No phase leaves the code in
a broken intermediate state.
