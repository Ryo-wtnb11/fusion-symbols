# SO(N)/Sp(2N) decomposition-sweep gauge specification

This document specifies the **gauge** of the SO(N)/Sp(2N) Clebsch–Gordan
isometries and projected generators produced by `LieGroup`'s internal B/C/D decomposition
sweep (`src/lie/bcd/sweep.rs`) — the deterministic rules that fix the
otherwise-free basis of each coupled multiplet.

## Status: FROZEN NORMATIVE SPECIFICATION

**This document is the authority; the code is an implementation of it.** The
rules below define the B/C/D coefficient values: a build that returns a value
contradicting one of them has a bug, and the fix is to restore the specified
value rather than to redefine the gauge. Every discrete choice that can move a
returned value (seed rule, operator-application order, Gram–Schmidt order, QR
sign convention, weight-sort and its tie-break, block sign convention,
outer-multiplicity assignment, canonical-parent order, intertwiner alignment, and
the value-fixing tolerance tier) is stated here.

`FusionCategory::authority_fingerprint()`'s `epoch` tag is **the version of this
specification**, not a label for the current build's output: it moves only when
this document is corrected. Changing a value therefore requires the four-step
specification-correction rule — spec edit, epoch bump, CHANGELOG breaking-change
entry, regenerated golden values — declared in
[`gauge.md`, "Status"](gauge.md#status-frozen-normative-specification), which
governs all three families. `tests/gauge_golden.rs` pins signed SO(5) and Sp(4)
coefficients as the in-repo tripwire (`src/lie/bcd/intertwiner_tests.rs` checks the
generators and the isotypic projectors, which are by construction blind to
exactly the coupled-side gauge this document fixes).

Reference: QSpace v4 revision `dd2cc7e`,
`Source/clebsch_aux.cc::getSymmetryStates` (53–348), `findMaxWeight`
(957–1045), and `Source/clebsch.cc` (6649–6656). Coefficients are `f64`;
label combinatorics, ordering and discrete gauge choices are exact.

Two deliberate fusion-symbols deviations from QSpace, called out once here and again at
their sites:

1. **PositiveDiagonal QR gauge** (§4a): the column orthonormalization uses
   the **positive-diagonal** QR gauge, a *tightening* of QSpace's
   unspecified `OrthoNormalizeColsQR` sign, fixing the sign of each retained
   orthonormal direction deterministically.
2. **Unconditional block sign convention** (§8): fusion-symbols applies
   `rangeSignConvention` to every block; QSpace applies it only when the
   weight-sort permutation is nontrivial (the `:304` call sits inside
   `if (!P.isIdentityPerm())`, `clebsch_aux.cc:297-305 @ dd2cc7e`).

Both are value-affecting gauge choices and therefore part of this contract.

## 0. Notation

- A B/C/D group is one of `B_r = SO(2r+1)`, `C_r = Sp(2r)`, `D_r = SO(2r)`; `r`
  is the rank. It carries `r` simple-root **raising** operators `Sp[i]`
  (`i = 0…r-1`) and `r` mutually Frobenius-orthogonal **Cartan**
  operators `Sz[j]` (`j = 0…r-1`), in QSpace's basis (not Chevalley) — see
  `src/lie/bcd/seeds.rs` and its gauge notes. Their eigenvalues are integers
  for tensor irreps and half-integers for B/D spinor irreps (§6, §16.2).
  Seed records are integers; `src/lie/bcd/sweep.rs::Generators::from_seed`
  applies their exact Cartan scale (1 for defining seeds, 1/2 for spinor seeds).
- `Sp[i]†` (conjugate transpose; real here, so plain transpose) is the
  corresponding **lowering** operator.
- An irrep `c` is labelled by its integer Dynkin label `a = (a_0,…,a_{r-1})`,
  matching `LieGroup::irrep` (whose label↔partition maps are pinned in
  `src/lie/bcd.rs`).
- The bootstrap works in the **cover** `Spin(2r+1)` / `Sp(2r)` / `Spin(2r)`, so
  the label set is the full dominant weight lattice: the **tensor** class
  (integer ε-basis `λ`) and, for `B`/`D`, the **spinor** class (half-integer
  `λ`). Weights are carried doubled (`2λ`, integral in both classes,
  the crate's `dj = 2j` convention). A *global form* — which class a caller may
  ask for — is a restriction of the label set applied at construction
  (`crate::lie`); it never changes a coefficient value.
- The CGC of a coupled multiplet is the isometry `V`, a real `d1·d2 × d3`
  column-major matrix; each column is one coupled state.

## 1. Input and the Kronecker convention (gauge)

The sweep takes the **product generators** of `a ⊗ b`, built by the caller from
the two irreps' generator sets via the generator tensor product (QSpace
`clebsch.cc:6649-6656 @ dd2cc7e`):

```
Sp_prod[i] = Sp_a[i] ⊗ 1_b + 1_a ⊗ Sp_b[i]      (i = 0…r-1)
Sz_prod[j] = Sz_a[j] ⊗ 1_b + 1_a ⊗ Sz_b[j]
```

**Kronecker convention — gauge, pinned here.** The product basis index of
`|m_a, m_b⟩` is

$$ \mathrm{composite}(m_a, m_b) = m_a + d_a \cdot m_b $$

i.e. the **first** factor (`a`) is the *fast* (least-significant) index and the
second (`b`) the slow index. This matches QSpace's
`wbsparray::setRec_kron` (`q[i] = i1[i] + SIZE_a·i2[i]`, `wbsparray.cc:3210 @
dd2cc7e`), which is the **reverse** of the textbook `kron(A,B)` (first factor
slow). A different convention permutes the CGC rows and is a different gauge.

The product `Sz_prod[j]` stays diagonal (a sum of a diagonal ⊗ identity terms);
its diagonal is `Sz_a[j][m_a] + Sz_b[j][m_b]` at the composite index.

Generator-count guards (QSpace `clebsch_aux.cc:85,90 @ dd2cc7e`): `nz ≠ 0`,
`np ≤ nz`, and `np = nz = r` — else `InvalidGeneratorCounts` generation diagnostic.

## 2. Seed selection (per multiplet)

Reference: `clebsch_aux.cc:105-116 @ dd2cc7e`. The sweep tiles the `D = d1·d2`
product space with multiplets `it = 0, 1, …`. The seed index `i0` is
**persistent across multiplets** (it never resets): the seed is the **lowest
product-basis index `i0` not already in the span of the accumulated basis `U`**.
Concretely, walking `i0` upward, `e_{i0}` is skipped iff `‖Uᵀ e_{i0}‖ = 1`
(within `EPS_SWEEP`), i.e. it lies in `span(U)` (`U` has orthonormal columns);
the first `i0` failing that test is the seed. The very first seed (`it = 0`,
`i0 = 0`) is `e_0` unconditionally.

For `it > 0` the seed is orthogonalized against `U` by **two** Gram–Schmidt
passes with renormalization (`clebsch_aux.cc:118-123 @ dd2cc7e`):
`v ← v − U(Uᵀv)`, normalize; repeat once.

The seed must be a simultaneous `Sz`-eigenvector (a definite-weight vector); if
some `Sz[j] v` is not parallel to `v` (and non-negligible), the labels are
ambiguous — `SeedNotWeightVector` generation diagnostic (QSpace `sameUptoFac` guard,
`clebsch_aux.cc:126-129 @ dd2cc7e`; QSpace's `sameUptoFac` returns 0 when the
vectors *are* parallel, so the abort fires when they are **not**).

## 3. Raise to the maximum weight

Reference: `clebsch_aux.cc:134-142 @ dd2cc7e`. Repeatedly apply the raising
operators in **ascending index order** `Sp[0], Sp[1], …`; whenever
`Sp[i] v` is non-negligible (`‖·‖ > EPS_SWEEP`), replace `v` by the normalized
result. Iterate until a full pass raises nothing. The result is the multiplet's
highest-weight (MW) vector.

## 4. Sweep down (the descent)

Reference: `clebsch_aux.cc:147-224 @ dd2cc7e`. From the MW vector, generate the
whole multiplet by lowering, level by level (constant weight-height per level).

Notation for one multiplet: `V` = states found so far (this multiplet, all
completed levels, MW first), `Vi` = new states of the **current** level, `U` =
globally accumulated basis of all previous multiplets. The lowering **frontier**
is the previous level (initially the MW).

For each level, loop the lowering operators in **ascending index order**
`Sp[i]†`; for each `i`:

1. `vi ← Sp[i]† · frontier` (lower the whole frontier block at once). Skip this
   operator if the block's per-column RMS `√(‖vi‖²/cols)` is `< EPS_SWEEP`
   (QSpace `sqrt(vi2/SIZE[1]) < eps`, `:154-158`).
2. **Gram–Schmidt pass 1, order self → pass → global** (`:161-196`):
   - project out `Vi` (this level): `vi ← vi − Vi(Viᵀ vi)`;
   - guard: the residual overlap with `V` must be `≤ EPS_SWEEP`
     (`OverlapWithVspace` generation diagnostic), then project out `V`;
   - guard: the residual max-overlap with `U` must be `≤ EPS_SWEEP`
     (`OverlapWithUspace` generation diagnostic), then project out `U`.
   The `V`/`U` guards are near-zero for a correct sweep (lowering stays within
   the multiplet, orthogonal to higher levels and to complete earlier
   multiplets); a large overlap signals a defective generator set.
3. Drop columns with norm `< EPS_SWEEP` (QSpace `SkipTinyCols`, `:198`), then
   **QR-orthonormalize** the survivors (`OrthoNormalizeColsQR(FL, CG_EPS1)`,
   `:200`) — see §4a.
4. **Gram–Schmidt pass 2, order global → pass → self** (`:206-217`): project the
   QR result out of `U`, then `V`, then `Vi`, and **QR again**. (Note the order
   is the reverse of pass 1; this is a faithful port of the QSpace ordering, not
   a symmetrization.)
5. Append the result to `Vi`.

After the operator loop, if any new states were found, append `Vi` to `V`, set
the frontier to `Vi`, and continue to the next level; otherwise the multiplet is
complete. The multiplet is then appended to `U`. Bounds `V,Vi,U ≤ D` are guarded
(`SpaceOutOfBounds` generation diagnostic, `:210,215,235`). The sweep ends when `U` fills
`D`; failing to fill `D` is `IncompleteDecomposition` generation diagnostic (`:236`).

### 4a. QR gauge — PositiveDiagonal with in-order column selection

`src/lie/bcd/linalg.rs::qr_positive_q` walks the input columns in order, maintaining
an orthonormal selection basis `Q_S`. For each column, subtract
`Q_S (Q_Sᵀ a_j)` twice (CGS2: one re-orthogonalization pass), using the
sequential backend matmul. Retain the column exactly when the residual norm
is strictly greater than the absolute tolerance `CG_EPS1 = 1e-10`, and append
the normalized residual to `Q_S`. Zero and dependent columns are skipped,
including leading zero columns and non-trailing dependencies.

The selection basis is used only to choose indices `S`. The output is the
thin unpivoted QR of the original selected columns `a[:, S]`, with each
R diagonal made positive by folding its sign into Q. These columns have full
column rank by construction, so the positive-diagonal QR is unique and
independent of backend Householder choices. Empty selection returns zero columns.
This is the deliberate fusion-symbols tightening of QSpace's unspecified
`OrthoNormalizeColsQR` sign convention.

*The defect this corrects (`epoch = 2` → `3`).* The old rule
factored all input columns, made the R diagonal nonnegative (leaving exact
zero unflipped), and retained Q columns whose R-row norm exceeded `CG_EPS1`.
A dependent middle column can shift an independent direction off the diagonal;
then a zero or round-off diagonal does not fix that direction's sign. The
assertion that descent dependencies were always trailing was false.

The correction is pinned by
`tests/gauge_golden.rs::d3_epoch3_independent_column_qr_golden` and the independent
rule check `src/lie/bcd/linalg.rs::tests::d3_epoch3_qr_uses_independent_input_columns`.

The documented projection order and second orthonormalization pass preserve
robustness; on converged orthonormal subspaces they affect only round-off.

## 5. Global orthogonality gate

Reference: `clebsch_aux.cc:251-257 @ dd2cc7e`. After the sweep, `UᵀU` must equal
the identity within `EPS_VERIFY` (QSpace `isIdentityMatrix(eps2)`); else
`NotOrthonormal` generation diagnostic.

## 6. Generator projection and Cartan snapping

Reference: `clebsch_aux.cc:264-283 @ dd2cc7e`. For each multiplet with isometry
`V`, project the product generators:

```
R.Sp[i] = Vᵀ (Sp_prod[i] V),      R.Sz[j] = Vᵀ (Sz_prod[j] V).
```

Each `R.Sz[j]` must be **diagonal** within `EPS_VERIFY` (QSpace
`isDiagMatrix(eps2)`, `:274`) — else `NonDiagonalCartan` generation diagnostic. Its
diagonal entries are the states' Cartan eigenvalues, and are snapped to the
nearest point of the carrier's **weight grid** (**FixRational**, QSpace
`FixRational(...,4)`, `:282`). A value farther than `FIXRATIONAL_TOL` from a
grid point is `NonIntegerWeight` generation diagnostic.

The grid is `1/denom` with `denom = 1` for a carrier built entirely from tensor
irreps — each column of `V` is then a definite-weight state with an **integer**
eigenvalue, exactly as before — and `denom = 2` once a spinor factor is present,
whose Cartan eigenvalues are half-integers (§16). `denom` is a property of the
generator set: `1` for the trivial and defining seeds, `2` for a spinor seed,
and the maximum of the two factors' for a product. A pure-tensor product
therefore keeps the strict integer gate unchanged. The `d3 × r` matrix of snapped
eigenvalues is the block's weight table `Z` (row = state, column = Cartan op).

The projected generators are gated by the commutator relations of the defining seeds (`src/lie/bcd/seeds.rs`), evaluated
in `f64` (the projected `Sp` are generally irrational, so the exact integer
`check_commutators` does not apply directly): `[Sz_j, Sp_i] = d_{i,j} Sp_i` and
`[Sp_i, Sp_i†] ∈ span(Sz)`, worst residual `≤ EPS_SWEEP` — else
`CommutatorResidual` generation diagnostic.

## 7. findMaxWeight: sort, tie-break, Dynkin conversion, uniqueness

Reference: `clebsch_aux.cc:957-1045 @ dd2cc7e`.

**Descending-weight sort (gauge).** The states are permuted into descending
weight order. The comparison is lexicographic on the Cartan columns read in
**reversed** order (column `r-1` first, …, column `0` last), descending (QSpace
`z2.FlipCols(); z2.sortRecs_float(P,-1)`, `:969-970`). **Tie-break:** states with
identical weight rows keep ascending original basis-index order (a stable,
deterministic total order). The max-weight state is the first after sorting.

The whole block — `V` columns, `Z` rows, and each `R.Sp`/`R.Sz` — is reordered by
this permutation (`clebsch_aux.cc:295-301 @ dd2cc7e`).

**Max-weight uniqueness (gate).** The top two sorted rows must differ:
`‖Z[k0,·] − Z[k1,·]‖² > EPS_MW_UNIQUE` — else `MaxWeightNotUnique` generation diagnostic
(QSpace `recDiff2(0,i) > 1e-8`, `:1035-1039`).

**Dynkin conversion (per series).** The max-weight state's Cartan eigenvalues
`qm` (in QSpace's `Sz` basis) map to the Dynkin label as follows
(`clebsch_aux.cc:977-1031 @ dd2cc7e`; each division must yield an integer within
`FIXRATIONAL_TOL`, else `NonIntegerWeight` generation diagnostic):

- **`C_r` (SpN, `:990-996`):** `a_i = (qm_i − qm_{i-1})/(i+1)` for
  `i = r-1…1`, and `a_0 = qm_0`.
- **`B_r` (SON, `:1001-1013`):** with `x = 2·qm_0`, set `a_{i-1} = qm_i − qm_{i-1}`
  for `i = 1…r-1`, then `a_{r-1} = x`, then reverse-swap `a_i ↔ a_{r-2-i}` for
  `i = 0…⌊(r-1)/2⌋-1`.
- **`D_r` (SEN, `:1018-1031`):** as `B_r` but with `x = qm_0 + qm_1`.

The resulting Dynkin label constructs `LieGroup::irrep` **in the
cover** (`GlobalForm::SimplyConnected`); an invalid label is
`InvalidDiscoveredLabel` generation diagnostic (unreachable for a faithful sweep). The
`B_r` conversion's `x = 2·qm_0` and the `D_r` conversion's `x = qm_0 + qm_1`
already return an integer `a_{r-1}` from half-integer eigenvalues, which is
exactly the spinor label; nothing in this section changes for the spinor class,
and a product of tensor irreps only ever discovers tensor labels.

QSpace's `r > 9` upper-rank guard (`:983,993,1004,1021`) is a fixed-buffer build
artifact, **not** a mathematical constraint — **N/A** here (same disposition as
`src/lie/bcd/seeds.rs`). The low-rank `r < 2` (`r ≤ 2` for `D`) redirect is inherited
from the seed layer (`ExcludedRank` generation diagnostic), so the sweep never receives such
generators.

## 8. Sign convention

Reference: `signFirstVal`/`rangeSignConvention` (`clebsch_aux.cc:26-51 @
dd2cc7e`). The **whole block's** sign is fixed so that the **first significant**
CGC entry (first `|·| > CG_EPS1` scanning the flattened, sorted `V` in storage
order) is **positive**; if it is negative, negate the whole block. This is
`rangeSignConvention` on `V.D`. The CGC entries are then integer-snapped where
they land on integers (FixRational, `:307`); genuinely irrational entries are
left as-is.

Unlike QSpace's conditional call inside `if (!P.isIdentityPerm())`
(`clebsch_aux.cc:297-305`), this block-sign rule applies unconditionally,
including blocks already in descending-weight order.

## 9. Outer-multiplicity assignment

Reference: `clebsch_aux.cc:331-345 @ dd2cc7e`. Blocks that share the same highest
weight (same discovered irrep) receive `(index, size)`: `index = 0, 1, …` in
**discovery (sweep) order**, and `size` = the number of such blocks. A
multiplicity-free irrep gets `(0, 1)`.

## 10. Exact-multiplicity production gate

For every discovered irrep `c`, the sweep multiplicity `M^c_sweep` (the number of
blocks labelled `c`) must equal the exact fusion multiplicity `N^c_ab` from
the exact B/C/D product decomposition, **and** the discovered support must equal the exact
support — **both directions**. A missing block (`found = 0 < N`) is as fatal as
a spurious one (`found > 0 = N`). Any mismatch is
`MultiplicityMismatch` generation diagnostic. This gate is on the production path; it is not optional or test-only. The internal decomposition operation takes the
exact decomposition `expected` and enforces it; its defining-product specialization
computes `expected` from the defining label via `directproduct`.

This is fusion-symbols' addition over QSpace, which has no exact multiplicity gate (it only
warns on discovered outer multiplicity, `:341`).

## 11. Tolerance tier (QSpace CG_EPS ladder)

Named constants (`src/lie/bcd/sweep.rs`), with provenance:

| Constant | Value | Source (`@ dd2cc7e`) | Role |
|---|---|---|---|
| `EPS_SWEEP` | `1e-8` | `getSymmetryStates` `eps` (`clebsch_aux.cc:76`) | significant-vector / overlap threshold in the sweep and commutator gate |
| `EPS_VERIFY` | `1e-10` | `getSymmetryStates` `eps2` (`clebsch_aux.cc:76`) | `UᵀU = I` and Cartan-diagonality checks (`isIdentityMatrix`/`isDiagMatrix(eps2)`) |
| `CG_EPS1` | `1e-10` | `clebsch.hh:244` (non-MPFR tier) | QR orthonormalization / rank-reveal (`OrthoNormalizeColsQR`); sign-scan threshold |
| `EPS_MW_UNIQUE` | `1e-8` | `clebsch_aux.cc:1035` (`recDiff2 > 1e-8`) | max-weight uniqueness |
| `FIXRATIONAL_TOL` | `1e-6` | see below | integer-snap tolerance for Cartan eigenvalues and Dynkin conversions |

QSpace's non-MPFR ladder is `CG_EPS1 = 1e-10`, `CG_EPS2 = 1e-12`,
`CG_SKIP_DEPS1 = 1e-12`, `CG_SKIP_DEPS2 = 1e-14` (`clebsch.hh:224-246`).
Here `FIXRATIONAL_TOL` snaps Cartans to their carrier grid (§6), Dynkin
quotients to integers (§7), and CGC entries only to nearby integers (§8).
Tightening a verification-only tolerance is not a gauge change; changing a
value-fixing cut or snap requires the specification-correction rule.

**Reference tolerance-regime deviations.** Three fusion-symbols tolerance choices differ
in *regime* (not just value) from QSpace; none moves a discovered label or
multiplicity, but each is recorded for any comparison with QSpace:

1. **Descent overlap guards are stricter/absolute.** QSpace's `U`-overlap abort
   fires at `x·x > eps` with `x = aMax(Uᵀvi)`, i.e. a **max-overlap `> 1e-4`**;
   its `V`-overlap abort is **norm-relative** (`√(‖Vᵀvi‖²/‖vi‖²) > eps`,
   `clebsch_aux.cc:176-194 @ dd2cc7e`). fusion-symbols uses a single **absolute**
   threshold `EPS_SWEEP = 1e-8` on the max-overlap for both guards — strictly
   tighter. For a faithful sweep both residuals are `~0`, so the choice only
   affects *how loudly a defective generator set fails*, never a valid result.
2. **"Skip if already inside `Vi`" is realized differently.** QSpace early-skips
   a lowering operator whose image already lies in the current level `Vi`
   (`|1 − √(‖Viᵀvi‖²/‖vi‖²)| < eps`, `clebsch_aux.cc:165-169`). fusion-symbols has no
   early test; it projects out `Vi` unconditionally and lets `skip_tiny_cols`
   drop the now-zero columns — **equivalent outcome, different mechanism** (a
   fully-redundant block becomes zero and is dropped either way).
3. **FixRational is integer-only vs denominator-bounded rational.** QSpace's
   `FixRational` is continued-fraction **rational** snapping (mode `'r'`,
   denominator `≤ 1024`, tol `~1e-12`; `clebsch_aux.cc:410-425`), applied to the
   CGC entries too. fusion-symbols snaps CGC entries **to integers only** (§8); Cartans use
   the carrier grid (§6). Consequence: an exact half-integer (or other small-rational)
   CGC entry that QSpace would snap may differ from fusion-symbols' value by `≤ 1e-12` —
   below every gate tolerance.

## 12. Numerical seams (backend)

`CoeffScalar = f64`. Dense operations reach `t4a-faer`
(`src/lie/bcd/linalg.rs`, no hand-rolled factorization kernels):

| Operation | Reference (`@ dd2cc7e`) | t4a-faer API |
|---|---|---|
| column orthonormalization (§4a) | `OrthoNormalizeColsQR` | `qr::no_pivoting::factor::qr_in_place` + sign fix → `Q`, after in-order CGS2 column selection |
| block CGC contractions & `UᵀU` (§5,6) | `Wb::MatProd` | `linalg::matmul::matmul` |
| intertwiner alignment (§15) | orthogonal Procrustes | `linalg::svd::svd` (thin vectors) |

All paths use explicit `Par::Seq` and scratch workspaces where required. Only
`std` is enabled, without `rayon`; global parallelism is neither set nor consulted.

The Gram–Schmidt sweep arithmetic (vector scaling, subtraction, norms, and the
sparse `Sp`/diagonal-`Sz` applications) is the **gauge algorithm itself** — the
analogue of `GroupRepresentations::cgc`'s `cref`, which `docs/gauge.md` §10 likewise carries in
plain code rather than routing through a factorization kernel. The
`t4a-faer` crates.io semver requirement is in `Cargo.toml`.

The contract is value agreement within verification tolerances. The determinism
test pins repeated runs in one process; cross-process bit identity is not
promised because backend reductions may vary.

## 13. Verification (independent oracles)

- **Exact decomposition** (`src/lie/bcd/sweep/tests.rs`): for `defining ⊗ defining` across
  `B2/B3/B4/C2/C3/C4/D3/D4`, the discovered labels and multiplicities equal
  the exact `directproduct` (itself checked against an in-test Brauer–Klimyk
  decomposition and Oscar 1.8.0 rows, `src/lie/bcd/tests.rs`); block dimensions sum to `d1·d2`; each
  CGC is an isometry within tier; and each block's Cartan-eigenvalue rows
  reproduce the irrep's exact weight system (distinct-weight count and total from
  Freudenthal + Weyl orbits).
- **Outer multiplicity ≥ 2**: `D3 (0,1,1) ⊗ (0,1,1)` — the exact layer predicts
  `N = 2` and the sweep reproduces two `(0,1,1)` blocks with OM indices `(0,2)`
  and `(1,2)`.
- **Determinism**: two runs produce bitwise-identical CGC bytes.
- **Sign convention**: each block's first significant CGC entry is positive.
- **Gates**: the multiplicity gate is exercised in both directions (spurious and
  missing expectation); ill-posed generator products are typed errors.

A change that moves any observed value beyond these oracles' tolerances is, by
definition of this document, a breaking release.

## 14. The canonical-parent rule (internal generator catalog)

An irrep's generator frame comes from one canonical parent, not the first
product queried. This is a gauge rule implemented in
`src/lie/bcd/catalog.rs::{canonical_parent,build_into}`.

### 14.1 The well-order `≺` on irreps

Define the total order `≺` on the tensor irreps of a fixed `(series, rank)` by
**box count first**:

```
box(c)  = Σ_i |2λ_i|                             (doubled ε-basis box count)
c₁ ≺ c₂ ⟺ ( box(c₁), dim(c₁), dynkin(c₁) ) <lex ( box(c₂), dim(c₂), dynkin(c₂) )
```

All comparisons are exact. The doubled count handles both tensor and spinor
weights: `box(V) = 2`, `box(ω_r) = r`. Removing a tensor box reduces it by 2.
Box count is primary because dimension need not decrease on removing a box
in D-series chiral weights. At fixed rank and box count the label set is finite,
so `≺` is a well-order.

### 14.2 The canonical parent

The **base cases** are the `≺`-minimal irreps of each class of `P/Q` and carry
generators directly, not from any product:

- the **trivial** rep (dim 1): all generators zero on a 1-dimensional carrier;
- the **defining** rep `(1,0,…,0)`: the exact defining seed (`src/lie/bcd/seeds.rs`);
- the **fundamental spinors** (`B`/`D` only): `ω_r` for `B_r`, and `ω_{r-1}`,
  `ω_r` for `D_r` — the Clifford seeds of §16. They are `≺`-minimal in the
  spinor class, and they are base cases of necessity rather than of
  convenience: a spinor lies in no tensor power of `V`, so no product of
  already-materialized irreps can produce one.

The trivial and defining reps are seeded into the catalog at construction; a
spinor base case is seeded the first time it is asked for, so a catalog that is
only ever asked for tensor irreps builds no spinor matrices at all.

**Base-case frame (normative; corrected in `epoch = 2`).** *Every*
catalog entry, base cases included, is stored in the sweep's **descending-weight
frame** (§7). The trivial rep is 1-dimensional and its frame is unique. The
defining seed's native state order is QSpace's `Setup_*` order, which is **not**
descending-weight, so before it enters the catalog it is passed through one
§1–§8 sweep pass over its own carrier (the internal base-seeding operation) — exactly
what a spinor seed already does (§16.3). One frame convention holds for the
whole catalog, and therefore the §15 coherence guard and the intertwiner
alignment apply to a base case exactly as they do to any other block: **nothing
is exempt from them.**

The epoch-2 correction removed the defining seed's raw `Setup_*` frame and
base-case coherence exemptions, which had mixed incompatible frames in F/R.

**Candidate-set restriction.** For a tensor irrep (integer ε-basis weight),
only tensor irreps are candidate factors. For a spinor, both classes are
candidates. Tensor frames therefore do not depend on the spinor sector.
Every other (`non-base`) irrep
`c` has a **canonical parent pair** `(a, b)`, defined as the minimum, under the
**pair order**

```
key(a,b) = ( dim(a) + dim(b),  dim(a),  dynkin(a),  dynkin(b) )   (lex)
```

over all pairs satisfying

1. `a ≺ c` and `b ≺ c`  (both factors strictly smaller in `≺`), and
2. `N^c_{ab} ≥ 1`, i.e. `c` appears in the exact decomposition `a ⊗ b`
   (the exact B/C/D product decomposition).

The pair order's `dim(a)` tie-break (second component) makes the winner unique
and puts it in canonical `a ⪯ b` form: of the two orderings `(a,b)`/`(b,a)` — which
share the first component `dim(a)+dim(b)` — the one with the smaller-dimensional
factor first wins. `c`'s generators are then the projected generators (§6) of the
`c`-block produced by decomposing `a ⊗ b` (the outer-multiplicity-0 copy when
`N^c_{ab} > 1`; §14.5 states precisely in what sense the copies agree).

The box-first order filters candidates and makes recursion well-founded;
the dimension-based pair key chooses among them. Neither order depends on
query order or numerical residuals.

### 14.3 On-demand recursion (materialization)

`generators(c)` (and the factor bases inside `cgc(a,b,c)`) materialize `c`'s
canonical-parent chain:

```
materialize(c):
    if c already has generators: return
    if c is a base case: it is already seeded; return
    (a, b) = canonical_parent(c)          # §14.2, pure over directproduct
    materialize(a); materialize(b)         # a ≺ c, b ≺ c
    decompose (a ⊗ b)                      # the §1–§13 sweep
    harvest the resulting blocks           # §14.5
```

Recursion descends strictly in `≺` and ends at base cases. This replaces
QSpace's fixed-pass, dimension-capped bootstrap (`clebsch.cc:6649-6773`).

### 14.4 Existence (every non-base irrep has a canonical parent)

The candidate set is nonempty for a non-base irrep: removing one tensor box
provides a smaller parent through the self-dual defining representation and
Frobenius reciprocity. The same reduction applies to non-base spinors, ending
at a fundamental spinor. Use the doubled count of §14.1 throughout.
For background see [Fulton–Harris](references.md#bibliography).

The search is finite and computable from `directproduct` alone: enumerate the finite set
`{ x : x ≺ c }` (contained in the irreps with `box(x) ≤ box(c)`, obtained by a
depth-first walk over integer partitions with **box-count** pruning — box count is
monotone in every coordinate, so the prune is exact for all three series, unlike a
`dim`-based prune which would skip valid D last-coordinate candidates). For each
candidate `a`, read the admissible `b` directly from `directproduct(a*, c)` (the
reciprocity above). Pruning the pair search: iterating `a` in ascending **`dim`**,
stop once `2·dim(a)` exceeds the best sum found — safe because the `key`-minimum
pair `{x,y}` with `dim(x) ≤ dim(y)` is always reached via its smaller factor `x`
(`2·dim(x) ≤ dim(x)+dim(y) ≤` best sum).

### 14.5 Harvest discipline (append-only, canonical-gated)

Decomposing `a ⊗ b` yields blocks for several irreps at once. The catalog appends
a block's generators **iff**:

- the block's irrep `c'` has **no** generators yet (append-only; never overwrite,
  never evict), **and**
- `(a, b)` is exactly `c'`'s canonical parent (§14.2), taking the
  outer-multiplicity-0 copy.

Blocks whose canonical parent is a *different* product are **not** written here;
they are materialized later from their own canonical parent (or are base cases).
This is what makes the stored gauge independent of which product happened to be
decomposed — a second product that rediscovers `c'` finds it already present and
**never** writes.

QSpace may replace stored generators by a numerically cleaner rediscovery
(`clebsch.cc:6710-6718`). Here the canonical parent fixes the stored copy;
rediscovery never replaces it. A loud `debug_assert` compares the Cartan weight
multisets, not complete generator matrices. Returning CGC additionally requires
the canonical-frame alignment and coherence gate of §15, including base cases.
Full generator matrices of unaligned copies are not claimed identical.

### 14.6 Atomicity of the byte budget

Materialize the whole missing chain into a staging buffer. Commit all new
generator sets only if their total charge plus retained charge fits the budget;
otherwise discard the staging buffer and return a typed error. No partial
chain is committed. This bounds retained charge, not transient memory.

### 14.7 What the catalog does NOT own

The catalog retains generator sets and their byte charge. CGC and F values
are cached in separate process-global coefficient caches; catalog reset does
not invalidate those caches. R is recomputed from cached CGC on every query
(`src/lie/bcd/fr.rs::r_symbol`); it has no separate cache. Every materialized
generator set passes the commutator gate at `EPS_SWEEP`.

## 15. Intertwiner alignment of rediscovered frames (this IS gauge)

Before returning CGC, align rediscovered coupled frames to the stored
canonical frame. This alignment is part of the gauge.

### 15.1 The problem it fixes

Numerical decomposition can rotate a degenerate weight space between
embeddings of the same irrep. Weight multisets alone cannot detect that frame
mismatch. Generator coherence must be checked on every returned coupled copy.

### 15.2 The alignment

For a rediscovered block with generators `R_block[i]` and the stored canonical
generators `R_can[i]` of the same irrep, solve the orthogonal intertwiner `W` with

```
R_can[i] · W  =  W · R_block[i]      for every generator i (all Sp and, trivially, Sz)
```

and return `V_can = V · Wᵀ` as the block's CGC (`|can_k⟩ = Σ_j W_kj |block_j⟩`).
`W` is unique up to an overall sign (the real-type commutant is `ℝ·I`; §15.4), and
the sign is pinned by §8 applied to the aligned block.

### 15.3 Why `W` is block-diagonal, and the solve

`W` commutes with the snapped Cartans (`Sz` is integer-equal between the two
frames, §6), so `W` is **block-diagonal over weight spaces**, with one orthogonal
block per weight space of dimension = the weight multiplicity. Non-degenerate
weight spaces (multiplicity 1) carry a `±1`; only degenerate spaces need a genuine
orthogonal block. The solve propagates from the **1-dim highest-weight space**
(block `= +1`, which fixes the global sign) down the ladder: for a target weight
space `T`, each lowering operator `Sp[i]ᵀ` from an already-solved higher space `S`
gives the exact relation

```
A · W_S  =  W_T · B ,   A = R_can[i]ᵀ|_{T,S},   B = R_block[i]ᵀ|_{T,S}
```

Stacking `C = A·W_S` and `B` over all `(i, S)`, `W_T` is the orthogonal
**Procrustes** solution of `W_T·B = C`, i.e. `W_T = U·Vᵀ` from `SVD(C·Bᵀ)`.

Process weight spaces in descending-weight order, with the highest-weight
block `+1`. The polar factor is unique for nonsingular `B·Bᵀ`; a numerical
failure must remain a post-alignment coherence error (§15.5).

### 15.4 Uniqueness / the `±1` (D-odd chirality included)

The commutant of an irrep in the real-matrix ladder basis is `ℝ·I` for all three
series and both chiralities of the D-odd spinor-tensor labels: complex-type
Frobenius–Schur structure lives in the compact-unitary picture and creates no
`SO(2)` freedom in the real ladder basis. So
`W` is determined up to the single global sign, which §8 pins. There is no
continuous gauge freedom left to fix beyond the per-degenerate-space orthogonal
block the ladder already determines.

### 15.5 Verification (the guard moved, it was not removed)

`W` is computed numerically, then **verified**: the aligned generators
`W·R_block·Wᵀ` are compared element-wise against `R_can` at the guard tolerance
(`TOL_BASIS_COHERENT = 1e-10`, QSpace `normDiff` provenance). A frame that still
disagrees after alignment — a genuinely different irrep, or a numerically hopeless
embedding whose only remedy is the out-of-scope extended-precision tier (§11, the
QSpace MPFR analogue) — stays a loud `CatalogError::BasisIncoherent`. The coherence
guard therefore runs **after** alignment, not before it; it is never bypassed, and a
coherent block (raw residual already `≤ tol`) skips alignment on a bit-exact fast
path so no stored value is perturbed.

### 15.6 OM ≥ 2

When `N^c_{ab} ≥ 2`, alignment runs **per copy**: each copy's coupled-side frame is
rotated onto the canonical `R_can` independently, so all copies share one carrier
frame and every four-CGC contraction over the shared `c` leg is coherent.
The OM *index* order is the discovery order (§9). The residual freedom of the O(N) multiplicity-space
mixing is **not** further canonicalized here: the F/R self-consistency gates and the
gauge-invariant isotypic projector `P = Σ_μ C_μ C_μᵀ` are both invariant under it,
and a full canonical multiplicity gauge is not specified.
The discovery order (§9) remains normative; alignment adds no multiplicity-space
rotation rule.

## 16. The spinor base cases (`Spin(N)`)

The B/D fundamental spinors are additional base cases, implemented in
`src/lie/bcd/seeds.rs::spinor_seeds`. Their state and generator conventions
are value-fixing. C-series has no spinor sector.

### 16.1 Carrier and index order

The carrier is the **fermionic Fock space of `r` modes** — the Clifford module
of `so(N)`. A basis state is an occupation string `n ∈ {0,1}^r`, and its ε-basis
weight is

```
λ_k = n_k − ½                     (k = 1…r)
```

so the `2^r` states carry exactly the `2^r` weights `(±½,…,±½)`, each once.

- **`B_r`**: the whole Fock space, dimension `2^r`, is the irreducible spinor
  `ω_r`.
- **`D_r`**: the generators below are all fermion-number-**even**, so the Fock
  space splits by the parity of `Σ_k n_k` into the two half-spinors, dimension
  `2^{r-1}` each. The even sector's highest weight is `ω_r` for `r` even and
  `ω_{r-1}` for `r` odd (the odd sector's is the other one); the seed derives
  its label from the built matrices — the unique state annihilated by every
  raising operator — rather than asserting it.

**State order (gauge).** The states of the carrier (of the sector, for `D_r`)
are listed in the sweep's **descending-weight order** of §7: lexicographic on
the Cartan columns read in reverse, i.e. descending in `(λ_1,…,λ_r)`, i.e.
descending in the bit-reversed occupation string. A spinor's weights are
non-degenerate, so there are no ties and §7's tie-break never fires.

Spinor and defining seeds both enter the catalog in the sweep frame; neither
is exempt from canonical-frame coherence checks (§14.2, §15).

### 16.2 Generators

The following formulas use one-based mode and generator indices.
With `c_k`, `c_k†` the Jordan–Wigner fermion operators of mode `k`
(`c_k|n⟩ = (−1)^{Σ_{l<k} n_l} n_k |n − e_k⟩`, and the adjoint for `c_k†`), on
QSpace's node order (`Sp[i]` carries the simple root `α_{r-i}` for `i < r` and
`α_r` for `i = r`; `Sz[j]` measures `λ_{r+1-j}` — the order §7's Dynkin
conversion inverts):

```
Sz[j] = n_{r+1-j} − ½
Sp[i] = c†_{r-i} c_{r-i+1}        (i < r,   root ε_{r-i} − ε_{r-i+1})
Sp[r] = c†_r / √2                 (B_r,     root ε_r)
Sp[r] = c†_{r-1} c†_r             (D_r,     root ε_{r-1} + ε_r)
```

The Jordan–Wigner sign convention is part of the gauge: it fixes the sign of
every seed entry, hence of every spinor coefficient downstream.

The B short-root scale gives `[Sp_r, Sp_r†] = Sz_1`. It is carried as the
exact squared rational `Seed::raising_scale2 = 1/2`; the Cartan scale is `1/2`.
Jordan–Wigner signs and these scales are part of the gauge.

### 16.3 Verification gate

A spinor seed is gated by the **same** exact `check_commutators` as a defining
seed (`CommutatorViolation` generation diagnostic), and additionally by the fact that the
structure constants it derives are properties of the algebra, not of the
representation: `f_{i,k}` and `d_{i,j}` must equal the defining seed's, exactly.
`src/lie/bcd/spin_tests.rs::spinor_seed_structure_constants_match_the_defining_seed` is that
gate. Before entering the catalog, the seed is passed through one §1–§8 sweep
over its own carrier, which is what puts it in the canonical frame §16.1
describes.

### 16.4 Fingerprint

The existing tag list in `src/lie/bcd.rs::bcd_authority_fingerprint` covers the
spinor sector without an extra spinor tag. The current epoch is 3 (independent
column QR); epoch 2 corrected base-case frames. A value-changing spinor
specification correction requires the same family epoch bump as any other
B/C/D correction. Fingerprint `racah:` prefixes are permanent identifiers.
