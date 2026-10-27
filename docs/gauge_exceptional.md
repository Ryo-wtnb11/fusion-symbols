# Exceptional Lie gauge specification (G2, F4, E6, E7, E8)

This document specifies the **gauge** of the Clebsch–Gordan coefficients (CGC)
and of the F, R, A and B symbols returned for the exceptional simple factors
`G2`, `F4`, `E6`, `E7` and `E8`. A gauge here means the deterministic rules
that fix the basis, sign and order of every coefficient where the mathematics
leaves them free. The coefficients come from one generic coefficient engine
that reads only the Cartan matrix.

## Status: FROZEN NORMATIVE SPECIFICATION

**This document is the authority. The code implements it.** It has the same
standing as [`gauge.md`](gauge.md) and [`gauge_soN.md`](gauge_soN.md), and the
four-step rule for changing a value applies unchanged: spec edit, epoch bump,
CHANGELOG breaking-change entry and regenerated golden values (see
[`gauge.md`, "Status"](gauge.md#status-frozen-normative-specification)).

Every rule below that can change a returned value is stated as a mathematical
definition. The engine computes in exact rational arithmetic, so an
implementation that follows these definitions returns **bit-identical** values,
including across processes and platforms (§8). No floating-point tolerance
fixes any value in this document.

There is no external reference implementation of this gauge to port. TensorKit
and SUNRepresentations.jl have no exceptional-group CGC. QSpace builds its
groups with a floating-point decomposition sweep, which this engine replaces
with an exact construction (see [`gauge_soN.md`](gauge_soN.md) for the QSpace
procedure). The external oracles are mathematical (§9).

**This gauge is unrelated to the SU(N) and B/C/D gauges.** It is defined by
its own rules and shares no basis, sign or multiplicity convention with
[`gauge.md`](gauge.md) or [`gauge_soN.md`](gauge_soN.md). No equality with
those gauges is claimed, even for an algebra both engines could in principle
handle: running these rules on an A, B, C or D Cartan matrix would give a
different, equally valid gauge, which is why A/B/C/D never use this engine.

## 1. Scope

- **Applies to:** the simple root systems `RootSystem::{G2, F4, E6, E7, E8}`
  (`src/lie/group.rs`), in their simply connected forms, built by the
  constructors `GroupId::{g2, f4, e6, e7, e8}()`. These are available only with
  the `lie-generated` feature.
- **Does not apply to:** `A_r`, `B_r`, `C_r` and `D_r` never use this engine.
  SU(2) keeps its exact Racah formulas ([`gauge.md`](gauge.md) §12), SU(N) its
  GT engine (`gauge.md` §1–§11) and B/C/D the decomposition sweep
  ([`gauge_soN.md`](gauge_soN.md)). Their values and fingerprints do not change.
- **Products.** A composite group that contains an exceptional factor uses the
  ordered simple-cover product rules of [`gauge.md` §14](gauge.md#14-semisimple-products-and-central-quotients):
  the last factor is fastest on every axis, and factor coefficients are
  multiplied left to right. An exceptional factor contributes the values
  defined here, unchanged.
- **Quotients.** `G2`, `F4` and `E8` have trivial center, so their cover is also
  their adjoint form. `E6` (center `Z3`) and `E7` (center `Z2`) quotients are
  built with `GroupId::new` from a character lattice. As in `gauge.md` §14, a
  lattice only restricts which irreps a caller may use. It never changes or
  renormalizes a coefficient.

## 2. Inputs and conventions

### 2.1 Cartan matrix

The only group input is the Bourbaki Cartan matrix exactly as
`src/lie/group.rs::cartan` builds it. With 0-based node indices,

```
cartan(i, j) = ⟨α_i, α_j∨⟩      (row i = α_i in fundamental-weight coordinates)
```

so `cartan(i, j)` is Kac's `a_ji`. The Bourbaki node numbering (Bourbaki,
*Lie Groups and Lie Algebras*, Ch. IV–VI, Plates V–IX) is 1-based. Node `k` in
that numbering is coordinate `k−1` here:

| Type | Diagram as coded in `cartan` (0-based) | Non-simply-laced entries |
|---|---|---|
| `G2` | 0 — 1 | `cartan(1,0) = −3`: α₁ (node 0) is short |
| `F4` | 0 — 1 ⇒ 2 — 3 | `cartan(1,2) = −2`: α₁, α₂ long; α₃, α₄ short |
| `E6`, `E7`, `E8` | chain 0 — 2 — 3 — 4 — …, with node 1 attached to node 3 | none |

Weights are written in **Dynkin coordinates** (fundamental-weight
coordinates). Subtracting a simple root is `(μ − α_i)_j = μ_j − cartan(i, j)`.

### 2.2 Symmetrized form

The invariant form is normalized so that long roots have length² 2:
`(α_i, α_j) = d_j · cartan(i, j)`, where `d_j = (α_j, α_j)/2`. This gives
`d = (1/3, 1)` for `G2`, `(1, 1, 1/2, 1/2)` for `F4`, and all ones for `E`. The
code has no form today, so this is a definition, not a citation.

**This normalization cannot move any value.** Every rule below uses only the
integers `cartan(i, j)` and the pairing `⟨μ, α_i∨⟩ = μ_i`. The form appears
only in the Weyl dimension formula (§9) and in `ρ∨`. Since
`⟨α_i, ρ∨⟩ = 1` for every simple root, `ρ∨` enters only through heights.

### 2.3 Chevalley generators

The Lie algebra acts through Chevalley generators `e_i`, `f_i`, `h_i`
(`i = 0…r−1`), with

```
[h_i, h_j] = 0,    [h_i, e_j] = cartan(j, i) e_j,    [h_i, f_j] = −cartan(j, i) f_j,
[e_i, f_j] = δ_ij h_i,
ad(e_i)^{1 − cartan(j, i)} e_j = 0,   ad(f_i)^{1 − cartan(j, i)} f_j = 0   (i ≠ j).
```

`h_i` acts on a vector of weight `μ` by the scalar `μ_i`. For A1 this is the
standard spin convention: `h = 2J_z`, and `f v_λ` has norm² `λ = 2j`, which
matches `J⁻|j, j⟩ = √(2j)|j, j−1⟩`.

### 2.4 Contravariant form

`L(λ)` is the irreducible module with dominant highest weight `λ` and
highest-weight vector `v_λ`. The **contravariant (Shapovalov) form** `⟨·,·⟩`
on `L(λ)` is the unique symmetric bilinear form with

```
⟨v_λ, v_λ⟩ = 1,     ⟨e_i x, y⟩ = ⟨x, f_i y⟩,     ⟨h_i x, y⟩ = ⟨x, h_i y⟩.
```

Different weight spaces are orthogonal. The form is positive definite on
`L(λ)`, because `L(λ)` is unitarizable for the compact real form (Kac,
*Infinite-Dimensional Lie Algebras*, 3rd ed., Thm. 11.7). Every basis below is
real and orthonormal for this form, so `e_i` is the transpose of `f_i`.

## 3. Weight order inside an irrep

For a weight `μ` of `L(λ)`, write `λ − μ = Σ_i k_i α_i` with integers
`k_i ≥ 0`, and call `k(μ) = (k_0, …, k_{r−1})` its **root-coefficient
vector**. Its **depth** is

```
dep(μ) = ⟨λ − μ, ρ∨⟩ = Σ_i k_i .
```

Weights are ordered by

1. `dep(μ)` **ascending** (the highest weight comes first); then
2. `k(μ)` **ascending lexicographically**, with `k_0` (Bourbaki node 1) most
   significant: `μ` precedes `μ′` if, at the first `i` where they differ,
   `k_i(μ) < k_i(μ′)`.

Two distinct weights with equal depth always have different `k`, so this is a
total order. The magnetic index `m ∈ [0, dim L(λ))` lists the weight spaces in
this order. Within a weight space, `m` follows the selection order of §4.
Index `0` is `v_λ`. The lowest weight `w₀λ` has a one-dimensional weight space
and depth `h_λ = ⟨λ − w₀λ, ρ∨⟩ = ⟨λ, 2ρ∨⟩` (because `−w₀` fixes `ρ∨`), so it is
the last index, `dim − 1`.

This order runs in the opposite direction from the SU(N) GT order, where the
highest weight is last (`gauge.md` §1). The two engines do not share bases, so
nothing depends on matching them.

## 4. Basis of each weight space

The basis is built level by level in the order of §3. At depth 0 it is
`u_0 = v_λ`.

### 4.1 Candidates

For a weight `μ` at depth `ℓ ≥ 1`, the **candidates** are the vectors
`f_j w_q` for

- `j = 0, 1, …, r−1` **ascending** (outer loop), keeping only `j` for which
  `μ + α_j` is a weight of `L(λ)` (it then has depth `ℓ − 1` and is already
  built); and
- `q` **ascending** over the basis vectors `w_q` of the weight space
  `μ + α_j`, in their §4.3 order (inner loop).

These candidates span `L(λ)_μ`, because `L(λ) = U(n⁻)v_λ`.

### 4.2 Greedy selection

Walk the candidates in the §4.1 order and **keep** a candidate exactly when it
is not in the span of the candidates already kept. Membership is decided by
exact rank: the candidate's Gram–Schmidt residual (§4.3) has exactly nonzero
norm under the exact rational form. The kept candidates `c_1, …, c_m` form a
basis of `L(λ)_μ`. A weight with no nonzero candidate is not a weight of
`L(λ)`.

**Gate.** `m` must equal the Freudenthal multiplicity `m_λ(μ)` (§9). A
mismatch is a typed generation error.

### 4.3 Exact Gram–Schmidt

The orthonormal basis of `L(λ)_μ` is the Gram–Schmidt orthonormalization of
`c_1, …, c_m` in that order:

```
b_k = c_k − Σ_{l<k} (⟨c_k, b_l⟩ / ⟨b_l, b_l⟩) b_l ,      u_k = b_k / √⟨b_k, b_k⟩ .
```

Each `u_k` has a **positive** coefficient (`1/√⟨b_k, b_k⟩`) on its own
candidate `c_k`. Since greedy selection and Gram–Schmidt use the same order,
the residuals that decide §4.2 are these `b_k`.

### 4.4 Computing the form

The form needs no module realization. If `w` is in `L(λ)_{μ+α_j}` and `w′` is
in `L(λ)_{μ+α_k}`, then

```
⟨f_j w, f_k w′⟩ = ⟨e_k w, e_j w′⟩ + δ_jk (μ + α_j)_j ⟨w, w′⟩ ,
```

where `e_k w` and `e_j w′` lie at depth `ℓ − 2` and are known from earlier
levels. This recursion is a recommended method, not a rule. Any exact
evaluation of the form gives the same values, because §4.1–§4.3 depend only on
the form.

### 4.5 Ladder entries are single signed square roots of rationals

**Claim.** Every matrix entry `⟨u_a, f_j u_b⟩` (and so every entry of `e_j`) is
`s·√q` with `s ∈ {−1, 0, +1}` and `q` a nonnegative rational. It is therefore
exactly an `exact.rs::SignedSqrtRational`.

**Proof.** Work with the unnormalized orthogonal vectors `b_k` of §4.3, with
norms `n_k = ⟨b_k, b_k⟩`. Induction on depth gives two facts: (i) every `n_k`
is rational; (ii) for every `j`, the coefficients `F[a, b]` in
`f_j b_b = Σ_a F[a, b] b_a` are rational.

- At depth 0, `b = v_λ` and `n = 1`.
- In this basis `e_j` has the rational coefficients
  `E[b, a] = F[a, b] · n_a / n_b`, by contravariance.
- So, by §4.4, the Gram matrix of the candidates at the next level is rational.
- Gram–Schmidt on a rational Gram matrix gives rational coefficients
  `b_k = Σ_t r_{kt} c_t` and rational `n_k`.
- Finally `F[a, b] = ⟨f_j b_b, b_a⟩ / n_a` is a rational combination of Gram
  entries.

The normalization `u = b/√n` then gives

```
⟨u_a, f_j u_b⟩ = F[a, b] · √(n_a / n_b) = sign(F) · √(F[a, b]² · n_a / n_b).  ∎
```

The engine stores `(F, n)` exactly and forms the `SignedSqrtRational` entries
from them. It never adds two square roots. The proof does **not** extend to F
and R symbols, which are sums of products of four CGC (§7).

## 5. Tensor products and CGC

### 5.1 Product basis

The product basis index of `|m_a⟩ ⊗ |m_b⟩` in `L(a) ⊗ L(b)` is

```
composite(m_a, m_b) = m_a + d_a · m_b
```

so the first factor is the fast index. This is the B/C/D convention
([`gauge_soN.md` §1](gauge_soN.md#1-input-and-the-kronecker-convention-gauge),
fingerprint tag `kron=a-fast`) and the row order that
`src/lie/bcd/fr.rs::sparse_entries` passes to `frcore`. The coproduct is
`Δ(x) = x ⊗ 1 + 1 ⊗ x`.

### 5.2 Highest-weight space

For a channel `a ⊗ b → c` with highest weight `λ_c`, let `P` be the coupling
pairs `(m_a, m_b)` with `wt(m_a) + wt(m_b) = λ_c`, sorted by
`composite(m_a, m_b)` ascending. The **highest-weight space** is

```
H = { x ∈ span(P) : Δ(e_i) x = 0 for every i },
```

the exact rational nullspace of the stacked `Δ(e_i)` restricted to `span(P)`.
The order of the stacked rows has no effect. **Gate:** `dim H` must equal the
Brauer–Klimyk multiplicity `N^c_{ab}` (§9).

### 5.3 Multiplicity basis

Coordinates are indexed by `P` in its §5.2 order. Define the **pivot rows**
`p_1 < … < p_N` as the rows `p` where
`dim{x ∈ H : x[t] = 0 for all t < p}` drops by one at `p + 1`. The **echelon
basis** `E_1, …, E_N` of `H` is the unique basis with `E_k[p_l] = δ_kl`, so
`E_k[t] = 0` for `t < p_k`. Equivalently, `Eᵀ` is the reduced row echelon form
(pivots equal to `+1`) of any matrix whose rows span `H`.

The multiplicity vectors are the Gram–Schmidt orthonormalization of
`E_1, …, E_N` **in pivot order**, under the inner product of `L(a) ⊗ L(b)`:

```
Ψ_μ = GS(E_1, …, E_N)_{μ+1},     μ = 0, …, N−1 .
```

`Ψ_μ` is a combination of `E_1, …, E_{μ+1}` in which `E_{μ+1}` has a
positive coefficient. Since `E_k[p_l] = δ_kl`, this gives exactly:

- `Ψ_μ[p_{μ+1}] > 0`;
- `Ψ_μ[p_l] = 0` for every `l > μ + 1`.

At the earlier pivots `p_l` (`l ≤ μ`), `Ψ_μ` is generally **nonzero**. For
example, in `G2 27⊗27 → 27` the pivots are `[0, 1]` and `Ψ_1[0] ≠ 0`. The
outer-multiplicity index `μ` is the pivot order.

**Coordinate independence.** An implementation can work in the rational
coordinates `b^a_{m_a} ⊗ b^b_{m_b}` (§4.5), whose inner-product metric is
`diag(n^a_{m_a} n^b_{m_b})`. Changing coordinates scales each row by a positive
factor. Under such a scaling the pivot rows stay the same, each `E_k` changes
only by a positive factor, and Gram–Schmidt gives the same output. The
rational computation therefore produces the vectors defined above.

### 5.4 Lowering: the coupled side follows c's own basis

For each `μ`, let `Φ_μ : L(c) → L(a) ⊗ L(b)` be the unique intertwiner with
`Φ_μ(v_{λ_c}) = Ψ_μ`. The CGC is

```
C[m_a, m_b, m_c, μ] = ⟨u^a_{m_a} ⊗ u^b_{m_b}, Φ_μ(u^c_{m_c})⟩ .
```

`Φ_μ` is computed by repeating c's own §4 recipe in the product. If
`b^c_k = Σ_t r_{kt} f_{j_t} b^c_{q_t}`, with the rational coefficients of
§4.3, then

```
Φ_μ(b^c_k) = Σ_t r_{kt} Δ(f_{j_t}) Φ_μ(b^c_{q_t}),      Φ_μ(u^c_k) = Φ_μ(b^c_k) / √n^c_k .
```

No Gram–Schmidt runs on the coupled side, so there is no coupled-frame
alignment problem like the one in [`gauge_soN.md` §15](gauge_soN.md#15-intertwiner-alignment-of-rediscovered-frames-this-is-gauge).
Every CGC into `c`, from every product, is expressed in the same basis: `c`'s
standalone basis.

`Φ_μ` is well defined. `Ψ_μ` is a singular vector in a completely reducible
module, so it generates a copy of `L(λ_c)`. Every linear relation among
`f`-words applied to `v_{λ_c}` therefore also holds for `Ψ_μ`. `Φ_μ` is an
isometry, and copies are orthogonal: `Φ_μᵀ Φ_ν` commutes with `L(c)`, so by
Schur's lemma it is the scalar `⟨Ψ_μ, Ψ_ν⟩ = δ_μν`.

**Exactness.** `Ψ_μ = B_μ / √N_μ` with `B_μ` rational in the rational
coordinates and `N_μ = ⟨B_μ, B_μ⟩` rational. The recipe above keeps the
coordinates rational. Each CGC entry is then
`x · √(n^a_{m_a} n^b_{m_b} / (N_μ n^c_{m_c}))` with `x` rational: a single
signed √rational, by the same argument as §4.5.

### 5.5 Normalization and sign: the complete rule set

The following rules fix every CGC sign and normalization. Everything else
follows from them by linearity.

1. **Irrep basis** (§4.3): `u_k` has a positive coefficient on its own
   candidate, and `u_0 = v_λ` with `⟨v_λ, v_λ⟩ = 1`.
2. **Highest-weight multiplicity vectors** (§5.3): the Gram–Schmidt
   orthonormalization of the echelon basis `E_1, …, E_N`, in pivot order.
   Unit norm and `Ψ_μ[p_{μ+1}] > 0` are consequences of this rule, not a
   substitute for it. For `N ≥ 2`, other orthonormal bases also satisfy them,
   for example Löwdin orthonormalization or Gram–Schmidt in reverse order.
3. **Everything else** is induced by the intertwiner (§5.4). The coupled-side
   normalization is the isometry property, with no separate sign rule.

Consequences that need no special case:

- **Trivial couplings.** For `1 ⊗ s → s` and `s ⊗ 1 → s`, `P` has the single
  pair `(0, 0)`, `Ψ = +1`, and the intertwiner reproduces `s`'s basis. So
  `C[0, m, m, 0] = C[m, 0, m, 0] = 1` exactly.
- **No exchange symmetry is imposed.** `a ⊗ b → c` and `b ⊗ a → c` are each
  built by these rules. Their relation is the R symbol (§7).

### 5.6 Entry order handed to `frcore`

Entries go to `frcore` (§7) as `MEntry` in this order: `μ` ascending (outer),
then `m_c` ascending, then `composite(m_a, m_b)` ascending. This is the order of
`src/lie/bcd/fr.rs::sparse_entries`. Only exactly nonzero entries are emitted,
and since values are exact there is no purge threshold. Each `value` is
`SignedSqrtRational::to_f64`, the correctly rounded conversion (`src/exact.rs`)
and the only rounding of a CGC. The entry order is part of the gauge because
`frcore` sums in `f64` in input order (§8).

## 6. Duality, Frobenius–Schur and conjugation

### 6.1 Dual label

The dual of `L(λ)` is `L(−w₀λ)`, as for B/C/D (`src/lie/bcd.rs::Irrep::dual`).
For `G2`, `F4`, `E7` and `E8` `−w₀ = 1`, so every irrep is self-dual. For `E6`,
`−w₀` is the diagram automorphism (Bourbaki Plate V):
`a₁ ↔ a₆` and `a₃ ↔ a₅` in Bourbaki numbering, with `a₂` and `a₄` fixed. In
0-based coordinates this swaps `0 ↔ 5` and `2 ↔ 4`.

**Dual bases are not transported.** `L(−w₀λ)` is built from its own label by
§4, not by carrying `L(λ)`'s basis through the diagram automorphism. That
automorphism reorders `k` (§3) and the `j` loop (§4.1), so the two bases do not
correspond index by index. The CGC `a ⊗ ā → 1` and `ā ⊗ a → 1` are each built
by §5. Their relation is the phase `κ_a` of §6.3.

### 6.2 Frobenius–Schur indicator of self-dual irreps

For a self-dual irrep of a compact simple group, the indicator is
`(−1)^{⟨λ, 2ρ∨⟩}`: the value of the central element `exp(2πi ρ∨)` on `L(λ)`
(Bourbaki Ch. VIII §7; the same formula is used by
`src/lie/bcd.rs::Irrep::frobenius_schur`). Write `2ρ∨ = Σ_i c_i α_i∨` in simple
coroots. Then `⟨λ, 2ρ∨⟩ = Σ_i c_i a_i` over the Dynkin labels `a_i`, with:

| Type | `(c_0, …, c_{r−1})` | Consequence |
|---|---|---|
| `G2` | `(6, 10)` | all even: every irrep real |
| `F4` | `(22, 42, 30, 16)` | all even: every irrep real |
| `E6` | `(16, 22, 30, 42, 30, 16)` | all even: every self-dual irrep real |
| `E7` | `(34, 49, 66, 96, 75, 52, 27)` | pseudo-real exactly when `a₂ + a₅ + a₇` (Bourbaki numbering) is odd |
| `E8` | `(92, 136, 182, 270, 220, 168, 114, 58)` | all even: every irrep real |

The coefficients are the Bourbaki Plates V–IX values of `2ρ` for the dual root
system. They were recomputed independently by summing the positive coroots
generated from `cartan`. For simply laced types they equal the `2ρ`
coefficients. For `G2` and `F4` they are those coefficients in reverse order.

**E7 check.** `56` (`a₇ = 1`) and `912` (`a₂ = 1`) are pseudo-real, while `133`
(`a₁ = 1`) and `1539` (`a₆ = 1`) are real. This agrees with the standard
reality table (Slansky 1981, Phys. Rep. 79, Table 25; the table number is
unverified). The parity `a₂ + a₅ + a₇ mod 2` is also E7's `Z2` central
character, consistent with `exp(2πi ρ∨)` generating the center.

### 6.3 The Frobenius–Schur phase `κ_a` in this gauge

`FusionCategory::frobenius_schur_phase` is defined as the phase of
`F^{a ā a}_a[𝟙, 𝟙]` (`src/category.rs`; TensorKitSectors 0.3.9
`frobenius_schur_phase_from_Fsymbol`). `A^{ab}_c` uses it
(`src/lie.rs::a_symbol`). This section derives that phase in this gauge.

**Reduction to the singlet.** With `e = f = 𝟙`, the trivial-coupling identities
of §5.5 make `frcore::f_block_raw` reduce to

```
F^{a ā a}_a[𝟙,𝟙] = Σ_m C^{a ā → 1}[0, m] · C^{ā a → 1}[m, 0] .
```

Write `ψ_{xy}` for the singlet in `L(x) ⊗ L(y)`. Swapping the factors of
`ψ_{aā}` gives an invariant unit vector of `L(ā) ⊗ L(a)`, so it equals
`s_a ψ_{āa}` with `s_a = ±1`. Then `F = s_a / d_a`, because `√d_a C^{aā→1}` is
an orthogonal matrix, and **`κ_a = s_a`**.

**Pivot.** In the singlet channel, the first pair of `P` is
`(m_a, m_ā) = (d_a − 1, 0)`: the lowest weight of `a` paired with the highest
weight of `ā`, with composite index `d_a − 1`. Every other pair has
`m_ā ≥ 1`, so composite index `≥ d_a`. The singlet component there is nonzero,
because the pairing of `L(a)_μ` with `L(ā)_{−μ}` is nondegenerate. So by
§5.3, `ψ_{aā}[d_a − 1, 0] > 0`.

**Closed form.** Let `L = Σ_i f_i` and `h = ⟨λ, 2ρ∨⟩`, the depth of the lowest
weight of both `a` and `ā`. Invariance gives `(L ⊗ 1)ψ = −(1 ⊗ L)ψ`, so
`(L^h ⊗ 1)ψ = (−1)^h (1 ⊗ L^h)ψ`. Take the `(low_a, low_ā)` component:

```
ψ_{aā}[0, d_ā − 1] · ⟨low_a| L^h |0⟩_a = (−1)^h · ψ_{aā}[d_a − 1, 0] · ⟨low_ā| L^h |0⟩_ā .
```

Write `σ_x = sign ⟨low_x| L^h |high_x⟩`, computed in `x`'s own §4 basis. This
matrix element is never zero. `Σ_i c_i f_i`, with the positive `c_i` of the
principal `sl₂` (Kostant 1959), is conjugate to `L` by a real positive torus
element, and `v_λ` is a highest-weight vector of weight `h` for that `sl₂`. So
`L^h v_λ ≠ 0`, and the torus rescaling keeps the sign. Combining the pivot
rule with the `ā ⊗ a` version of the same identity gives

```
κ_a = s_a = (−1)^{⟨λ, 2ρ∨⟩} · σ_a · σ_ā .
```

**Normative rule.**

- **Self-dual irreps** (`a = ā`). `σ_a² = 1`, so
  `κ_a = (−1)^{⟨λ, 2ρ∨⟩}`. This is the gauge-invariant indicator of §6.2, the
  same closed form as `bcd.rs::frobenius_schur_phase`. It needs no CGC.
- **E6 non-self-dual irreps.** `κ_a = σ_a σ_ā` (`⟨λ, 2ρ∨⟩` is even for `E6`).
  `κ_ā = κ_a` follows from `s_a s_ā = 1`. The implementation must compute
  `σ_a` and `σ_ā` exactly, using the rational recursion (`L^h v_λ` in the
  unnormalized basis of §4.5; positive rescalings keep the sign), unless the
  conjecture below is proved and the spec is amended first.
- **Frobenius–Schur indicator.** `κ_a` when `a = ā`, otherwise `0`, as for
  every family.

**Comparison with B/C/D (`src/lie/bcd.rs::Irrep::frobenius_schur_phase`).**
B/C/D returns `(−1)^{⟨λ, 2ρ∨⟩}` for every irrep, including the complex `D_r`
(`r` odd) irreps. `tests/lie_bcd.rs` checks that value against the F-symbol
sign in the B/C/D gauge. The exceptional engine uses the same formula for every
self-dual irrep. For E6 complex irreps the same formula would give `+1`. This
spec does not assert `+1`; it fixes the gauge-derived `σ_a σ_ā`. The reason is
that `σ_a` depends on the §4 construction, and its positivity is proved only
for minuscule irreps.

- **Minuscule irreps (proved).** This covers `E6 27, 27̄` and `E7 56`; the
  other exceptional types have no nontrivial minuscule irrep. Every weight is
  Weyl-conjugate to `λ`, so it is multiplicity-free and
  `⟨μ, α_i∨⟩ ∈ {−1, 0, 1}`.

  **Claim:** in the §4 basis every `f_i` entry is `0` or `+1`. Then `L^h v_λ`
  is a positive multiple of `u_low`, so `σ = +1`.

  **Proof**, by induction on the depth of the target weight `μ`:
  1. If `μ + α_i` is a weight, then `⟨μ + α_i, α_i∨⟩ = 1`, so
     `e_i u_{μ+α_i} = 0` and `‖f_i u_{μ+α_i}‖² = 1`.
  2. By §4.3, `u_μ = f_k u_{μ+α_k}` for the first such `k`, so that entry is
     `+1`.
  3. Take another `i ≠ k` with `μ + α_i` a weight. Then
     `⟨μ + α_i, α_k∨⟩ = −1 + cartan(i, k)`. This must lie in `{−1, 0, 1}`, so
     `cartan(i, k) = 0` and nodes `i` and `k` commute.
  4. Since `⟨μ + α_i, α_k∨⟩ = −1`, the weight `τ = μ + α_i + α_k` exists. By
     the induction hypothesis `u_{μ+α_i} = f_k u_τ` and
     `u_{μ+α_k} = f_i u_τ`.
  5. So `f_i u_{μ+α_i} = f_i f_k u_τ = f_k f_i u_τ = f_k u_{μ+α_k} = u_μ`. ∎
- **Conjecture: `σ_a = +1` for every irrep (unverified remark, not part of
  the normative rule).** If true, `κ_a = (−1)^{⟨λ, 2ρ∨⟩}` for every
  exceptional irrep, exactly as in B/C/D. The normative `σ_a σ_ā` holds by
  definition and does not depend on this conjecture. If the conjecture is
  proved, replacing the computation with the closed form needs a spec
  amendment. The returned values would be identical, so the fingerprint would
  not change.
  The prototype (§9) found `σ = +1` for all of these: `G2` `(a, b)` with
  `a ≤ 4`, `b ≤ 3`; `F4` `26, 52, 273, 324, 1053, 1274, 4096`; `E6`
  `27, 351, 351′, 1728, 3003, 7371` and their duals, and `78`; `E7`
  `56, 133, 1463, 1539`; and `E8` `248`.

## 7. F, R, A and B

These symbols use the generic core `src/frcore.rs` exactly as SU(N) and B/C/D
do:

- **F**: `f_block_raw` contracts four CGC with axis order `[μ, ν, κ, λ]`. It
  slices the `d` leg at `m_d = 0`, which here is the highest weight of `d`.
  F does not depend on the slice.
- **R**: `r_block_raw` contracts `CGC(a, b, c)` with `CGC(b, a, c)` at
  `m_c = 0`. The CGC are real, so conjugation is the identity, as `frcore`
  documents.
- **A, B**: the `src/lie.rs` overrides `a_symbol` and `b_symbol` evaluate the
  TensorKitSectors F forms, with `κ_a` from §6.3.
- **Twist and braiding**: as for every group: bosonic, with twist `1`
  (`bcd.rs::Irrep::twist`).
- **Empty channels**: zero blocks shaped by the vertex multiplicities, as for
  every family.

There are **no exceptional-specific conventions** in F, R, A and B. The only
exceptional-specific inputs are the CGC (§5) and `κ_a` (§6.3).

## 8. Fingerprint

`authority_fingerprint()` for an exceptional factor names this document's
convention set and epoch. It covers:

- the Cartan input (§2.1);
- the weight order and tie direction (§3);
- candidate order, greedy rank selection and positive-own-candidate
  Gram–Schmidt (§4);
- the `a`-fast product basis (§5.1);
- the RCEF-pivot Gram–Schmidt multiplicity basis (§5.3);
- the intertwiner lowering (§5.4);
- the `frcore` entry order and single correct rounding (§5.6);
- the `κ_a` rule (§6.3);
- the `frcore` F/R wiring (§7).

It does not cover the dense backend: this engine uses none for values.

**Contract.** All values are exact until one correctly rounded conversion per
CGC entry. `frcore` then runs a deterministic `f64` reduction: sorted keys,
input order, and no implicit FMA contraction in Rust. So, as for base SU(2),
**equal fingerprints mean bit-identical CGC, F, R, A and B values**. This is
stronger than the B/C/D contract (`gauge_soN.md` §12).

**Epoch.** A new, independent family epoch, starting at 1. The su2, sun and
bcd epochs are unchanged. The bytes are:

```
fusion-symbols:lie-exceptional:ref=gauge-exceptional-spec:input=bourbaki-cartan:order=depth-rootlex-asc:basis=greedy-rank-gs-own-positive:kron=a-fast:mult=rcef-pivot-gs:descent=intertwiner-own-basis:sign=own-candidate-and-pivot-positive:round=once-correct:fs=gauge-derived:fr=frcore:twist=one:scalar=f64:epoch=1
```

Each `key=value` segment names a rule of this document, in the style of the
su2 and bcd tags:

- `ref` is this specification;
- `input` is §2.1;
- `order` is §3;
- `basis` is §4;
- `kron` is §5.1;
- `mult` is §5.3;
- `descent` is §5.4;
- `sign` is §5.5;
- `round` is §5.6;
- `fs` is §6.3;
- `fr` is §7;
- `twist=one` and `scalar=f64` are §7 (bosonic twist `1`) and the `f64` value
  type, matching the other `fusion-symbols:` fingerprints.

The `fusion-symbols:` prefix follows the newer fingerprints (`src/finite.rs`
ZN, `src/fermion.rs` and `src/lie/cu1.rs`). The `racah:` prefix stays only on
the older su2, sun and bcd epochs. The implementation pins these bytes in
`tests/exceptional_fingerprint.rs`.

## 9. Validation oracles

An implementation must pass each oracle below. The oracle must not use the
engine's own descriptors to compute expected values.

1. **Weyl dimension.** `dim L(λ)` must equal the Weyl formula. Fundamental
   dimensions in Bourbaki order, as recomputed:
   - `G2`: `7, 14`;
   - `F4`: `52, 1274, 273, 26`;
   - `E6`: `27, 78, 351, 2925, 351, 27`;
   - `E7`: `133, 912, 8645, 365750, 27664, 1539, 56`;
   - `E8`: `3875, 147250, 6696000, 6899079264, 146325270, 2450240, 30380, 248`.
2. **Freudenthal multiplicities.** Each weight space's selected rank (§4.2)
   must equal Freudenthal's `m_λ(μ)`. This check is the production gate.
3. **Brauer–Klimyk N.** `dim H` (§5.2) must equal the Brauer–Klimyk
   multiplicity. Labels are in Bourbaki order:
   - `G2`: `7⊗7 = 1 + 7 + 14 + 27`;
   - `G2`: `7⊗14 = 7 + 27 + 64`;
   - `G2`: `14⊗14 = 1 + 14 + 27 + 77(0,2) + 77(3,0)`;
   - `G2`: `7⊗27 = 7 + 14 + 27 + 64 + 77(3,0)`;
   - `F4`: `26⊗26 = 1 + 26 + 52 + 273 + 324`;
   - `E6`: `27⊗27 = 27̄ + 351(0,0,1,0,0,0) + 351′(2,0,0,0,0,0)`;
   - `E6`: `27⊗27̄ = 1 + 78 + 650`;
   - `E7`: `56⊗56 = 1 + 133 + 1463 + 1539`;
   - `E8`: `248⊗248 = 1 + 248 + 3875 + 27000 + 30380`.

   The `14⊗14` labels name the two 77s: `77(0,2)` and `77(3,0)`.
   **First G2 channel with outer multiplicity ≥ 2:** `27⊗27`, ordered by
   `d_a·d_b`. All G2 irreps of dimension below 104 were searched.
   `27⊗27 = 1 + 7 + 14 + 2·27 + 2·64 + 77(0,2) + 77(3,0) + 182 + 189`. This
   was verified with an independent Brauer–Klimyk computation. A literature
   cross-check is unverified.
4. **Generator relations.** Each built irrep must satisfy
   `[e_i, f_j] = δ_ij h_i`, `[h_i, e_j]`, and both Serre relations of §2.3,
   **exactly**, in the rational basis of §4.5.
5. **Orthonormality and intertwining.**
   - Each CGC, reshaped as `M[(m_a, m_b), (m_c, μ)]`, is an exact isometry.
   - `Δ(x) Φ_μ = Φ_μ x` exactly, for every `e_i` and `f_i`, including the
     candidates that §4.2 did not keep.
   - `C[m_a, m_b, m_c, μ] = 0` unless `wt(m_a) + wt(m_b) = wt(m_c)`.
6. **Category gates.** Pentagon, hexagon and F-unitarity residuals
   (`src/category/residual.rs`) must be within the generated-family tolerance,
   over channels that include outer multiplicity `≥ 2` (`G2 27⊗27`).
   `tests/category_conformance.rs` also checks `κ_a` against `sign(F)`.
7. **Frobenius–Schur.**
   - `κ_a` must equal `sign(d_a F^{aāa}_a[𝟙,𝟙])` for `G2 7, 14`, `E6 27, 27̄`
     and both non-minuscule E6 complex `351` pairs, `(0,0,1,0,0,0)` and
     `(2,0,0,0,0,0)` with their duals (`κ = +1` for each).
   - The indicator of `E7 56` and `912` is `−1`.
   - The E6 complex indicator is `0`.
8. **Byte-identical regeneration.** Two cold generations of the same channel,
   in separate processes, must give identical CGC and F/R bytes. Golden values
   pin representative signed entries, at least one per type and one OM ≥ 2
   entry.

The checks behind the claims in this document come from a test-only exact
Python prototype of §3–§6, written while drafting this specification. It is not
part of the crate. It found:

- exact relations for `G2 7, 14, 27, 64`, `F4 26, 52`, `E6 27, 27̄, 78`,
  `E7 56` and `E8 248`;
- exact isometries for every `G2 7⊗7`, `7⊗14` channel and the OM = 2
  `27⊗27 → 27, 64`;
- `κ` equal to the formula of §6.3 for `G2 7, 14`, `E6 27, 27̄` (`+1`) and
  `E7 56` (`−1`).

## 10. Resource notes

Prototype measurements in exact rational arithmetic:

| Object | dim | levels (`h + 1`) | max weight multiplicity | CGC nonzeros |
|---|---|---|---|---|
| `G2 7` | 7 | 7 | 1 | — |
| `G2 14` | 14 | 11 | 2 | — |
| `G2 27` | 27 | 13 | 3 | — |
| `G2 64` | 64 | 17 | 4 | — |
| `G2 27⊗27 → 27` (N = 2) | — | — | — | 1354 |
| `G2 27⊗27 → 64` (N = 2) | — | — | — | 2904 |
| `E8 248` | 248 | 59 | 8 (zero weight) | — |

The largest G2 `27⊗27` radicands have at most 5-digit numerators and 7-digit
denominators. Basis norms of `E6 1728` stay below 4 digits.

**E8 adjoint.** Building `248` is cheap: an 8×8 Gram matrix at weight zero.
`248⊗248` has `61504` product states. For channels with `N = 1`, the total
nonzero count over all five channels is at most
`Σ_μ P(μ)² = 1 787 056`, where `P(μ)` is the number of product pairs of weight
`μ`. A dense form would hold `61504²` entries. The dominant cost is building
`27000` and `30380` (§4), with exact Gram matrices whose size is bounded by
`8 · m_λ(μ + α_j)` candidates per weight. These sizes have not been measured;
the implementation must report them.

**Out of scope for this specification:**

- coefficient cache tiers;
- byte budgets for exceptional generation;
- floating-point or modular fast paths;
- non-simply-connected constructors beyond `GroupId::new`;
- any change to A/B/C/D;
- a proof of the `σ = +1` conjecture (§6.3).
