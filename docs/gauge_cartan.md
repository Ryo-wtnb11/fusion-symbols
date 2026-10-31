# Simple Lie factors: gauge specification (Cartan engine)

This document specifies the **gauge** of the Clebsch–Gordan coefficients (CGC)
and of the F, R, A and B symbols of every simple factor of a `LieGroup`: the
root systems `A_r`, `B_r`, `C_r`, `D_r`, `G2`, `F4`, `E6`, `E7` and `E8`. A
gauge is the set of deterministic rules that fix the basis, sign and order of
every coefficient where the mathematics leaves them free.

All simple types use one exact pipeline, the **Cartan engine**: a
highest-weight nullspace followed by an intertwiner descent, computed in exact
rational arithmetic from the Cartan matrix and rounded once. The only
type-dependent input is the **irrep basis** (§3): the Gelfand–Tsetlin basis
for `A_r`, and a Gram–Schmidt basis of the contravariant form for every other
type. SU(2) additionally fixes one vertex sign (§4.6) and keeps closed-form F,
R and Frobenius–Schur phases ([`gauge.md` §1](gauge.md#1-su2-closed-forms)).

## Status: FROZEN NORMATIVE SPECIFICATION

**This document is the authority. The code implements it.** The rule for
changing a value (spec edit, CHANGELOG breaking-change entry, regenerated
golden values, semver-breaking release) is stated once in
[`gauge.md`, "Status"](gauge.md#status-frozen-normative-specification) and
applies here unchanged.

Every rule below that can change a returned value is a mathematical
definition. The pipeline computes in exact rational arithmetic and rounds each
CGC entry once, so an implementation that follows these definitions returns
**bit-identical** values across processes and platforms (§6.2). No
floating-point tolerance fixes any value in this document.

The external references are cited as provenance and as oracles only
(§8): SUNRepresentations.jl v0.4.0 and TensorKitSectors for `A_r`, Molev
(arXiv:math/0211289) for the Gelfand–Tsetlin formulas, Bourbaki for the root
data. No external implementation of the `B_r`–`E8` gauge exists to port.

Generation-time verification is normative and separate from test oracles:
§7 lists the checks every generation performs, §8 the oracles an
implementation must pass.

## 1. Scope

The gauge of a simple factor is chosen by its **presented root system**
(`GroupId::root_systems`), not by an isomorphism class:

| Presented root system | Groups | Irrep basis | Magnetic index order | F, R, `κ` |
|---|---|---|---|---|
| `A_1` (also `B_1`, `C_1`, and each factor of `D_2`) | SU(2), SO(3), Sp(2), PSU(2); SO(4) factors | §3.1 (Condon–Shortley `\|j m⟩`) | `m` ascending: index `m + j` | closed forms, [`gauge.md` §1](gauge.md#1-su2-closed-forms) |
| `A_r`, `r ≥ 2` | SU(r+1) and its quotients | §3.1 (Gelfand–Tsetlin) | GT pattern order (§3.1.4) | §5, §6 |
| `B_r` (`r ≥ 2`), `C_r` (`r ≥ 2`), `D_r` (`r ≥ 3`), `G2`, `F4`, `E6`, `E7`, `E8` | Spin/SO(2r+1), Sp(2r), Spin/SO(2r), half-spin forms, exceptional groups and their quotients | §3.2 (Gram–Schmidt) | depth order (§3.2.1) | §5, §6 |

- **Isomorphic presentations are different gauges.** `B_2` and `C_2`, and
  `D_3` and `A_3`, are each built from their own Cartan matrix and their own
  basis rule. No value is transported through an isomorphism.
- **Global forms and quotients** (SO vs Spin, PSp, PSU(N), SU(N)/Z_k, half-spin,
  E6/Z3, E7/Z2, character lattices passed to `GroupId::new`) only restrict which
  labels a caller may use (`GroupId::admits`). They never change or renormalize
  a coefficient.
- **Products** of simple factors and tori use the ordered product rules of
  [`gauge.md` §2](gauge.md#2-semisimple-products-and-central-quotients). A
  simple factor contributes the values defined here, unchanged. Torus factors
  (U(1)) have no Cartan matrix and are outside this document.
- **Build features.** `A_1` coefficients are available in every build;
  higher-rank simple types require the `lie-generated` feature. Feature
  selection admits or rejects a group; it never changes a value.
- **Not covered here:** the closed-form SU(2) F, R and Frobenius–Schur
  conventions and the free functions `lie::su2::{clebsch_gordan, wigner_3j,
  wigner_6j}` ([`gauge.md` §1](gauge.md#1-su2-closed-forms)).

## 2. Inputs and conventions

### 2.1 Cartan matrix

The only group input is the Bourbaki Cartan matrix exactly as
`src/lie/group.rs::cartan` builds it. With 0-based node indices,

```
cartan(i, j) = ⟨α_i, α_j∨⟩      (row i = α_i in fundamental-weight coordinates)
```

so `cartan(i, j)` is Kac's `a_ji`. Bourbaki node `k` (1-based, Bourbaki,
*Lie Groups and Lie Algebras*, Ch. VI, Plates I–IX) is coordinate `k−1`:

| Type | Diagram as coded in `cartan` (0-based) | Non-simply-laced entries |
|---|---|---|
| `A_r` | chain `0 — 1 — … — r−1` | none |
| `B_r` | chain `0 — … — r−2 ⇒ r−1` | `cartan(r−2, r−1) = −2`: `α_r` (node `r−1`) is short |
| `C_r` | chain `0 — … — r−2 ⇐ r−1` | `cartan(r−1, r−2) = −2`: `α_r` (node `r−1`) is long |
| `D_r` | chain `0 — … — r−2`, with node `r−1` also attached to node `r−3` | none |
| `G2` | `0 — 1` | `cartan(1, 0) = −3`: `α₁` (node 0) is short |
| `F4` | `0 — 1 ⇒ 2 — 3` | `cartan(1, 2) = −2`: `α₁, α₂` long; `α₃, α₄` short |
| `E6`, `E7`, `E8` | chain `0 — 2 — 3 — 4 — …`, with node 1 attached to node 3 | none |

All other off-diagonal entries are `−1` on an edge and `0` otherwise.
Weights are written in **Dynkin coordinates** (fundamental-weight
coordinates). Subtracting a simple root is `(μ − α_i)_j = μ_j − cartan(i, j)`.

### 2.2 Symmetrized form

The invariant form is normalized so that long roots have length² 2:
`(α_i, α_j) = d_j · cartan(i, j)` with `d_j = (α_j, α_j)/2`:

| Type | `d` |
|---|---|
| `A_r`, `D_r`, `E6`–`E8` | all `1` |
| `B_r` | `(1, …, 1, 1/2)` |
| `C_r` | `(1/2, …, 1/2, 1)` |
| `G2` | `(1/3, 1)` |
| `F4` | `(1, 1, 1/2, 1/2)` |

**This normalization cannot move any value.** Every rule below uses only the
integers `cartan(i, j)` and the pairing `⟨μ, α_i∨⟩ = μ_i`. The form appears
only in the Weyl dimension formula (§8) and through `ρ∨`, which enters only
through heights because `⟨α_i, ρ∨⟩ = 1` for every simple root.

### 2.3 Chevalley generators

The Lie algebra acts through Chevalley generators `e_i`, `f_i`, `h_i`
(`i = 0…r−1`), with

```
[h_i, h_j] = 0,    [h_i, e_j] = cartan(j, i) e_j,    [h_i, f_j] = −cartan(j, i) f_j,
[e_i, f_j] = δ_ij h_i,
ad(e_i)^{1 − cartan(j, i)} e_j = 0,   ad(f_i)^{1 − cartan(j, i)} f_j = 0   (i ≠ j).
```

`h_i` acts on a vector of weight `μ` by the scalar `μ_i`. For `A_1` this is
the spin convention `h = 2J_z`, `f = J⁻`, `e = J⁺`.

### 2.4 Contravariant form

`L(λ)` is the irreducible module with dominant highest weight `λ` and
highest-weight vector `v_λ`. The **contravariant (Shapovalov) form** `⟨·,·⟩`
on `L(λ)` is the unique symmetric bilinear form with

```
⟨v_λ, v_λ⟩ = 1,     ⟨e_i x, y⟩ = ⟨x, f_i y⟩,     ⟨h_i x, y⟩ = ⟨x, h_i y⟩.
```

Different weight spaces are orthogonal. The form is positive definite on
`L(λ)`, because `L(λ)` is unitarizable for the compact real form (Kac,
*Infinite-Dimensional Lie Algebras*, 3rd ed., Thm. 11.7). Every basis in §3 is
real and orthonormal for this form, so `e_i` is the transpose of `f_i`.

### 2.5 Labels

An irrep is named by the Dynkin labels `a = (a_1, …, a_r)` of its highest
weight in Bourbaki numbering, `λ = Σ a_i ω_i` (`LieGroup::irrep`,
[Dynkin labels](user-guide/labels.md)). The pipeline builds the simply
connected cover, so every tuple of nonnegative integers is a valid input; this
includes the spinor classes:

- `B_r`: spinor irreps have `a_r` odd (`ω_r` is the `2^r`-dimensional spinor);
- `D_r`: spinor irreps have `a_{r−1} + a_r` odd (`ω_{r−1}`, `ω_r` are the
  half-spinors).

No separate spinor construction exists: spinor irreps are built by §3.2 like
every other irrep.

`A_r` labels map to the Young-diagram partition used by §3.1:
`λ_i = Σ_{j=i}^{r} a_j` for `i = 1, …, r+1` (so `λ_{r+1} = 0`). For `A_1`,
`a_1 = 2j` (the doubled spin); `irrep_spin` and `HalfInt` are label encodings
only.

## 3. Irrep bases

The pipeline needs, for each irrep `L(λ)`, a real orthonormal basis
`u_0, …, u_{d−1}` of weight vectors, an order on it (the public **magnetic
index** `m`), and the convention that the highest-weight basis vector is
`+v_λ`. Every coefficient in §4–§6 is a function of these data only.

### 3.1 `A_r`: Gelfand–Tsetlin basis

Write `n = r + 1` and use the 1-based matrix units `E_{ij}` of `gl_n`.

#### 3.1.1 Patterns

A **Gelfand–Tsetlin pattern** `Λ` is a triangular array of integers
`λ_{ki}`, `1 ≤ i ≤ k ≤ n`, with top row `λ_{ni} = λ_i` (§2.5) and the
betweenness conditions

```
λ_{k,i} ≥ λ_{k−1,i} ≥ λ_{k,i+1}      (2 ≤ k ≤ n, 1 ≤ i ≤ k−1).
```

In code, `λ_{ki} = GtPattern::get(i, k)`. The patterns of `λ` index a basis of
`L(λ)`.

The `gl_n` weight of `Λ` is `w_k = Σ_i λ_{ki} − Σ_i λ_{k−1,i}` (with an empty
row `k = 0` summing to `0`). Its Dynkin weight is `μ_j = w_{j+1} − w_{j+2}`
for `j = 0, …, r−1`.

#### 3.1.2 Generators and Molev's unnormalized basis

The Chevalley generators are `e_j = E_{j+1, j+2}` and `f_j = E_{j+2, j+1}`
(`j` 0-based). Let `l_{ki} = λ_{ki} − i + 1`, and let `Λ ± δ_{ki}` be `Λ` with
`λ_{ki}` changed by `±1` (a term whose pattern violates betweenness is
absent). Molev's unnormalized vectors `ξ_Λ` (A. I. Molev, *Gelfand–Tsetlin
bases for classical Lie algebras*, arXiv:math/0211289, Theorem 2.3) satisfy

```
(2.6)  E_{k,k+1} ξ_Λ = − Σ_{i=1}^{k}  [ Π_{j=1}^{k+1} (l_{ki} − l_{k+1,j}) / Π_{j≠i, j≤k} (l_{ki} − l_{kj}) ]  ξ_{Λ+δ_{ki}}
(2.7)  E_{k+1,k} ξ_Λ =   Σ_{i=1}^{k}  [ Π_{j=1}^{k−1} (l_{ki} − l_{k−1,j}) / Π_{j≠i, j≤k} (l_{ki} − l_{kj}) ]  ξ_{Λ−δ_{ki}}
```

for `k = 1, …, n−1`. So `f_j` acts by (2.7) with `k = j + 1`, and every
matrix element is rational. `ξ_Λ` for the pattern with every row maximal is
`v_λ`. The vectors are orthogonal for the contravariant form
(`E_{ij}^* = E_{ji}`), with norms (Molev, Proposition 2.4)

```
⟨ξ_Λ, ξ_Λ⟩ = Π_{k=2}^{n} [ Π_{1≤i≤j<k} (l_{ki} − l_{k−1,j})! / (l_{k−1,i} − l_{k−1,j})!
                          · Π_{1≤i<j≤k} (l_{ki} − l_{kj} − 1)! / (l_{k−1,i} − l_{kj} − 1)! ] .
```

Any exact evaluation of these norms gives the same values. Propagating them
from `⟨v_λ, v_λ⟩ = 1` by contravariance, `n_t = A·n_s / B` where `E_{k,k+1} ξ_t = A ξ_s + …` and
`E_{k+1,k} ξ_s = B ξ_t + …` for `s = t + δ_{ki}`, is a recommended method.

#### 3.1.3 The orthonormal basis and its signs

The basis vector of `Λ` is `ζ_Λ = ξ_Λ / √⟨ξ_Λ, ξ_Λ⟩`.

**Every matrix element of every `e_j` and `f_j` is nonnegative** in this
basis. Proof: the norms are positive, so it suffices to check the
coefficients of (2.6) and (2.7) on valid patterns. In (2.7) the numerator
factors with `j < i` are negative (`λ_{k−1,j} ≥ λ_{k,i}` and `j < i`) and the
others are nonnegative, so the numerator has sign `(−1)^{i−1}`; the
denominator factors with `j < i` are negative and the others positive
(`l_{k1} > l_{k2} > …`), so the denominator also has sign `(−1)^{i−1}`. In
(2.6) the numerator has `i` negative factors (`j ≤ i`), the denominator
`i − 1`, and the leading minus makes the coefficient positive. ∎

This is the orthonormal Gelfand–Tsetlin basis of SUNRepresentations.jl
v0.4.0 (`src/gtpatterns.jl::creation`, matrix elements `signedroot(coef)`),
entry for entry (§8 oracle 2).

#### 3.1.4 Magnetic index order

Patterns are ordered **lexicographically ascending** on their entries read top
row first (`λ_{n1}, …, λ_{nn}`, then `λ_{n−1,1}, …`, down to `λ_{11}`). This
is SUNRepresentations' `GTPatternIterator` order and the order of the public
`LieIrrep::gt_patterns()`. The magnetic index `m` is the position in this
order. The **lowest weight** (every row minimal) is `m = 0` and the
**highest weight** (every row maximal) is `m = d − 1`.

For `A_1` the patterns are `(λ_1, 0; x)` with `x = 0, …, λ_1`, the weight is
`m = x − j`, and the basis is the Condon–Shortley `|j, m⟩`
(`J⁻|j, m⟩ = √((j+m)(j−m+1)) |j, m−1⟩`). The magnetic index is `m + j`.

### 3.2 `B_r`–`E8`: Gram–Schmidt basis of the contravariant form

#### 3.2.1 Weight order

For a weight `μ` of `L(λ)`, write `λ − μ = Σ_i k_i α_i` with integers
`k_i ≥ 0` and call `k(μ) = (k_0, …, k_{r−1})` its **root-coefficient vector**.
Its **depth** is `dep(μ) = ⟨λ − μ, ρ∨⟩ = Σ_i k_i`. Weights are ordered by

1. `dep(μ)` **ascending** (the highest weight comes first); then
2. `k(μ)` **ascending lexicographically**, `k_0` most significant.

Distinct weights of equal depth have distinct `k`, so this is a total order.
The magnetic index lists the weight spaces in this order; within a weight
space it follows the selection order of §3.2.3. Index `0` is `v_λ`. The lowest
weight `w₀λ` has a one-dimensional weight space and depth
`h_λ = ⟨λ, 2ρ∨⟩`, so it is the last index `d − 1`.

#### 3.2.2 Candidates

The basis is built level by level in the order of §3.2.1, starting from
`u_0 = v_λ`. For a weight `μ` at depth `ℓ ≥ 1`, the **candidates** are the
vectors `f_j w_q` for

- `j = 0, 1, …, r−1` **ascending** (outer loop), keeping only `j` for which
  `μ + α_j` is a weight of `L(λ)`; and
- `q` **ascending** over the basis vectors `w_q` of the weight space `μ + α_j`
  in their §3.2.3 order (inner loop).

These candidates span `L(λ)_μ`, because `L(λ) = U(n⁻)v_λ`.

#### 3.2.3 Greedy selection and exact Gram–Schmidt

Walk the candidates in §3.2.2 order and **keep** a candidate exactly when it is
not in the span of those already kept, decided by exact rank. The kept
candidates `c_1, …, c_m` are orthonormalized in that order:

```
b_k = c_k − Σ_{l<k} (⟨c_k, b_l⟩ / ⟨b_l, b_l⟩) b_l ,      u_k = b_k / √⟨b_k, b_k⟩ .
```

Each `u_k` has a **positive** coefficient on its own candidate `c_k`. A
candidate is kept exactly when its residual `b_k` has nonzero norm.

**Gate.** `m` must equal the Freudenthal multiplicity `m_λ(μ)`; a mismatch is
a typed generation error.

#### 3.2.4 Computing the form

The form needs no module realization. For `w ∈ L(λ)_{μ+α_j}` and
`w′ ∈ L(λ)_{μ+α_k}`,

```
⟨f_j w, f_k w′⟩ = ⟨e_k w, e_j w′⟩ + δ_jk (μ + α_j)_j ⟨w, w′⟩ ,
```

with `e_k w`, `e_j w′` at depth `ℓ − 2`. This recursion is a recommended
method, not a rule: §3.2.2–§3.2.3 depend only on the form.

### 3.3 Ladder entries are single signed square roots of rationals

**Claim.** In either basis, every matrix entry `⟨u_a, f_j u_b⟩` (and so every
entry of `e_j`) is `s·√q` with `s ∈ {−1, 0, +1}` and `q ≥ 0` rational. It is
therefore exactly an `exact.rs::SignedSqrtRational`.

**Proof.** It suffices to have an orthogonal basis `b_k` with rational norms
`n_k = ⟨b_k, b_k⟩` in which every `f_j` has rational coefficients
`f_j b_b = Σ_a F[a, b] b_a`; then `u = b/√n` gives
`⟨u_a, f_j u_b⟩ = F[a, b]·√(n_a/n_b) = sign(F)·√(F[a, b]²·n_a/n_b)`.

- `A_r`: `b = ξ` (§3.1.2). `F` is (2.7) and `n` is Proposition 2.4.
- Other types: `b` is the unnormalized Gram–Schmidt basis of §3.2.3. By
  induction on depth, `E[b, a] = F[a, b]·n_a/n_b` (contravariance) is
  rational, so by §3.2.4 the next level's candidate Gram matrix is rational,
  Gram–Schmidt on it gives rational `b_k = Σ_t r_{kt} c_t` and rational `n_k`,
  and `F[a, b] = ⟨f_j b_b, b_a⟩/n_a` is rational. ∎

Implementations store `(F, n)` exactly and never add two square roots. The
claim does not extend to F and R symbols, which are sums of products of four
CGC (§6).

## 4. Tensor products and CGC

### 4.1 Coupling pairs and their order

The coproduct is `Δ(x) = x ⊗ 1 + 1 ⊗ x`; the inner product of
`L(a) ⊗ L(b)` is the product form, so `{u^a_{m_a} ⊗ u^b_{m_b}}` is
orthonormal.

For a channel `a ⊗ b → c` and a weight `w`, let `P_w` be the coupling pairs
`(m_a, m_b)` with `wt(m_a) + wt(m_b) = w`, ordered **lexicographically by
`(m_a, m_b)` ascending: the left factor is slow** ("a-slow"). The indices are
the public magnetic indices of §3. This single rule applies to every simple
type.

### 4.2 Highest-weight space

Let `P = P_{λ_c}`. The **highest-weight space** is

```
H = { x ∈ span(P) : Δ(e_i) x = 0 for every i },
```

the exact rational nullspace of the stacked `Δ(e_i)` restricted to `span(P)`;
the order of the stacked rows has no effect. **Gate:** `dim H` must equal the
Brauer–Klimyk multiplicity `N^c_{ab}`.

### 4.3 Multiplicity basis

Index coordinates by `P` in its §4.1 order. The **pivot rows**
`p_1 < … < p_N` are the rows `p` where
`dim{x ∈ H : x[t] = 0 for all t < p}` drops by one at `p + 1`. The **echelon
basis** `E_1, …, E_N` of `H` is the unique basis with `E_k[p_l] = δ_kl` (so
`E_k[t] = 0` for `t < p_k`); equivalently, `Eᵀ` is the reduced row echelon
form, pivots `+1`, of any matrix whose rows span `H`.

The multiplicity vectors are the Gram–Schmidt orthonormalization of
`E_1, …, E_N` **in pivot order**:

```
Ψ_μ = GS(E_1, …, E_N)_{μ+1},     μ = 0, …, N−1 .
```

Consequently `Ψ_μ[p_{μ+1}] > 0` and `Ψ_μ[p_l] = 0` for `l > μ + 1`. At the
earlier pivots `Ψ_μ` is generally nonzero (in `G2 27⊗27 → 27`,
`Ψ_1[p_1] ≠ 0`). The outer-multiplicity index `μ` is the pivot order. Unit
norm and the positive pivot do not by themselves fix the basis for `N ≥ 2`
(Löwdin orthonormalization or reverse-order Gram–Schmidt satisfy them too);
the rule is the pivot-order Gram–Schmidt.

**Coordinate independence.** An implementation may work in the rational
coordinates `b^a_{m_a} ⊗ b^b_{m_b}` (§3.3) with metric
`diag(n^a_{m_a} n^b_{m_b})`. Changing coordinates scales each row by a
positive factor, which keeps the pivots, scales each `E_k` positively and
leaves the Gram–Schmidt output unchanged.

### 4.4 Lowering: the intertwiner

For each `μ`, let `Φ_μ : L(c) → L(a) ⊗ L(b)` be the unique intertwiner with
`Φ_μ(v_{λ_c}) = Ψ_μ`. The pipeline CGC is

```
C°[m_a, m_b, m_c, μ] = ⟨u^a_{m_a} ⊗ u^b_{m_b}, Φ_μ(u^c_{m_c})⟩ .
```

`Φ_μ` is well defined: `Ψ_μ` is a singular vector in a completely reducible
module, so it generates a copy of `L(λ_c)` and every relation among
`f`-words on `v_{λ_c}` also holds on `Ψ_μ`. `Φ_μ` is an isometry, and copies
are orthogonal: `Φ_μᵀΦ_ν` commutes with `L(c)`, so by Schur's lemma it is
`⟨Ψ_μ, Ψ_ν⟩ = δ_μν`.

Every CGC into `c`, from every product, is expressed in `c`'s own basis of §3.
There is no Gram–Schmidt on the coupled side and no frame alignment.

**Computing `Φ_μ`** (recommended method, not a rule). Express each
unnormalized basis vector of `c` as a rational combination of lowered
vectors, `b^c_k = Σ_t R_{kt} f_{j_t} b^c_{q_t}`, and replay it in the product:
`Φ_μ(b^c_k) = Σ_t R_{kt} Δ(f_{j_t}) Φ_μ(b^c_{q_t})`,
`Φ_μ(u^c_k) = Φ_μ(b^c_k)/√n^c_k`. For §3.2 bases `R` is the Gram–Schmidt
recipe of §3.2.3; for §3.1 any rational recipe (for example a greedy choice
of candidates and an exact inverse) gives the same `Φ_μ`.

### 4.5 Complete sign and normalization rules

Every CGC sign and normalization follows from:

1. **Irrep basis** (§3): orthonormal, with the highest-weight vector `+v_λ`;
   `A_r` signs by §3.1.3, other types by the positive own-candidate rule of
   §3.2.3.
2. **Highest-weight multiplicity vectors** (§4.3): pivot-order Gram–Schmidt
   of the echelon basis over the a-slow pair order.
3. **Intertwiner** (§4.4) for everything else.
4. **`A_1` vertex sign** (§4.6), for `A_1` only.

Consequences with no special case:

- **Trivial couplings.** For `1 ⊗ s → s` and `s ⊗ 1 → s`, `P` has one pair,
  `Ψ = +1`, and the intertwiner reproduces `s`'s basis:
  `C[0, m, m, 0] = C[m, 0, m, 0] = 1` exactly.
- **No exchange symmetry is imposed.** `a ⊗ b → c` and `b ⊗ a → c` are each
  built by these rules; their relation is the R symbol.

### 4.6 The `A_1` vertex sign

For `A_1` the returned CGC is

```
C[m_1, m_2, m_3, 0] = (−1)^{j_1 + j_2 − j_3} · C°[m_1, m_2, m_3, 0],      (−1)^{j_1+j_2−j_3} = (−1)^{(a_1 + a_2 − a_3)/2},
```

applied once per channel. **This is exactly the Condon–Shortley CGC**
`⟨j_1 m_1; j_2 m_2 | j_3 m_3⟩`. TensorKitSectors `SU2Irrep` `fusiontensor`
has the same values with descending projections: its arrays equal these after
reversing all three magnetic axes (crate index `i = m + j` is TensorKit's
1-based index `d − i`).

Proof. The §3.1 basis for `A_1` is the Condon–Shortley basis, and both
conventions descend from the highest-weight vector by the same positive
`J⁻`, so they can differ only in the sign of the highest-weight vector. Its
components alternate, `⟨j_1 m_1; j_2 J−m_1 | J J⟩ ∝ (−1)^{j_1 − m_1}`, and
Condon–Shortley makes the `m_1 = j_1` component positive. The a-slow pivot
(§4.3) is the smallest admissible `m_1`, which is `J − j_2` (because
`J ≥ j_2 − j_1`); its Condon–Shortley sign is `(−1)^{j_1+j_2−J}`. ∎

The sign is exact, so the returned values equal the closed-form
`su2::clebsch_gordan` bit for bit (both are the exact value rounded once).
F and R are unaffected: in the F contraction the four vertex signs multiply to
`(−1)^{2(j_a+j_b+j_c−j_d)} = 1`, and in R they cancel. `A_1` F, R and `κ` use
the closed forms anyway ([`gauge.md` §1](gauge.md#1-su2-closed-forms)).

This sign is the only type-specific rule after the irrep basis. It applies to
every `A_1` presentation: SU(2), SO(3), Sp(2), PSU(2), `B_1`, `C_1` and each
factor of `D_2`.

### 4.7 Exactness, rounding and entry order

`Ψ_μ = B_μ/√N_μ` with `B_μ` rational in the rational coordinates and
`N_μ = ⟨B_μ, B_μ⟩` rational. The intertwiner keeps coordinates rational, so
each CGC entry is `x·√(n^a_{m_a} n^b_{m_b} / (N_μ n^c_{m_c}))` with `x`
rational: a single signed √rational. Its `f64` value is the conversion of
`SignedSqrtRational::to_f64` (`src/exact.rs`), correctly rounded (to nearest,
ties to even, once) over the whole `f64` range including subnormal results and
underflow to a signed zero, and is
**the only rounding of a CGC**. The conversion is a function of the radicand's
value alone, so it is applied to the integer ratio `U/V` of the products of
the operands' numerators and denominators (`U = p_x²·p_a·p_b·q_t`,
`V = q_x²·q_a·q_b·p_t` for `x = p_x/q_x`, `n^a = p_a/q_a`, `n^b = p_b/q_b`,
`N_μ n^c = p_t/q_t`) without reducing it; the bits equal those of the
reduced radicand, including at ties and in the subnormal range. Only exactly zero
coefficients are omitted; an exactly nonzero coefficient that rounds to a
signed zero keeps its entry. An absent entry reads `+0.0`. There is no purge
threshold.

*Specification correction (#262).* The former rule was correctly rounded only
for exact magnitudes at least `2^-1022`: a smaller value was first rounded to
a 53-bit significand and then rounded again to the subnormal spacing
`2^-1074`, which can miss the nearest `f64` by one subnormal step (or return
the smallest normal `2^-1022` for a value just below it). The rule is now one
rounding on the `2^-1074` grid below `2^-1022`; values of magnitude at least
`2^-1022` are unchanged.

The public block `GroupRepresentations::cgc(a, b, c)` has axes
`[m_a, m_b, m_c, μ]` with lengths `[d_a, d_b, d_c, N]`, indexed as in §3.

The **entry list** of a channel is its exactly nonzero entries
`(m_a, m_b, m_c, μ, value)` in the order: `μ` ascending (outer), then the
public `m_c` ascending, then `(m_a, m_b)` in the §4.1 order. This order is
normative: the floating-point contractions of §6.2 consume entry lists in
this order, and their bits depend on the order within each bucket.

## 5. Duality and Frobenius–Schur phase

### 5.1 Dual label

The dual of `L(λ)` is `L(−w₀λ)`. On Dynkin labels `−w₀` is:

| Type | `−w₀` on labels (Bourbaki numbering) |
|---|---|
| `A_r` | reversal `a_i ↔ a_{r+1−i}` |
| `D_r`, `r` odd | `a_{r−1} ↔ a_r` |
| `E6` | `a₁ ↔ a₆`, `a₃ ↔ a₅` |
| `B_r`, `C_r`, `D_r` (`r` even), `G2`, `F4`, `E7`, `E8` | identity |

**Dual bases are not transported.** `L(−w₀λ)` is built from its own label by
§3, not by carrying `L(λ)`'s basis through the diagram automorphism. The CGC
`a ⊗ ā → 1` and `ā ⊗ a → 1` are each built by §4; their relation is `κ_a`
(§5.3).

### 5.2 Frobenius–Schur indicator

For a self-dual irrep the indicator is `(−1)^{⟨λ, 2ρ∨⟩}`, the value of the
central element `exp(2πi ρ∨)` (Bourbaki Ch. VIII §7). With
`2ρ∨ = Σ_i c_i α_i∨`, `⟨λ, 2ρ∨⟩ = Σ_i c_i a_i`. The `c_i` solve
`Σ_i cartan(j, i) c_i = 2` for every `j` (1-based `i` below):

| Type | `c_i` |
|---|---|
| `A_r` | `i(r + 1 − i)` |
| `B_r` | `i(2r + 1 − i)` for `i < r`; `c_r = r(r + 1)/2` |
| `C_r` | `i(2r − i)` |
| `D_r` | `i(2r − 1 − i)` for `i ≤ r − 2`; `c_{r−1} = c_r = r(r − 1)/2` |
| `G2` | `(6, 10)` |
| `F4` | `(22, 42, 30, 16)` |
| `E6` | `(16, 22, 30, 42, 30, 16)` |
| `E7` | `(34, 49, 66, 96, 75, 52, 27)` |
| `E8` | `(92, 136, 182, 270, 220, 168, 114, 58)` |

Examples: the `Sp(2r)` defining irrep (`c_1 = 2r − 1`) and the SU(2) spin-½
irrep are pseudo-real; the `Spin(2r+1)` spinor is pseudo-real exactly when
`r(r+1)/2` is odd; E7 irreps are pseudo-real exactly when `a₂ + a₅ + a₇` is
odd (`56`, `912`), consistent with Slansky (1981), Phys. Rep. 79, reality
tables. The indicator of a non-self-dual irrep is `0`.

### 5.3 The phase `κ_a` in this gauge

`FusionCategory::frobenius_schur_phase` is the phase of
`F^{a ā a}_a[𝟙, 𝟙]` (`src/category.rs`; TensorKitSectors
`frobenius_schur_phase_from_Fsymbol`). `A^{ab}_c` uses it.

**Reduction.** With `e = f = 𝟙` and the `d = a` leg sliced at `m_d = 0`
(§6), the trivial couplings of §4.5 reduce the F contraction to

```
F^{a ā a}_a[𝟙,𝟙] = Σ_m C^{a ā → 1}[0, m, 0, 0] · C^{ā a → 1}[m, 0, 0, 0] .
```

Write `ψ_{xy}` for the singlet in `L(x) ⊗ L(y)`. Swapping the factors of `ψ_{aā}` gives
`s_a ψ_{āa}` with `s_a = ±1`, so `F = s_a/d_a` and **`κ_a = s_a`**.

**Pivot.** The singlet's `P` consists of pairs of opposite weights. Under the
a-slow order its first pair is (`m_a` minimal, partner of opposite weight):
`(high_a, low_ā)` for §3.2 bases, `(low_a, high_ā)` for §3.1 bases. Both are
one-dimensional weight spaces whose singlet component is nonzero, so by §4.3
that component of `ψ` is positive.

**Closed form.** Let `L = Σ_i f_i` and `h = ⟨λ, 2ρ∨⟩` (the depth of the lowest
weight of both `a` and `ā`). Invariance gives `(L^h ⊗ 1)ψ = (−1)^h (1 ⊗ L^h)ψ`;
taking the `(low_a, low_ā)` component relates the two extreme components of
`ψ`. Write `σ_x = sign ⟨low_x| L^h |high_x⟩` in `x`'s own basis; this element
is never zero (`Σ_i c′_i f_i` with the positive principal-`sl₂` coefficients,
Kostant 1959, is conjugate to `L` by a positive torus element, and
`L^h v_λ ≠ 0`). Combining the `a ⊗ ā` and `ā ⊗ a` pivots gives, for either
basis family,

```
κ_a = s_a = (−1)^{⟨λ, 2ρ∨⟩} · σ_a · σ_ā .
```

**Normative rule.**

- **Self-dual irreps.** `κ_a = (−1)^{⟨λ, 2ρ∨⟩}` (§5.2); no CGC needed.
- **`A_r`.** Every `f_j` entry is nonnegative (§3.1.3) and
  `⟨low|L^h|high⟩ ≠ 0`, so `σ = +1` and
  `κ_a = (−1)^{⟨λ, 2ρ∨⟩}` for every irrep, complex ones included.
- **`D_r` (`r` odd) complex irreps.** `κ_a = (−1)^{⟨λ, 2ρ∨⟩} σ_a σ_ā`, with
  `σ_a`, `σ_ā` computed exactly (`L^h v_λ` in the rational basis of §3.3;
  positive rescalings keep the sign). The parity factor does not drop out:
  for `D_3` the half-spinors `[0,0,1]` and `[0,1,0]` are minuscule
  (`σ = +1`) with `⟨λ, 2ρ∨⟩ = 3`, so `κ = −1`; for `D_5` the half-spinors
  have `⟨λ, 2ρ∨⟩ = 10` and `κ = +1`. In general the half-spinor exponent is
  `r(r−1)/2`, odd exactly when `r ≡ 3 (mod 4)`.
- **`E6` complex irreps.** Every `E6` coefficient `c_i` is even, so the same
  rule reduces to `κ_a = σ_a σ_ā`.
- In both cases `κ_ā = κ_a`, and the computation is replaced by the closed
  form only after the conjecture below is proved and this spec amended.
- **Indicator.** `κ_a` when `a = ā`, otherwise `0`.

**Minuscule irreps (proved, §3.2 bases).** If every weight is Weyl-conjugate
to `λ`, every `f_i` entry of the §3.2 basis is `0` or `+1`, so `σ = +1`.
Proof by induction on depth: if `μ + α_i` is a weight then
`⟨μ + α_i, α_i∨⟩ = 1`, so `‖f_i u_{μ+α_i}‖ = 1`; `u_μ = f_k u_{μ+α_k}` for the
first such `k`; for another such `i`, `⟨μ + α_i, α_k∨⟩ = −1 + cartan(i, k)`
must lie in `{−1, 0, 1}`, so `cartan(i, k) = 0`; then
`τ = μ + α_i + α_k` is a weight with `u_{μ+α_i} = f_k u_τ`,
`u_{μ+α_k} = f_i u_τ`, and `f_i u_{μ+α_i} = f_k f_i u_τ = u_μ`. ∎
This covers the `D_r` half-spinors `ω_{r−1}, ω_r`, the `B_r` spinor, the
`C_r` defining irrep, `E6 27, 27̄` and `E7 56`.

**Conjecture (not normative): `σ_a = +1` for every irrep in a §3.2 basis.**
If true, `κ_a = (−1)^{⟨λ, 2ρ∨⟩}` for every irrep of every type. It is
consistent with the exceptional irreps `G2 (a, b)`, `a ≤ 4`, `b ≤ 3`;
`F4 26, 52, 273, 324, 1053, 1274, 4096`; `E6 27, 351, 351′, 1728, 3003, 7371`
with duals, and `78`; `E7 56, 133, 1463, 1539`; `E8 248`; and with
`σ_a σ_ā = +1` for the `D_3` labels `[0,0,1]`, `[0,1,0]`, `[1,0,1]`,
`[0,0,2]`, `[1,1,0]`, `[2,0,1]` and the `D_5` labels `[0,0,0,0,1]`,
`[0,0,0,1,0]`, `[1,0,0,1,0]`, `[0,0,1,0,0]`, `[0,0,0,0,2]`. Replacing the
computation by the closed form after a proof is a spec amendment with
identical values. `σ` depends only on the irrep
basis, not on the pair order of §4.1.


## 6. F, R, A and B

### 6.1 Definitions

For every simple type except `A_1`, F and R are contractions of four and two
CGC (TensorKitSectors `Fsymbol_from_fusiontensor`, SUNRepresentations
`sector.jl::{_Fsymbol,_Rsymbol}`):

- **F.** With `A = C(a,b,e)`, `B = C(e,c,d)`, `Cb = C(b,c,f)`, `D = C(a,f,d)`,
  and the `d` leg of `B` and `D` fixed at public index `m_d = 0`,

  ```
  F^{abc}_d[e,f][μ,ν,κ,λ] = Σ_{m_a,m_b,m_c,m_e,m_f} A[m_a,m_b,m_e,μ] B[m_e,m_c,0,ν] Cb[m_b,m_c,m_f,κ] D[m_a,m_f,0,λ] .
  ```

  The block has axes `[μ, ν, κ, λ]` with lengths
  `[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`. F does not depend on the slice;
  the bits do, so the slice is part of the gauge (it is the highest weight for
  §3.2 bases and the lowest for §3.1).
- **R.** `R^{ab}_c[μ,ν] = Σ_{m_a,m_b} C(a,b,c)[m_a,m_b,0,μ] · C(b,a,c)[m_b,m_a,0,ν]`,
  axes `[μ, ν]`. The CGC are real, so conjugation is the identity.
- **A, B.** The TensorKitSectors forms, for every simple type including `A_1`:
  `A^{ab}_c = (√d_a √d_b / √d_c) · κ_a · F^{ā a b}_b[𝟙, c]` and
  `B^{ab}_c = (√d_a √d_b / √d_c) · F^{a b b̄}_a[c, 𝟙]`, each read as an
  `N(a,b,c) × N(·)` block from the two nontrivial F axes.
- **Twist and braiding**: bosonic, twist `1`.
- **Empty channels**: a block with a zero multiplicity is all `+0.0`, shaped by
  the vertex multiplicities.
- **`A_1`**: F, R and `κ` are the closed forms of
  [`gauge.md` §1](gauge.md#1-su2-closed-forms), not contractions.

### 6.2 Floating-point evaluation (normative)

Every operation below is one IEEE 754 binary64 operation rounded to nearest,
ties to even. No operation is fused (no FMA), reassociated or reordered.
Accumulators start at `+0.0`. Inputs are the CGC entry lists of §4.7 in their
normative order; "in list order" below means that order, restricted by the
stated filter.

**F** (`frcore::f_block_raw` implements this recipe):

1. `Bl[m_e]`: the entries of `B = C(e,c,d)` with `m_d = 0`, in list order, grouped by their first index `m_e`; each item is `(m_c, ν, v_B)`.
2. `Al[m_e]`: the entries of `A` whose coupled index `m_e` has a nonempty
   `Bl[m_e]`, in list order, grouped by `m_e`; each item is `(m_a, m_b, μ, v_A)`.
3. `AB`: for `m_e` ascending, for each item of `Al[m_e]` in order, for each
   item of `Bl[m_e]` in order, append `(μ, ν, v_A ⊗ v_B)` to the list keyed
   `(m_a, m_b, m_c)` (`⊗` is the rounded product).
4. `Dl[m_f]`: the entries of `D` at coupled index `0`, in list order, grouped
   by their second index `m_f`; each item is `(m_a, λ, v_D)`.
5. `Cl[m_f]`: the entries of `Cb` whose coupled index `m_f` has a nonempty
   `Dl[m_f]`, in list order; each item is `(m_b, m_c, κ, v_C)`.
6. `CD`: for `m_f` ascending, for each item of `Cl[m_f]` in order, for each
   item of `Dl[m_f]` in order, if the key `(m_a, m_b, m_c)` exists in `AB`,
   append `(κ, λ, v_C ⊗ v_D)` to the list keyed `(m_a, m_b, m_c)`.
7. For the keys of `AB` in ascending lexicographic `(m_a, m_b, m_c)` order
   that also exist in `CD`, for each `(μ, ν, x)` of `AB[key]` in order, for each
   `(κ, λ, y)` of `CD[key]` in order:
   `F[μ,ν,κ,λ] ← F[μ,ν,κ,λ] ⊕ (x ⊗ y)`.

**R** (`frcore::r_block_raw`): group the entries of `C(a,b,c)` at coupled
index `0` by `(m_a, m_b)` and those of `C(b,a,c)` at coupled index `0` by
`(m_a, m_b) = (its m_2, its m_1)`, each in list order. For keys ascending,
for each `(μ, x)` of the first group in order and each `(ν, y)` of the second
in order: `R[μ,ν] ← R[μ,ν] ⊕ (x ⊗ y)`.

**A and B.** `d_x` as binary64 is the exact dimension rounded to nearest;
`s_x = sqrt(d_x)` (correctly rounded); the scale is `(s_a ⊗ s_b) ⊘ s_c`, in
that association. Each entry is `scale ⊗ (κ_a ⊗ F)` for A and `scale ⊗ F`
for B, in the row-major order of the F block. For `A_1`, `F` is the closed
form of [`gauge.md` §1](gauge.md#1-su2-closed-forms). (For example
`(√8·√8)/√1` and `√(8·8/1)` give different binary64 scales; only the first is
this gauge.)

**Contract.** All CGC values are exact until one correctly rounded conversion
per entry (§4.7), and F, R, A and B follow this evaluation, so every build
returns **bit-identical CGC, F, R, A and B values** for every simple type. No
dense linear-algebra backend enters any value.

## 7. Generation gates (normative)

Every generation performs these checks on the production path, in every
build, before the value is stored in a cache or returned. A failure is a
typed `LieError::Generation`; a value that failed a gate is never returned.
Every operation that can generate propagates it, including
`FusionCategory::frobenius_schur_phase` and the A symbol through it; no
phase is ever substituted. The checks never alter a value.

Exact gates (rank decisions are made in exact arithmetic, so no tolerance
enters them):

1. **Module dimension.** §3.1: the pattern count equals the Weyl dimension.
   §3.2: each weight space's selected rank equals the Freudenthal multiplicity
   `m_λ(μ)`, every kept residual has positive norm, and the total equals the
   Weyl dimension.
2. **Highest-weight nullity.** `dim H` equals the Brauer–Klimyk multiplicity
   `N^c_{ab}` (§4.2), for every requested channel, in both directions (a
   channel with `N = 0` has no CGC; a channel with `N > 0` has exactly `N`).
   A direct CGC request verifies the nullity even when `N = 0`, unless an
   exact certificate (an empty coupling-pair set `P`) proves it zero. A
   zero-vertex F or R query requests no CGC.
3. **Principal ladder.** Where §5.3 computes `σ_x`,
   `⟨low_x| L^h |high_x⟩ ≠ 0`.

Floating-point gates on each rounded CGC channel (verification tolerances,
not value-fixing; tightening them is not a gauge change):

4. **Isometry.** With `C` reshaped as `M[(m_a,m_b), (m_c,μ)]`,
   `max |MᵀM − I| ≤ TOL_ORTHO = 1e-9`.
5. **Ladder consistency.** For each `μ` and each simple `j` with
   `λ_c − α_j` a weight of `c`: evaluate in binary64, with correctly rounded
   ladder entries of `a`, `b` and `c`, `y = Δ(f_j) C[·,·,top,μ]` and
   `z = Σ_m ⟨u^c_m, f_j u^c_top⟩ C[·,·,m,μ]` over the weight space
   `λ_c − α_j`, where `top` is the public index of `c`'s highest weight;
   require `max |y − z| ≤ TOL_LADDER = 1e-9`.

Floating-point gates on each F and R block, before it is cached or
returned (same failure behavior; a failing block is never repaired or
re-orthogonalized):

6. **R orthogonality.** Each nonempty `R^{ab}_c` is the complete change of
   orthonormal multiplicity basis induced by the factor flip: with `A`, `B`
   the two CGC sliced at `m_c = 0` and `P` the flip, `R = AᵀPB`, exact
   intertwining gives `PB = AR`, so `RᵀR = I`. Require every entry finite and
   `max_{μ,ν} |Σ_ρ R[ρ,μ] R[ρ,ν] − δ_{μν}| ≤ TOL_UNITARY = 1e-9`; a nonfinite
   residual fails. An empty block passes vacuously. The `A_1` closed-form
   values `±1` satisfy the identity exactly. The check costs `O(N³)` on the
   computed block and generates nothing else.
7. **F finiteness.** Every entry of each F block is finite. A fixed
   `F^{abc}_d[e,f]` is only a subblock of the change of basis between the
   `e`- and `f`-channel bases, so it has no unitarity of its own.

F-unitarity, pentagon and hexagon are category-level identities that couple
many blocks. They are checked through
`category::{f_unitarity_residual, pentagon_residual, hexagon_residual}` by
callers and by §8 oracle 7, not at generation time.

## 8. Validation oracles

An implementation must pass each oracle below. An oracle must not use the
pipeline's own descriptors to compute expected values.

1. **Weyl dimension.** `dim L(λ)` equals the Weyl formula. Fundamental
   dimensions in Bourbaki order:
   - `G2`: `7, 14`; `F4`: `52, 1274, 273, 26`;
   - `E6`: `27, 78, 351, 2925, 351, 27`;
   - `E7`: `133, 912, 8645, 365750, 27664, 1539, 56`;
   - `E8`: `3875, 147250, 6696000, 6899079264, 146325270, 2450240, 30380, 248`;
   - classical series as in the [label guide](user-guide/labels.md#common-irreps).
2. **Basis.** Every `f_j` entry of §3.1 equals SUNRepresentations v0.4.0
   `gtpatterns.jl::creation` (`signedroot(coef)`) exactly, in square and sign,
   and the norms equal Proposition 2.4.
3. **Brauer–Klimyk N.** Literal rows: `G2 7⊗7 = 1 + 7 + 14 + 27`;
   `G2 14⊗14 = 1 + 14 + 27 + 77(0,2) + 77(3,0)`;
   `G2 27⊗27 = 1 + 7 + 14 + 2·27 + 2·64 + 77(0,2) + 77(3,0) + 182 + 189`
   (the first G2 channel with `N ≥ 2`); `F4 26⊗26 = 1 + 26 + 52 + 273 + 324`;
   `E6 27⊗27̄ = 1 + 78 + 650`; `E7 56⊗56 = 1 + 133 + 1463 + 1539`;
   `E8 248⊗248 = 1 + 248 + 3875 + 27000 + 30380`; the B/C/D product rows
   of Oscar 1.8.0; `D3 (0,1,1)⊗(0,1,1)` and `Sp(4) 35⊗35` contain `N ≥ 2`
   channels.
4. **Generator relations.** Every built irrep satisfies §2.3 exactly in the
   rational basis of §3.3.
5. **Orthonormality and intertwining, exactly.** Each CGC is an exact
   isometry before rounding; `Δ(x)Φ_μ = Φ_μ x` exactly for every `e_i`, `f_i`;
   `C = 0` unless `wt(m_a) + wt(m_b) = wt(m_c)`.
6. **References.**
   - SU(2): every block CGC equals `su2::clebsch_gordan` bit for bit (§4.6).
   - SU(N): signed CGC (including the `μ` order of `SU(3) 8⊗8 → 8`), F and R
     agree with SUNRepresentations.jl v0.4.0 within `1e-12`.
7. **Category gates.** Pentagon, hexagon and F-unitarity residuals over
   channels with `N ≥ 2` for every family (`SU(3) 8`, `G2 27`, `Sp(4) 35`,
   `D3 (0,1,1)`).
8. **Frobenius–Schur.** `κ_a = sign(d_a F^{aāa}_a[𝟙,𝟙])`, with F from §6.2,
   for representative self-dual, minuscule and complex irreps of every type,
   including `D3 [0,0,1]` and `[0,1,0]` (`κ = −1`), `D5` half-spinors
   (`κ = +1`), and `E6 27, 27̄, 351, 351′`; the indicators of `E7 56`, `912`
   and `Sp(4) 4` are `−1`.
9. **Bit identity.** Golden values (`tests/gauge_golden.rs`) pin CGC, F, R,
   A and B entries for every family, including at least one `N ≥ 2` channel
   for every family that has one (`A_1` is multiplicity free, so its goldens
   are all `N = 1`), and compare them **bitwise**. The E8 `N ≥ 2` golden
   (`3875 ⊗ 30380 → 30380`, `N = 2`, CGC and R) is required like the others
   but, at about 31 GB peak RSS and 19 minutes, runs as a manual pre-release
   gate on a machine with at least 48 GB instead of in CI
   ([developer guide](developer/README.md)). Two cold generations in separate
   processes give identical bytes. External-reference oracles (6) use
   tolerances; goldens do not.

## 9. Resource notes

Variables: `r` rank, `m_μ` weight multiplicity, `P_w` coupling pairs of
weight `w`, `N` outer multiplicity, `deg ≤ r·max m` nonzeros per ladder
column.

- **Irrep basis.** §3.1: closed form, `O(d·r)` nonzero ladder entries per
  generator family, no linear algebra. §3.2: with `M = max_ν m_ν`, per
  weight exact Gram–Schmidt over `C_μ = Σ_j m_{μ+α_j} ≤ r·M` candidates with
  the form evaluated lazily: every candidate's diagonal, and the row to later
  candidates only of the `m_μ` kept ones, at most `C_μ + Σ_l (C_μ − t_l)` form
  evaluations (`t_l` the kept positions) instead of `C_μ(C_μ+1)/2`. Each
  evaluation scans at most `M` coordinates, and projecting a kept row, its
  recipe and its lowering column costs `O(C_μ m_μ)`, so the form and
  Gram–Schmidt phase is `O(M Σ_μ C_μ (1 + m_μ))` rational operations, which
  is `O(r M³)` per weight in the uniform dense bound. No `C_μ × C_μ` matrix is
  formed: the working set is one `C_μ`-slot row per kept candidate, which
  Gram–Schmidt and the lowering matrices read.
- **CGC per channel** is the highest-weight elimination plus the descent:
  - *Elimination* (§4.2–§4.3): exact echelon reduction of at most
    `r·|P_{λ_c}|` constraint rows over `|P_{λ_c}|` columns, dense worst case
    `O(r·|P_{λ_c}|³)` rational operations. It is not linear in the output:
    the SU(2) singlet `j⊗j → 0` has `P = 2j+1` pairs, `P − 1` two-entry
    constraints and `P` output entries, and its reduction performs
    `(P−1)(P−2)/2` row eliminations.
  - *Descent* (§4.4): `O(N · Σ_{m_c} |P_{wt(m_c)}| · r · deg)` rational
    operations.

  Work is per channel and per weight space of `c`; the full product dimension
  `d_a d_b` never appears as a dense dimension.
- **SU(2).** The closed form remains the evaluator for a single coefficient
  (`su2::clebsch_gordan`) and for F and R (an `O(j)` Racah sum against
  `Ω(j²)` for any four-CGC contraction).
- **Retention.** Modules, completed channels and computed dual-pair phases
  are retained only by the process-wide CGC tier of
  [`cache`](user-guide/resources.md). Its configured charged-byte and entry
  limits apply in aggregate across presented roots and groups; after a query,
  retained cache-owned data obey those limits, while in-flight working memory
  and caller-owned values are excluded. Reset and trim cover every retained
  kind. Values are exact, so eviction never changes a returned bit. A single
  channel's working set is not bounded by these limits: `E8 248⊗248` has
  `61504` product states and at most `Σ_μ P(μ)² = 1 787 056` CGC nonzeros
  over its five channels.
  The working set of one channel miss, by stage. A rational whose reduced
  numerator and denominator fit `i64` and `u64` costs its slot alone; any
  other costs its slot, a boxed header and its numerator and denominator
  limb capacities (retained module values are re-allocated at their length,
  so the charge bounds them); `P_w` is the
  number of coupling pairs of weight `w` of `c` and `m_c(w)` its
  multiplicity in `c`.
  1. *Modules* (live throughout): one irrep module per distinct label among
     `a, b, c`, shared by equal labels, with their weight spaces, norms,
     public indices with their within-weight ranks and ladder arrays, one
     raising and one lowering sparse matrix per simple root and weight space. For `SU(n)`
     the ladder arrays alone hold `2(n−1)` matrix headers per weight space,
     already quadratic in `n` for the fundamental.
  2. *Pair storage*: for each weight `w` of `c`, its pair list, `O(P_w)`,
     live from its construction until its last lowered reader. A pair is located by row arithmetic, not a map: for
     a left state `m_a`, the pairs of `P_w` are the whole weight space of
     `w − wt(m_a)` in public order, so `(m_a, m_b)` sits at the row start of
     `m_a` plus the public rank of `m_b` within its weight space. One row-start
     scratch of `d_a` slots per channel (written only for the left states of
     the target in use) and one rank per magnetic index of each module serve
     every target; a pair not of the target weight is rejected by pair
     equality.
  3. The maximum of three successive stages:
     - *elimination*: the sparse raising constraint matrix on `P_{λ_c}`, the
       pivot row in use and the emerging nullspace basis (`N` rational
       vectors of length `P_{λ_c}`);
     - *descent and rounding*: a state at depth `ℓ = Σ_i k_i` is lowered only
       from states at depth `ℓ − 1`, and a recipe may mix several of them,
       so each exact vector is rounded once when it completes and kept
       exactly until its last kept candidate has been lowered. With
       `D_ℓ = Σ_{depth w = ℓ} m_c(w) P_w`, the live exact slots (zeros
       included) are at most `N (max_ℓ (D_{ℓ−1} + D_ℓ) + max_w kept_w P_w)`:
       two depth layers plus one weight's lowered candidates (`kept_w` the
       candidates kept at `w`), instead of the whole channel
       `N Σ_w m_c(w) P_w`. The rounded entries are written per column
       `(μ, m_c)` as they are formed and concatenated in §4.7 order at the
       end, `O(nnz + N d_c)` with no sort; that final copy briefly holds the
       rounded entries twice;
     - *verification*, after the rational channel is released: the rounded
       entries, read in place through one `(start, end)` run per column
       `(m_c, μ)` (`N d_c` pairs; a column split into two runs is a support
       failure), plus one weight block at a time: its `nnz_w` entries
       grouped by pair row and its `(N m_c(w))²` Gram matrix, so
       `O(N d_c + max_w (nnz_w + N² m_c(w)²))` beyond the entries instead of
       a second copy of the channel; then the residual map of one
       first-descent column at a time.
- **F and R requests.** The channel misses of one F or R request (§6) share
  their modules by label: each distinct label is loaded once per request
  instead of once per missed channel, and a channel served from the tier
  loads none. A module is released after the last channel of the request
  whose labels contain it, in the channel order of §6. R's two channels need
  the same labels, so the module working set is that of one channel; F's
  channels `(a,b,e)`, `(e,c,d)`, `(b,c,f)`, `(a,f,d)` release `e` after the
  second and `b, c` after the third, so with six distinct labels at most
  five modules are live (not all six). No module is held during the
  contraction.
- **No catalog.** There is no generator catalog, catalog budget or catalog
  error for any simple type.

**Out of scope:** the public budgets of the cache tiers
([caches](user-guide/resources.md)); floating-point or modular fast paths; a
proof of the `σ = +1` conjecture (§5.3).
