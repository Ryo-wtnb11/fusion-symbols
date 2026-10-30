# Coefficient gauge specification

This document specifies the **gauge** of the coefficients this crate returns:
the deterministic rules that fix the otherwise free basis, sign, and ordering
of every coefficient. The rules live in one place each:

| Coefficients | Specification |
|---|---|
| Simple Lie factors `A_r`–`E8`: irrep bases, CGC, F/R contraction, Frobenius–Schur phases | [`gauge_cartan.md`](gauge_cartan.md) |
| SU(2) closed forms: 3j, 6j, single CGC, F, R, Frobenius–Schur | §1 below |
| Semisimple products, tori and central quotients | §2 below |
| U(1), Z_N, parity, fermion parity, CU(1) closed forms | [category conventions](references_categories.md) |
| AnyonWiki tables | [category conventions](references_categories.md#pinned-tables-table-module) |

## Status: FROZEN NORMATIVE SPECIFICATION

**These documents are the authority. The code is an implementation of them.**

Every discrete choice that fixes values is normative. A conflicting
coefficient is a bug; restore the specified value rather than redefine the
gauge. These rules are complete for value-fixing purposes; implementation
choices not specified are not gauge.

### Operational rule for changing a value

A change to a returned coefficient value, its normalization, or the canonical
convention it is expressed in is not an ordinary change. It is a
**specification correction**, and it requires, in one pull request:

1. **The spec edit.** The changed rule edited in the governing document, with
   the defect it corrects stated: what the old rule got wrong, not merely what
   the new rule says.
2. **A CHANGELOG breaking-change entry** naming the spec correction, so a
   consumer reading only the changelog learns that persisted coefficients are
   stale.
3. **Regenerated golden values** (`tests/gauge_golden.rs`).
4. **A semver-breaking release** carrying the change.

A value change without these four steps is a defect. `tests/gauge_golden.rs`
pins representative signed coefficients.

Prose, cache and implementation changes that preserve specified values are
ordinary changes.

## 1. SU(2) closed forms

The SU(2) closed forms have no free basis to fix: every coefficient is a
closed-form expression in exact big-rational arithmetic, so their gauge is the
set of **phase, normalization, and argument-order conventions** the formulas
are written in. They apply to every `A_1`
presentation (SU(2), SO(3), Sp(2), PSU(2), and each factor of SO(4)).

The SU(2) **CGC blocks** returned by `GroupRepresentations::cgc` come from the
Cartan engine with the `A_1` vertex sign
([`gauge_cartan.md` §4.6](gauge_cartan.md#46-the-a_1-vertex-sign)); they equal
the closed-form `clebsch_gordan` below bit for bit. Their magnetic index is
`m + j`, lowest weight first. F, R and the Frobenius–Schur phase of an `A_1`
factor are the closed forms below.

- **Evaluation model.** Every value is computed
  as a big-rational Racah sum carried as `exact.rs:SignedSqrtRational` (a signed
  rational under a square root) with a **single final rounding** to `f64`,
  to nearest with ties to even over the whole `f64` range including
  subnormal results (`lie/su2.rs` module docs, `docs/gauge_cartan.md` §4.7).
  *Specification correction (#262):* the former rounding was single only for
  exact magnitudes at least `2^-1022`; a smaller value was rounded to 53 bits
  and then again to the subnormal spacing, which can miss the nearest `f64`
  by one subnormal step (for example
  `clebsch_gordan(1035, -1035, 1015, 1015, 2050, -20)`). No intermediate rounding: the dimension factors fold
  into the radicand (`times_sqrt_int`) and phases into the sign. Where the
  presented value is `f64` (`lie/su2.rs::f_symbol`) it is a *presentation* of that
  same exact value, never an independent computation. This is value-fixing: a
  differently-associated float evaluation is a different (worse) value.
- **3j.** `lie/su2.rs:wigner_3j` (`wigner_3j_uncached`): the
  Condon–Shortley phase convention, evaluated by the prime-factorized Racah
  single sum (`primefactor.rs`). Non-admissible labels return exact zero, never
  an error and never a panic.
- **6j.** `lie/su2.rs:wigner_6j` (`wigner_6j_uncached`): the
  Racah single-sum closed form, with the triangle coefficients from
  `lie/su2.rs:delta_sq_pf`.
- **Clebsch–Gordan.** `lie/su2.rs:clebsch_gordan`, composed
  from the 3j as
  $\sqrt{dj_3+1}\,(-1)^{(dj_2-dj_1-dm_3)/2}\, \begin{pmatrix} dj_1 & dj_2 & dj_3 \\ dm_1 & dm_2 & -dm_3\end{pmatrix}$.
  Both the $\sqrt{dj_3+1}$ normalization and the sign are part of the gauge:
  this is its own convention, not inherited from the 3j.
- **F symbol.** `lie/su2.rs:f_symbol_exact` is the value
  authority:
  $F = (-1)^{j_1+j_2+j_3+j_4}\sqrt{(dj_5+1)(dj_6+1)}\; \{dj_1\,dj_2\,dj_5 / dj_3\,dj_4\,dj_6\}$,
  matching TensorKitSectors `su2irrep.jl:Fsymbol`. **The 6j argument order is
  gauge**, not an implementation detail — a permuted argument order is a
  different F convention with the same 6j.
- **R symbol.** `lie/su2.rs::r_symbol`: $(-1)^{j_1+j_2-j_3}$ on
  an admissible triangle, exact zero otherwise (an empty category block) (TensorKitSectors
  `su2irrep.jl:Rsymbol`). The zero on a non-admissible triple mirrors
  `Nsymbol == 0` and is normative: it is what stops a caller multiplying a
  spurious sign into a forbidden channel.
- **Frobenius–Schur.** `lie/su2.rs::fs_phase`:
  $(-1)^{dj}$, i.e. `+1` for integer `j` and `-1` for half-integer. It is a distinct formula, not the R convention.

**Deliberately not gauge** (so changing it is not a specification correction):

- The **doubled-spin label encoding** `dj = 2j`. It is input addressing / API
  shape; changing it would be an API break that leaves every value unchanged.
- The **canonical Regge keys** (`lie/su2.rs:canonical_regge_3j`,
  `canonical_regge_6j`, `canonicalize3j`). These canonicalize a symbol onto a
  representative of its symmetry class *for caching*, carrying the compensating
  phase so the returned value is identical either way. They are value-neutral by
  construction, and pinned by the symmetry tests in `tests/properties.rs`.

## 2. Semisimple products and central quotients

Composite `LieGroup` coefficients use the ordered simple-cover gauges through
`Product<Vec<LieGroup>>`. A D₂ block expands into α₁ then α₂, each in the
`A_1` gauge. Each F/R/A/B multiplicity axis and each CGC magnetic or
multiplicity axis has the last factor fastest. Factor coefficients multiply
left to right. These conventions introduce no change to simple-factor values.
A character lattice `Q ⊆ X ⊆ P` only restricts admitted objects; all fusion
outcomes remain admitted because `ν−λ−μ ∈ Q`. It neither changes coefficients
nor renormalizes them.
