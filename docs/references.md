# Algorithm references

Local paths below include private engines, not public modules. Public Lie
operations use `lie::LieGroup` through `category::{FusionCategory,GroupRepresentations}`;
exact SU(2) evaluators are public under `lie::su2`. Reference-column paths
refer to the external project, not this checkout. Closed-form and table
provenance is in [category conventions](references_categories.md).

## Implementation map

| Local authority (`file::symbol`, paths relative to `src/`) | Pinned reference | Contract or intentional difference |
|---|---|---|
| `lie/su2.rs::{wigner_3j,wigner_6j,clebsch_gordan}`; `exact.rs::SignedSqrtRational`; `primefactor.rs` | [WignerSymbols.jl 2.0.0](https://github.com/Jutho/WignerSymbols.jl/tree/v2.0.0), `src/WignerSymbols.jl::{_wigner3j,_wigner6j,compute6jseries}` and `src/primefactorization.jl` | Exact Racah sums; one final rounding |
| `lie/su2.rs::{canonical_regge_3j,canonical_regge_6j}`; `cache.rs::CoefficientCache` | WignerSymbols.jl `reorder3j`, per-kind Wigner caches | Complete deterministic tie-breaking; byte-weighted sharded Clock-PRO rather than LRU |
| `lie/su2.rs::{f_symbol_exact,r_symbol,fs_phase}` | TensorKitSectors `src/irreps/su2irrep.jl::{Fsymbol,Rsymbol}` | Phase, normalization and argument order fixed in [gauge §1](gauge.md#1-su2-closed-forms) |
| `lie/general/roots.rs::CartanData::gt_product`; `lie/gt.rs::gt_enumerate`; `lie/general/gt.rs::module` | [SUNRepresentations.jl 0.4.0](https://github.com/QuantumKitHub/SUNRepresentations.jl/tree/v0.4.0), `src/gtpatterns.jl::{GTPatternIterator,creation,directproduct}`; Molev, arXiv:math/0211289, Theorem 2.3 and Proposition 2.4 | Exact structure and index-for-index GT basis order; ladder entries equal `creation` in square and sign |
| `lie/general/roots.rs::CartanData`; `lie/general/roots/classical.rs` | Fulton–Harris; Humphreys; QSpace `Source/clebsch_aux.cc::{wdim_B,wdim_C,wdim_D,findMaxWeight}` | Exact Weyl/Freudenthal/Brauer–Klimyk structure; low-rank redirects |
| `lie/general/module.rs::IrrepModule`; `lie/general/cgc.rs::{Ladders,Channel,GeneralFamily}` | SUNRepresentations.jl `src/clebschgordan.jl::{highest_weight_CGC,lower_weight_CGC!}` (the highest-weight nullspace and descent structure) | One exact pipeline for every simple type ([Cartan engine](gauge_cartan.md)): rational nullspace with a reduced-echelon multiplicity basis instead of `qrpos!`/`cref!`, intertwiner descent instead of least squares, one rounding |

Every simple type uses the Cartan engine: an exact highest-weight nullspace in
the a-slow coupling-pair order and an intertwiner descent through the irrep
basis, the Gelfand–Tsetlin basis for `A_r` and a contravariant Gram–Schmidt
basis otherwise. The library-specific choices are in the gauge
specifications; Alex et al. and Weichselbaum below describe the numerical
algorithms that preceded it.

## F/R contraction and gates

`src/frcore.rs::{f_block_raw,r_block_raw}` implements the contraction wiring
of every Cartan-engine F and R block, and `frcore::{check_f_block,
check_r_block}` the block gates. The references are
SUNRepresentations.jl 0.4.0 `src/sector.jl::{_Fsymbol,_Rsymbol}` and
TensorKitSectors `src/sectors.jl::Fsymbol_from_fusiontensor`.
Public blocks live in `src/category.rs::{FBlock,RBlock}`; axis order is in
[the API guide](user-guide/README.md#symbol-layout). Generated F blocks use no
SU(2)-style Regge canonicalization.

`src/category/residual.rs::{f_unitarity_residual,pentagon_residual,hexagon_residual}`
provides explicit consistency checks. Pentagon/hexagon wiring follows
TensorKitSectors `src/sectors.jl::{pentagon_equation,hexagon_equation}`.
Callers choose tolerances; these checks are separate from internal generation
gates and signed gauge oracles.

## Global form — fixed conventions

`src/lie/group.rs::GroupId::admits` decides which dominant weights are representations of which
connected compact group (central character trivial on the quotiented subgroup).
Two entries below are *choices*, not facts; they are pinned here so they are
not re-litigated.

| Convention | What is pinned | Reference | Deviation from the reference |
|---|---|---|---|
| `D_r`, `r` odd: the `Z4` class | Generator `c = [ω_r]`, so `κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ (mod 4)`, uniformly in `r`. Chosen to agree with `FusionCategory::dual` (`λ_r ↦ −λ_r`), so conjugation *is* negation of the class. | Slansky §5 p. 37 (general `D_n` class); Table 41 note (SO(10): adjoint `45 ↦ 0`, spinor `16 ↦ 1`, conjugate `16bar ↦ −1`, vector `10 ↦ 2`)  | Slansky's uniform formula equals `+κ` for `r ≡ 1 (mod 4)` and `−κ` for `r ≡ 3 (mod 4)`. fusion-symbols pins `φ(ω_r) = +1` for all odd `r`. This is the `Z4` automorphism `x ↦ −x`, i.e. a generator choice, not a discrepancy; Slansky's SO(10) note is the `r ≡ 1` instance of ours. |
| `D_r`, `r` even: the two half-spin forms | Named by the class they **retain**: `half_spin_plus` retains `[ω_r]` (admissible iff `a_{r-1} + t ≡ 0`), `half_spin_minus` retains `[ω_{r-1}]` (iff `a_r + t ≡ 0`), where `t ≡ Σ_{i odd ≤ r-2} aᵢ`. | Slansky Table 36 note (SO(8))  | Slansky's `Ss` / `Sc` are accepted as documented **aliases**: `Ss(2r) = half_spin_plus` (`ω_r ↦ s`), `Sc(2r) = half_spin_minus` (`ω_{r-1} ↦ c`). A name based on the central element *killed* is not `r`-uniform: `Ann(z_{ω_r}) = {0, s₊}` for `r ≡ 0 (mod 4)` but `{0, s₋}` for `r ≡ 2 (mod 4)`, since `(ω_r, ω_r) = r/4`. Slansky himself calls the `D₄` labelling arbitrary. Only the `SO(2r)` row is `r`-safe, because `(ε₁,ε₁) = 1` for every `r`. |

## Verification oracles

[tools/README.md](../tools/README.md#test-oracles-binding) defines oracle independence.
Representative sources in the tests are:

| Family | Local evidence |
|---|---|
| SU(2) | `tests/oracle.rs`, `tests/fixtures_large.rs`, `tests/fr_oracle.rs`: closed forms, orthogonality, TensorKitSectors F/R |
| SU(N) | `tests/sun_oracle.rs`, `tests/sun_cgc_gauge.rs`, `tests/sun_fr_su3_values.rs`, `tests/gauge_golden.rs::sun_oracle`: independent GT enumeration and signed SUNRepresentations values; GroupMath product rows |
| Every simple type | `src/lie/general/{cgc,module,gt}/tests.rs`: exact isometry, intertwining and generator relations, Molev formulas; `tests/bcd_fr.rs`, `tests/lie_exceptional.rs`: category identities |
| Gauge drift | `tests/gauge_golden.rs`: bitwise values; `tests/bit_identity.rs`: separate-process bytes |

## Bibliography

Use these sources for derivations; the specifications retain only this library's
conventions and gates.

- A. Alex et al., *A numerical algorithm for the explicit calculation of SU(N)
  and SL(N,C) Clebsch–Gordan coefficients*, JMP 52, 023507 (2011),
  [doi:10.1063/1.3521562](https://doi.org/10.1063/1.3521562),
  [arXiv:1009.0437](https://arxiv.org/abs/1009.0437).
- A. Weichselbaum, *Non-abelian symmetries in tensor networks*, Ann. Phys. 327,
  2972–3047 (2012), [doi:10.1016/j.aop.2012.07.009](https://doi.org/10.1016/j.aop.2012.07.009);
  *QSpace*, SciPost Phys. Codebases 40 (2024),
  [doi:10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40).
- A. I. Molev, *Gelfand–Tsetlin bases for classical Lie algebras*, Handbook of
  Algebra 4 (2006) 109–170, [arXiv:math/0211289](https://arxiv.org/abs/math/0211289),
  Theorem 2.3 and Proposition 2.4: rational Gelfand–Tsetlin matrix elements and
  norms used by [gauge_cartan §3.1](gauge_cartan.md#31-a_r-gelfandtsetlin-basis).
- V. G. Kac, *Infinite-Dimensional Lie Algebras*, 3rd ed., Cambridge (1990),
  Thm. 11.7: positivity of the contravariant form.
- G. Racah, *Theory of Complex Spectra. II*, Phys. Rev. 62, 438 (1942),
  [doi:10.1103/PhysRev.62.438](https://doi.org/10.1103/PhysRev.62.438).
- W. Fulton and J. Harris, *Representation Theory*, Springer (1991), §§18, 24,
  [doi:10.1007/978-1-4612-0979-9](https://doi.org/10.1007/978-1-4612-0979-9).
  J. E. Humphreys, *Introduction to Lie Algebras and Representation Theory*,
  Springer (1972), Freudenthal recursion and Weyl formulas,
  [doi:10.1007/978-1-4612-6398-2](https://doi.org/10.1007/978-1-4612-6398-2).
- R. Slansky, *Group theory for unified model building*, Phys. Rep. 79, 1 (1981),
  §5 and notes to Tables 36/41,
  [doi:10.1016/0370-1573(81)90092-2](https://doi.org/10.1016/0370-1573(81)90092-2).
- P. Etingof et al., *Tensor Categories*, AMS (2015), §§2.1, 8.1,
  [doi:10.1090/surv/205](https://doi.org/10.1090/surv/205).
  A. Kitaev, *Anyons in an exactly solved model and beyond*, Appendix E,
  [doi:10.1016/j.aop.2005.10.005](https://doi.org/10.1016/j.aop.2005.10.005).
- [TensorKitSectors source](https://github.com/QuantumKitHub/TensorKitSectors.jl),
  [TensorKit manual](https://quantumkithub.github.io/TensorKit.jl/stable/):
  category concepts, symbol conventions and consumer-side tensor operations.
