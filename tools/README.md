# Tools and test-oracle rules

## Data-build tool

`anyonwiki/gen_category_tables.py` builds the shipped `table` catalog:
`data/anyonwiki-badbfe2/**` and `src/table/catalog.rs` (`@generated`). It is
production-data provenance, not a test oracle. It also writes per-symbol audit
files to `tests/fixtures/anyonwiki-badbfe2-*check.txt`; those are gitignored
and no test reads them.

## Test oracles (binding)

- **The expected value is readable from the test**: an exact formula (closed
  form, identity, dense expansion, hand calculation, small exact computation
  written in the test), or a few hardcoded values with a comment naming where
  they were checked (tool and version, e.g. "checked against
  SUNRepresentations.jl 0.4.0").
- **Representative cases, not corpora.** Test what the contract promises; each
  test states the contract it protects.
- **No generator scripts or opaque fixture files**, and no dev-dependency used
  only as an oracle when a formula or a few literals suffice.
- **A literal is commented with a source only after it was re-checked against
  that source.** Values this crate produced itself are a drift detector of the
  frozen gauge (`tests/gauge_golden.rs`) and say so.

## Spec rules (binding)

- **Reference-inventory rule.** Any claim of the form "the existing X" (a
  function, symbol, test, or fixture that a spec/issue/PR builds on) must carry
  a concrete `file::symbol` and be verified to exist before anything is built
  on it. An unverified "existing X" is not evidence; it is a hypothesis.
- **Warm-no-resweep rule.** Any coefficient-*generation* API (SU(N)/SO(N)/Sp
  CGC/F/R sweeps and their caches) must ship a test proving a warm second call
  returns from the cache without re-running the sweep. Code that only
  *consumes* an existing decomposition (through `FusionCategory::fusion`) is exempt.
