//! Compatibility-policy and determinism tests for the exceptional authority
//! fingerprint (`LieGroup::authority_fingerprint`).
//!
//! Mirrors `tests/bcd_fingerprint.rs`, with the exceptional contract: the
//! values are exact until one correctly rounded conversion, so equal bytes mean
//! bit-identical CGC, F, R, A and B values (`docs/gauge_exceptional.md` §8).
#![cfg(feature = "lie-generated")]

use fusion_symbols::category::FusionCategory;
use fusion_symbols::lie::{GroupId, LieGroup, RootSystem};
use fusion_symbols::product::Product;

fn fingerprint() -> Vec<u8> {
    LieGroup::new(GroupId::g2())
        .unwrap()
        .authority_fingerprint()
        .to_vec()
}

/// Compatibility policy: the fingerprint's exact current bytes.
///
/// This literal is written out by hand from `docs/gauge_exceptional.md` §8 —
/// deliberately NOT derived from `fingerprint()`, so that any change to the
/// returned bytes breaks this assertion and forces a review event.
///
/// Updating this literal REQUIRES a specification correction: the fingerprint
/// is the **version of the exceptional gauge specification**, so it moves only
/// when that document is corrected in a way that changes a returned value. The
/// epoch is per-family: this literal moves independently of
/// `tests/su2_fingerprint.rs`, `tests/sun_fingerprint.rs` and
/// `tests/bcd_fingerprint.rs`, which stay unchanged.
///
/// Do not update it to make a test pass. Update it only as one of the four steps
/// of `docs/gauge.md`, "Status": the spec edit, the `epoch=N` bump in
/// `LieGroup::authority_fingerprint`, the CHANGELOG breaking-change entry, and the
/// regenerated golden values — all in the same PR.
///
/// Epoch history:
/// - `epoch=1` — the initial frozen exceptional specification.
#[test]
fn fingerprint_matches_pinned_compatibility_bytes() {
    assert_eq!(
        fingerprint(),
        b"fusion-symbols:lie-exceptional:ref=gauge-exceptional-spec:input=bourbaki-cartan:order=depth-rootlex-asc:basis=greedy-rank-gs-own-positive:kron=a-fast:mult=rcef-pivot-gs:descent=intertwiner-own-basis:sign=own-candidate-and-pivot-positive:round=once-correct:fs=gauge-derived:fr=frcore:twist=one:scalar=f64:epoch=1",
    );
}

#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(fingerprint(), fingerprint());
}

/// A composite with a G2 factor composes the factor fingerprints exactly as
/// the ordered `Product` of its factor groups does.
#[test]
fn composite_fingerprint_composes_factors() {
    let g2_su2_u1 = GroupId::cover(&[RootSystem::G2, RootSystem::A(1)], 1).unwrap();
    let composite = LieGroup::new(g2_su2_u1).unwrap();
    let factors = Product::new(vec![
        LieGroup::new(GroupId::g2()).unwrap(),
        LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
        LieGroup::new(GroupId::torus(1).unwrap()).unwrap(),
    ]);
    assert_eq!(
        composite.authority_fingerprint(),
        factors.authority_fingerprint()
    );
    assert_ne!(composite.authority_fingerprint(), fingerprint().as_slice());
}
