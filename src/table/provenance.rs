//! The frozen source/projection contract that every shipped table carries.
//!
//! See `docs/references_categories.md` for the full provenance tables and their verification
//! record; the types here are the machine-readable form of the same contract.

/// Identifier of the one exact-source -> [`num_complex::Complex64`] conversion
/// convention this crate implements.
///
/// The convention is spelled out in `docs/references_categories.md`; in one
/// line it is "each exact algebraic symbol is rounded correctly to binary64
/// offline (`tools/anyonwiki/gen_category_tables.py`) and written as its
/// shortest round-trip decimal; at load, `str::parse::<f64>()` reads it back
/// bit for bit, pairs as `re + i·im`, and absent records are exact zero".
///
/// It is folded into every provider's authority fingerprint, so a change makes
/// previously-built fingerprints compare unequal: cached recoupling data
/// computed under the old projection must not be reused under the new one.
///
/// Minted for this crate and named for its source: the AnyonWiki tables at the
/// pinned commit, projected to `f64` on 2026-10-16. Values are the
/// CategoryData.jl decimals, parsed.
pub const PROJECTION_EPOCH: &str = "fusion-symbols/tables/anyonwiki-badbfe2/projection/2026-10-16";

/// Immutable lineage of one shipped category table.
///
/// Every field is part of the category's identity. The struct is deliberately
/// a flat record of `&'static str`: the values are frozen at authoring time,
/// there is nothing to configure, and the canonical serialisation
/// ([`TableProvenance::write_canonical_bytes`]) must stay trivially
/// auditable because it feeds `RuleIdentity`.
///
/// Why not a builder, a `Default`, or public construction: a provenance record
/// that a caller can assemble is a provenance record that can lie. Values are
/// authored next to the data they describe and nowhere else.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct TableProvenance {
    /// Repository the exact symbols are taken from (the authority).
    pub source_repository: &'static str,
    /// Commit pinned in that repository.
    pub source_commit: &'static str,
    /// License of the source data.
    pub source_license: &'static str,
    /// Source files read, each with its SHA-256.
    pub source_files: &'static str,
    /// Which entries of those files: category and fusion-ring identifiers.
    pub source_entry: &'static str,
    /// How the exact symbols became the shipped binary64 tables.
    pub projection: &'static str,
    /// Upstream category identifier.
    pub category_key: &'static str,
    /// Frozen object ordering, stated in upstream's one-based labels.
    pub object_order: &'static str,
    /// What was and was not done to the source gauge.
    pub gauge: &'static str,
    /// The projection epoch in force, normally [`PROJECTION_EPOCH`].
    pub projection_epoch: &'static str,
}

impl TableProvenance {
    /// Appends the canonical, order-fixed byte serialisation of this record.
    ///
    /// The format is one `key=value` line per field in declaration order. It
    /// is a byte contract, not a display form: it is hashed and compared, so
    /// reordering fields, renaming a key, or changing a separator changes
    /// every derived authority fingerprint.
    ///
    /// Why not `Debug` or a serde encoding: `Debug` output is explicitly not a
    /// stability guarantee, and a serialisation framework would put the byte
    /// layout of an identity key behind a dependency's version policy.
    pub fn write_canonical_bytes(&self, out: &mut Vec<u8>) {
        let Self {
            source_repository,
            source_commit,
            source_license,
            source_files,
            source_entry,
            projection,
            category_key,
            object_order,
            gauge,
            projection_epoch,
        } = self;
        // Destructured above so that adding a field without extending this
        // list fails to compile: a silently unhashed provenance field would
        // let two different categories share one identity.
        for (key, value) in [
            ("source_repository", source_repository),
            ("source_commit", source_commit),
            ("source_license", source_license),
            ("source_files", source_files),
            ("source_entry", source_entry),
            ("projection", projection),
            ("category_key", category_key),
            ("object_order", object_order),
            ("gauge", gauge),
            ("projection_epoch", projection_epoch),
        ] {
            out.extend_from_slice(key.as_bytes());
            out.push(b'=');
            out.extend_from_slice(value.as_bytes());
            out.push(b'\n');
        }
    }
}

/// FNV-1a over 64 bits — the prehash a consumer's identity carries beside the
/// fingerprint bytes. Used by the tests to pin the composed fingerprint.
///
/// Why not `std::collections::hash_map::DefaultHasher`: its output is
/// explicitly not stable across Rust releases, and the acceptance criterion
/// here is that two independent builds agree. Why not a real cryptographic
/// digest: identity compares the full byte slice on equality, so the
/// prehash only has to be a cheap, deterministic bucket key — a collision
/// costs a memcmp, not a wrong answer.
#[cfg(test)]
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: TableProvenance = TableProvenance {
        source_repository: "source_repository",
        source_commit: "source_commit",
        source_license: "source_license",
        source_files: "source_files",
        source_entry: "source_entry",
        projection: "projection",
        category_key: "category_key",
        object_order: "object_order",
        gauge: "gauge",
        projection_epoch: PROJECTION_EPOCH,
    };

    #[test]
    fn canonical_bytes_cover_every_field_in_declaration_order() {
        let mut bytes = Vec::new();
        SAMPLE.write_canonical_bytes(&mut bytes);
        let text = String::from_utf8(bytes).expect("ascii");
        let keys: Vec<&str> = text
            .lines()
            .map(|line| line.split('=').next().unwrap())
            .collect();
        assert_eq!(
            keys,
            [
                "source_repository",
                "source_commit",
                "source_license",
                "source_files",
                "source_entry",
                "projection",
                "category_key",
                "object_order",
                "gauge",
                "projection_epoch",
            ]
        );
        assert!(text.contains(&format!("projection_epoch={PROJECTION_EPOCH}\n")));
    }

    #[test]
    fn every_provenance_field_change_changes_the_bytes() {
        let baseline = {
            let mut bytes = Vec::new();
            SAMPLE.write_canonical_bytes(&mut bytes);
            bytes
        };
        // Mutation check: flipping each field in turn must move the bytes, so
        // no field is decorative.
        let mutants = [
            TableProvenance {
                source_repository: "x",
                ..SAMPLE
            },
            TableProvenance {
                source_commit: "x",
                ..SAMPLE
            },
            TableProvenance {
                source_license: "x",
                ..SAMPLE
            },
            TableProvenance {
                source_files: "x",
                ..SAMPLE
            },
            TableProvenance {
                source_entry: "x",
                ..SAMPLE
            },
            TableProvenance {
                projection: "x",
                ..SAMPLE
            },
            TableProvenance {
                category_key: "x",
                ..SAMPLE
            },
            TableProvenance {
                object_order: "x",
                ..SAMPLE
            },
            TableProvenance {
                gauge: "x",
                ..SAMPLE
            },
            TableProvenance {
                projection_epoch: "x",
                ..SAMPLE
            },
        ];
        for mutant in mutants {
            let mut bytes = Vec::new();
            mutant.write_canonical_bytes(&mut bytes);
            assert_ne!(bytes, baseline);
            assert_ne!(fnv1a64(&bytes), fnv1a64(&baseline));
        }
    }

    #[test]
    fn fnv1a64_matches_the_published_reference_vectors() {
        // Reference values from the FNV-1a 64-bit specification test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
