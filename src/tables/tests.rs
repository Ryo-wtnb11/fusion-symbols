use super::provenance::fnv1a64;
use super::*;

#[test]
fn index_and_shipped_agree() {
    // What: the status index lists every database entry once, and its
    // supported rows are exactly the shipped entries, under the same codes.
    assert_eq!(catalog::INDEX.len(), 989);
    let mut uuids: Vec<_> = catalog::INDEX.iter().map(|(uuid, _, _)| *uuid).collect();
    let mut codes: Vec<_> = catalog::INDEX.iter().map(|(_, code, _)| *code).collect();
    uuids.sort_unstable();
    uuids.dedup();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!((uuids.len(), codes.len()), (989, 989));
    let supported: Vec<_> = catalog::INDEX
        .iter()
        .filter(|(_, _, reason)| reason.is_none())
        .map(|(uuid, code, _)| (*uuid, *code))
        .collect();
    let shipped: Vec<_> = catalog::SHIPPED
        .iter()
        .map(|entry| (entry.uuid, entry.code))
        .collect();
    assert_eq!(supported, shipped);
    for entry in catalog::SHIPPED {
        assert!(entry.code[0] <= 4, "{:?}", entry.code);
        assert!(entry.provenance.source_entry.contains(entry.uuid));
        assert_eq!(entry.provenance.projection_epoch, PROJECTION_EPOCH);
    }
}

#[test]
fn every_shipped_entry_constructs() {
    // The invariant behind the infallible named constructors.
    assert_eq!(TableCategory::shipped_uuids().len(), 74);
    let mut braided = 0;
    for uuid in TableCategory::shipped_uuids() {
        let category = TableCategory::by_uuid(uuid).expect("shipped entry is valid");
        assert_eq!(category.uuid(), uuid);
        assert_eq!(category.rank(), usize::from(category.code()[0]));
        assert_eq!(
            TableCategory::by_code(category.code())
                .expect("valid")
                .uuid(),
            uuid
        );
        // Braiding index 0 is AnyonWiki's "unbraided".
        assert_eq!(category.is_braided(), category.code()[5] != 0);
        braided += usize::from(category.is_braided());
    }
    assert_eq!(braided, 58);
    let fib = TableCategory::fibonacci();
    assert_eq!(
        (fib.uuid(), fib.code()),
        (FIBONACCI_UUID, [2, 1, 0, 2, 1, 2, 1])
    );
    let ising = TableCategory::ising();
    assert_eq!(
        (ising.uuid(), ising.code()),
        (ISING_UUID, [3, 1, 0, 1, 1, 1, 1])
    );
}

#[test]
fn unknown_and_unsupported_entries_are_typed_errors() {
    for (result, expected) in [
        (
            TableCategory::by_uuid("not-a-uuid"),
            TableError::UnknownCategory,
        ),
        (
            TableCategory::by_code([2, 1, 0, 2, 1, 3, 1]),
            TableError::UnknownCategory,
        ),
        // Fibonacci's Yang–Lee Galois conjugate: not unitary.
        (
            TableCategory::by_code([2, 1, 0, 2, 2, 1, 1]),
            TableError::Unsupported {
                reason: UnsupportedReason::NonUnitary,
            },
        ),
        (
            TableCategory::by_code([7, 1, 2, 11, 2, 1, 1]),
            TableError::Unsupported {
                reason: UnsupportedReason::PivotalNotUnitary,
            },
        ),
        (
            TableCategory::by_code([5, 1, 0, 1, 1, 1, 1]),
            TableError::Unsupported {
                reason: UnsupportedReason::NotShipped,
            },
        ),
    ] {
        assert_eq!(result.map(|c| c.uuid()), Err(expected));
    }
}

// Golden FNV-1a digests of every shipped N/F/R text (the empty string for
// a fusion-only R) and of each composed fingerprint. This is the only test
// that can fail on a *tamper*: every other test reads the shipped bytes and
// would agree with an edited table whose values still round to the same
// `f64`. Updating a row means the pinned source changed, and
// `docs/references_categories.md` must change with it. Fibonacci's three
// file digests are those of the earlier `FibonacciTable` (same bytes); its
// fingerprint differs from that provider's `0x52e5_b6dc_1a15_96c5` only
// because the per-entry provenance strings are now uniform.
#[rustfmt::skip]
const GOLDEN: [(&str, u64, u64, u64, u64); 74] = [
    ("37fa4c58-97c6-11f1-a6a6-639d9c4bace6", 0x74e0dd86e10d8747, 0xa9b8e69a65c2cc4a, 0xb9efc49feea59817, 0xb563c8857b0fd571),
    ("37fa4d16-97c6-11f1-9c13-c70bad3f9e94", 0x06e7ccd16c14c6e5, 0x0e19b5779865c86e, 0x57ac379192956c49, 0xdc5441b3274a4c1e),
    ("37fa4d34-97c6-11f1-99a1-e3f41b8fcf8f", 0x06e7ccd16c14c6e5, 0x0e19b5779865c86e, 0x507c5266c9665d8e, 0x5184f54dd08535d7),
    ("37fa4e38-97c6-11f1-96af-3f698e299019", 0x06e7ccd16c14c6e5, 0x4a8f513b8e4e86f5, 0x0b6531c4735213e5, 0x6c31a761d3a64e7d),
    ("37fa4e38-97c6-11f1-b8da-8fa994c9232a", 0x06e7ccd16c14c6e5, 0x4a8f513b8e4e86f5, 0x3c84c3c5c8b07d7e, 0xc00bbea4e98f6134),
    ("37fa4e42-97c6-11f1-9e48-17415817148b", 0xaba77aa331769df0, 0x9adee8738a99da25, 0x58206652b75cbd3d, 0xe87c3986c5a8457c),
    ("37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9", 0xaba77aa331769df0, 0x9adee8738a99da25, 0x75d1cc533d6765a1, 0x5271bc87b3a9de78),
    ("37fa4e56-97c6-11f1-907a-eb6ab8e8bc11", 0x19a1b7ec5a16713a, 0x711d5f3f29289500, 0xc7964e2911ea7032, 0x41d48eb7be32fe0e),
    ("37fa4e60-97c6-11f1-ad4e-e132966151ac", 0x19a1b7ec5a16713a, 0x711d5f3f29289500, 0xc90dbf6004fc38d8, 0xf1399835c620cc9c),
    ("37fa4e60-97c6-11f1-9010-93a12ab32138", 0x19a1b7ec5a16713a, 0x711d5f3f29289500, 0xbbc84cebf422ab14, 0xd84a7f26147e14b6),
    ("37fa4e6a-97c6-11f1-8b07-33a13543ef72", 0x19a1b7ec5a16713a, 0x711d5f3f29289500, 0xc96d8a22ecce521a, 0x6086a65b0854eedc),
    ("37fa4e76-97c6-11f1-b235-c1b5d6228630", 0x19a1b7ec5a16713a, 0xfca44ae8d279e1a8, 0xd554fad1068c0258, 0xb179525156f09bb4),
    ("37fa4e76-97c6-11f1-b344-2713cca46212", 0x19a1b7ec5a16713a, 0xfca44ae8d279e1a8, 0xb150655459040b1e, 0x2978f48433a084d3),
    ("37fa4e7e-97c6-11f1-9ae0-3f68119fb6bc", 0x19a1b7ec5a16713a, 0xfca44ae8d279e1a8, 0x59e02b2e01016bd4, 0xf6be0aecb92f87cf),
    ("37fa4e88-97c6-11f1-ac69-3b902638054d", 0x19a1b7ec5a16713a, 0xfca44ae8d279e1a8, 0x1ecc4899c61b03ce, 0x5acecaeff03dc79d),
    ("37fa4e92-97c6-11f1-a160-09d79d636553", 0x667021aa5662bfa6, 0xde844cb69dfe0b71, 0xbbf6e3c27f3be4b7, 0x4a1a2ae6433267e7),
    ("37fa4e92-97c6-11f1-8bf0-535d13ce86a2", 0x667021aa5662bfa6, 0xde844cb69dfe0b71, 0x26b0f4ddca7082db, 0x6e2736bdc3bb8ba9),
    ("37fa4e9c-97c6-11f1-ab5c-3765b017a397", 0x667021aa5662bfa6, 0xde844cb69dfe0b71, 0x4b00448c7312d124, 0x78f02c781a29a458),
    ("37fa4e9c-97c6-11f1-8ce9-19309dfc597c", 0x667021aa5662bfa6, 0xd8262e46910f6a08, 0xcbf29ce484222325, 0x0983c3e06ab3be9a),
    ("37fa4e9c-97c6-11f1-b94f-4f93a9bcdde2", 0x667021aa5662bfa6, 0xcb95b969ff7dc1ec, 0xcbf29ce484222325, 0x29c45fe176e2d866),
    ("37fa4ea8-97c6-11f1-8b25-ad9b74351a3e", 0xb29c4db85be67d5a, 0x17f3cc12cade1b80, 0xe41e29930c25f7a5, 0xd51bdff746694d29),
    ("37fa4ea8-97c6-11f1-a7ef-efe4cbc8aa54", 0xb29c4db85be67d5a, 0x17f3cc12cade1b80, 0x30a3e1366b044ff0, 0x3b4ed064a781ab0b),
    ("37fa4eba-97c6-11f1-8d6a-5dcedc1c3e4c", 0x30dc6ace7a1cb11a, 0x0915bd3f1b4c5ae6, 0xe49d5f21861d110c, 0xe8e9cfed09c3163c),
    ("37fa4eba-97c6-11f1-a0ff-e77a9b67e913", 0x30dc6ace7a1cb11a, 0x0915bd3f1b4c5ae6, 0x1c610364557ca804, 0xe2df321ad9c9fb90),
    ("37fa4ec4-97c6-11f1-bce7-a7ac649058a2", 0x30dc6ace7a1cb11a, 0x0915bd3f1b4c5ae6, 0x01da320c221d2582, 0xdafe4abc753719f4),
    ("37fa4ece-97c6-11f1-9f20-5192b383a942", 0x30dc6ace7a1cb11a, 0x5723e4781e1b274c, 0xcbf29ce484222325, 0x8ef666ea88e8d5d8),
    ("37fa4eda-97c6-11f1-9b2a-61a5a64dcea5", 0x30dc6ace7a1cb11a, 0x8ef59385b389320c, 0xcbf29ce484222325, 0xb4ee5fd47e8ee848),
    ("37fa4eda-97c6-11f1-b133-fb9f653cc598", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0x558d8a3d5452c7f9, 0x64f36a42dd901082),
    ("37fa4eec-97c6-11f1-85d9-4ff09ec508f0", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0x807947718a49e77f, 0x1404488226297ab3),
    ("37fa4ef6-97c6-11f1-aa0b-73297b1c7138", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0x5a2c382d26b11ca9, 0x866f2730f1840e79),
    ("37fa4f00-97c6-11f1-847b-b106f4ef9302", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0xe84bfb25627a4ab9, 0xeeb1889b3aebc3a8),
    ("37fa4f08-97c6-11f1-b2ac-c75753090d40", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0xce8b79b4a8da8691, 0x425ac840c5e80c66),
    ("37fa4f1e-97c6-11f1-85ce-615887999768", 0x82191b671a05aa65, 0xdfb1342e4f9e4135, 0x011ee2a4a3ff693d, 0x1e5cedbe31be7463),
    ("37fa4f3a-97c6-11f1-81ee-1dde08099519", 0x82191b671a05aa65, 0xada1136c117d9be5, 0x0046a474e1165451, 0xf4a5ce1bd5d9f120),
    ("37fa4f50-97c6-11f1-9cc3-cb1a5a83f8c5", 0x82191b671a05aa65, 0xada1136c117d9be5, 0xd3ca3305aa3df477, 0x5b8e04e7094bb8d4),
    ("37fa4f50-97c6-11f1-9b55-7b65e8ab37a1", 0x82191b671a05aa65, 0xada1136c117d9be5, 0xa0d2c1374a5d88a5, 0x1c0f251ae543c719),
    ("37fa4f5a-97c6-11f1-b337-a76e2bfad4f9", 0x82191b671a05aa65, 0xada1136c117d9be5, 0xe8d6c297c8f4c619, 0xcaec7c561d72f22c),
    ("37fa4f64-97c6-11f1-a154-9dcd47d3623e", 0x82191b671a05aa65, 0x205b730a0dec8729, 0xcbf29ce484222325, 0xb8180dec2583c310),
    ("37fa4f6c-97c6-11f1-b76f-8be3c0f07f22", 0x82191b671a05aa65, 0x43a1eabaa9bf1a87, 0xcbf29ce484222325, 0xb11d5529a5c2a8aa),
    ("37fa4f78-97c6-11f1-90cf-c3d624fdb861", 0xb849ce3543a558ad, 0xab5f2193e3d8060c, 0x8ba640581ffe492e, 0x893eda3f601df2a4),
    ("37fa4f82-97c6-11f1-b783-8362fedc1bf3", 0xb849ce3543a558ad, 0xab5f2193e3d8060c, 0x9584272ff0f2377a, 0x4058a1a61fed98ca),
    ("37fa4f8c-97c6-11f1-95a9-6de163756062", 0xb849ce3543a558ad, 0xab5f2193e3d8060c, 0x00ae6ca1ada0bbdf, 0x06f1d3315f27a37b),
    ("37fa4f8c-97c6-11f1-9313-21cc7a151d2e", 0xb849ce3543a558ad, 0xab5f2193e3d8060c, 0xc5f580ade85317a7, 0xa13f409503e4219a),
    ("37fa4f96-97c6-11f1-b0f8-f5286170a3d8", 0xb849ce3543a558ad, 0xe5e9d60102e0103d, 0xb1d6fb5b5b6bd0cd, 0xbc34348b54e9fc4f),
    ("37fa4fa0-97c6-11f1-80b5-09cf754d0b11", 0xb849ce3543a558ad, 0xe5e9d60102e0103d, 0x964bfa0e14a7edad, 0xf213876bf6f832ce),
    ("37fa4faa-97c6-11f1-9dab-d7278e31bde9", 0xb849ce3543a558ad, 0xe5e9d60102e0103d, 0xdfe831ae35cbd4e2, 0x3f84868e3a936037),
    ("37fa4fb4-97c6-11f1-b114-012287e60ca6", 0xb849ce3543a558ad, 0xe5e9d60102e0103d, 0x2ad75dca1e64f43a, 0x36a9ebb1fbea3962),
    ("37fa4ff0-97c6-11f1-8044-5f06728109e8", 0x66aabc0a22a4cdbe, 0xfa4fa32ca66efeb1, 0x550eaa86bfce486b, 0x25848814318f06f4),
    ("37fa4ffa-97c6-11f1-867c-3d31933f8d31", 0x66aabc0a22a4cdbe, 0xfa4fa32ca66efeb1, 0x78435270b7acdaa9, 0xf70749532dc5ecdf),
    ("37fa4ffa-97c6-11f1-a6da-4f8a199e104e", 0x66aabc0a22a4cdbe, 0xfa4fa32ca66efeb1, 0x5d63212272be5b34, 0xb8ec80b6ebb5d842),
    ("37fa5004-97c6-11f1-b0d3-bfeb70fd603f", 0x66aabc0a22a4cdbe, 0xadb253613fa57bae, 0xcbf29ce484222325, 0xc69148576765db8b),
    ("37fa5004-97c6-11f1-9ac3-9109f9a87686", 0x66aabc0a22a4cdbe, 0xf0f55427831d74fc, 0xcbf29ce484222325, 0x9c290fcabf543ed0),
    ("37fa5004-97c6-11f1-ac82-51b65800daec", 0x567a823fdad6a49d, 0xdbad66d31f5e3bb7, 0x1e42d0a684bfddbf, 0x0889c533c7fd796c),
    ("37fa5010-97c6-11f1-aee4-8b83643be453", 0x80b86826eff94b12, 0x35bb9379dfee2fca, 0xd46984fb4d17a21c, 0x106dbc5ccb0294ad),
    ("37fa5018-97c6-11f1-b275-e16d4d02513f", 0x80b86826eff94b12, 0x35bb9379dfee2fca, 0x788a4ff264535505, 0xde432da34a6734e6),
    ("37fa5018-97c6-11f1-942c-29c8943f8a70", 0x80b86826eff94b12, 0x35bb9379dfee2fca, 0x128c1e21d4be3bbe, 0x8957251423f8215d),
    ("37fa504a-97c6-11f1-b013-39138477deb3", 0x2b4579337198805c, 0x406a2b24dcc3c183, 0x0542811defe22b50, 0xddc8284e12340e4e),
    ("37fa504a-97c6-11f1-9a1c-1fd1676ff5b7", 0x2b4579337198805c, 0x406a2b24dcc3c183, 0xc85b7ce846eb5fcc, 0x85f2c21ee072fe3e),
    ("37fa505e-97c6-11f1-bae1-b18e3688ead0", 0x696d7c04011583a5, 0x43258194b1b27e68, 0x9da3c475f81db822, 0xbeed145a0d38102d),
    ("37fa5070-97c6-11f1-8d35-dbac41cd7b27", 0x696d7c04011583a5, 0x43258194b1b27e68, 0xe8dfc16264e1687a, 0x175e800e3fbe1313),
    ("37fa507c-97c6-11f1-95db-0dac8e1a6076", 0x696d7c04011583a5, 0x43258194b1b27e68, 0xa0bd6c13996549be, 0xf962f44cc3f5ed69),
    ("37fa5086-97c6-11f1-a6e9-fd53bad0d671", 0x696d7c04011583a5, 0x43258194b1b27e68, 0x0a865cabb9de8466, 0xd7cb4fedc1447ea2),
    ("37fa509a-97c6-11f1-8556-1d179383acf5", 0x696d7c04011583a5, 0x7cd0a56f34449ed5, 0xa25c353c68bd3595, 0x13dd4c80ecee1559),
    ("37fa50a2-97c6-11f1-9ffd-6bfab4e46a13", 0x696d7c04011583a5, 0x7cd0a56f34449ed5, 0x274df942e85901b5, 0xdd612f2f7cd7d376),
    ("37fa50ae-97c6-11f1-9089-b30e985b6864", 0x696d7c04011583a5, 0x7cd0a56f34449ed5, 0xa741c54faf7f91c5, 0xa870b227d03f170e),
    ("37fa50b8-97c6-11f1-8156-21c77f312f7f", 0x696d7c04011583a5, 0x7cd0a56f34449ed5, 0x7c157cd9605c72fd, 0x69f386b5d16334f6),
    ("37fa50c2-97c6-11f1-8f15-f39cab26dff4", 0x696d7c04011583a5, 0x9e25969333d1ec72, 0xcbf29ce484222325, 0x7a1905054bfa7d8a),
    ("37fa50cc-97c6-11f1-96e9-c5ec31d5f018", 0x696d7c04011583a5, 0x7f6a678a3a23ade9, 0xcbf29ce484222325, 0xfd48d47ac34fa940),
    ("37fa50d4-97c6-11f1-99df-99ff47ca6936", 0xab0ceccc5270ff49, 0x743527c24338fef8, 0xcbf29ce484222325, 0x843117e6375c1a54),
    ("37fa50e0-97c6-11f1-9021-c1f80c8d0a34", 0xab0ceccc5270ff49, 0xfeafafbfbced89a9, 0xcbf29ce484222325, 0x4c7aceb009ef8fb7),
    ("37fa50ea-97c6-11f1-b793-11451dddaf9b", 0xab0ceccc5270ff49, 0x2759a0a22f0d2967, 0xcbf29ce484222325, 0x7112d2230bcefeb3),
    ("37fa50f4-97c6-11f1-888f-0b1c97d9f8ef", 0xab0ceccc5270ff49, 0x8dc3ed10ea173c6e, 0xcbf29ce484222325, 0x283d4b4744961eaa),
    ("37fa50f4-97c6-11f1-b674-0306f7ad8dcb", 0x5efe2ed8946b085d, 0x6f6250b2624fd911, 0xcbf29ce484222325, 0x6007ab83b311f578),
    ("37fa50fe-97c6-11f1-b9de-6dfcb9ac3aea", 0x5efe2ed8946b085d, 0xe89bb89c9df8f69b, 0xcbf29ce484222325, 0x1eb3ef13f1d45243),
];

#[test]
fn shipped_tables_and_fingerprints_are_pinned() {
    assert_eq!(catalog::SHIPPED.len(), GOLDEN.len());
    for (entry, (uuid, n, f, r, fingerprint)) in catalog::SHIPPED.iter().zip(GOLDEN) {
        assert_eq!(entry.uuid, uuid);
        assert_eq!(fnv1a64(entry.n.as_bytes()), n, "{uuid} Nsymbols changed");
        assert_eq!(fnv1a64(entry.f.as_bytes()), f, "{uuid} Fsymbols changed");
        assert_eq!(
            fnv1a64(entry.r.unwrap_or("").as_bytes()),
            r,
            "{uuid} Rsymbols changed"
        );
        let category = TableCategory::by_uuid(uuid).expect("valid");
        assert_eq!(
            fnv1a64(category.authority_fingerprint()),
            fingerprint,
            "{uuid}"
        );
    }
    let mut prehashes: Vec<u64> = GOLDEN.iter().map(|row| row.4).collect();
    prehashes.sort_unstable();
    prehashes.dedup();
    assert_eq!(prehashes.len(), GOLDEN.len(), "fingerprints are distinct");
}

#[test]
fn fingerprint_is_reproducible_and_epoch_bound() {
    assert_eq!(
        TableCategory::fibonacci().authority_fingerprint(),
        TableCategory::fibonacci().authority_fingerprint()
    );
    let provenance = *TableCategory::fibonacci().provenance();
    let mut baseline = Vec::new();
    provenance.write_canonical_bytes(&mut baseline);
    let mut bumped = Vec::new();
    TableProvenance {
        projection_epoch: "other",
        ..provenance
    }
    .write_canonical_bytes(&mut bumped);
    assert_ne!(baseline, bumped);
}

#[test]
fn shipped_fibonacci_braiding_is_tensorkits_not_its_mirror() {
    // The other braiding of the same fusion category (AnyonWiki b = 1)
    // has the conjugate R-symbols; TensorKitSectors' `FibonacciAnyon`
    // has R^{ττ}_𝟙 = e^{+4πi/5}.
    let fib = TableCategory::fibonacci();
    let tau = TableObject(2);
    let r = fib.r_symbol(tau, tau, fib.unit());
    assert!((r.arg() - 0.8 * std::f64::consts::PI).abs() < 1e-15, "{r}");
    let mirror = TableCategory::by_code([2, 1, 0, 2, 1, 1, 1]).expect("shipped");
    assert_eq!(mirror.r_symbol(tau, tau, mirror.unit()), r.conj());
}

#[test]
fn fingerprint_moves_on_a_source_edit_too_small_to_move_a_coefficient() {
    let entry = TableCategory::fibonacci().entry;
    // Appends one trailing zero digit to every decimal literal: value
    // preserving and below binary64 resolution.
    let edited: String = entry
        .f
        .lines()
        .map(|line| {
            let fields: Vec<String> = line
                .split_whitespace()
                .map(|field| {
                    if field.contains('.') {
                        format!("{field}0")
                    } else {
                        field.to_owned()
                    }
                })
                .collect();
            format!("{}\n", fields.join(" "))
        })
        .collect();
    assert_ne!(edited, entry.f, "the edit did not apply");
    let fields = |text: &str| {
        text.split_whitespace()
            .map(|field| field.parse::<f64>().expect("numeric field").to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        fields(&edited),
        fields(entry.f),
        "the edit moved a coefficient"
    );

    let provider = TableCategory::fibonacci();
    let fingerprint = |f| fingerprint_bytes(&entry.provenance, entry.n, f, entry.r);
    assert_eq!(fingerprint(entry.f), provider.authority_fingerprint());
    assert_ne!(fingerprint(&edited), provider.authority_fingerprint());
}
