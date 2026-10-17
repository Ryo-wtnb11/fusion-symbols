#!/usr/bin/env python3
"""Offline generator for the `tables` catalog (fusion-symbols).

Authority: AnyonWikiDatabase (https://github.com/anyonwiki/AnyonWikiDatabase,
MIT) at commit badbfe299dd533c12a84e817f5890059705e0a3c. Nothing in the crate
runs this script; it needs Python 3 with mpmath (recorded run: Python 3.14.6,
mpmath 1.3.0) and a checkout of the database at the pinned commit (only the
two files below are read; both are verified by SHA-256).

    python3 tools/anyonwiki/gen_category_tables.py \\
        --database <AnyonWikiDatabase checkout>

It classifies all 989 entries, and for every supported entry evaluates each
exact symbol (qqb_id: minimal polynomial + root
index) at 80 digits, rounds the real and imaginary parts correctly to
binary64 (confirmed unchanged at 160 digits), and writes, relative to the
crate root:

  data/anyonwiki-badbfe2/N/<ring code>.txt    a b c Nabc
  data/anyonwiki-badbfe2/F/<code>.txt         a b c d α e β μ f ν Re Im
  data/anyonwiki-badbfe2/R/<code>.txt         a b c α μ Re Im   (braided only)
  src/tables/catalog.rs                       shipped entries + status index
  tests/fixtures/anyonwiki-badbfe2-check.txt  every exact symbol per entry
                                              of rank <= DEFAULT_MAX_RANK
  tests/fixtures/anyonwiki-badbfe2-unitary-check.txt
      above that rank: exact dimensions and twists per entry, and every
      distinct exact F/R symbol once with its rounded value

Entries above DEFAULT_MAX_RANK are compiled only with the crate feature
`anyonwiki-unitary`: their catalog entries carry that `cfg`, and their status
row reads `GATED` (`NotShipped` without the feature), so the default build
compiles the rank <= DEFAULT_MAX_RANK data only.

N and F files are shared by content: an F file is named after the lowest code
that uses it. AnyonWiki keys map onto the layouts as F "a_b_c_d_e_α_β_f_γ_δ"
-> "a b c d α e β γ f δ" and R "a_b_c_α_β" -> "a b c α β". Decimals are
Python's shortest round-trip repr, which Rust's correctly rounded
`str::parse::<f64>` reads back bit for bit.

Supported = `is_unitary` and every signed quantum dimension equals
|1 / F^{a ā a}_a[1, 1]|: exactly the entries whose dimension, Frobenius-Schur
phase and twist the TensorKitSectors derivations (which the crate uses)
reproduce. That is checked at 80 digits for every unitary entry.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import sys

COMMIT = "badbfe299dd533c12a84e817f5890059705e0a3c"
CATEGORIES = "AlgebraicStructures/FusionCategories/MultFreeFusionCategories/Rank1-7.json"
RINGS = "AlgebraicStructures/FusionRings/mult1.json"
SHA256 = {
    CATEGORIES: "383cb2db58ae839f73c3d67ad79186e6ee44d69c32fd8fb202d1a4089ca7e859",
    RINGS: "66117d72d204a5359ef43e584ed46cfe503240558aa8b9c381922b4861aa1153",
}
DATA = "data/anyonwiki-badbfe2"
CATALOG = "src/tables/catalog.rs"
CHECK = "tests/fixtures/anyonwiki-badbfe2-check.txt"
UNITARY_CHECK = "tests/fixtures/anyonwiki-badbfe2-unitary-check.txt"
# Rank cut of the default build; every supported entry above it is compiled
# only with FEATURE.
DEFAULT_MAX_RANK = 4
FEATURE = "anyonwiki-unitary"
# Working precision (decimal digits) and the higher precision used to confirm
# that the rounding to binary64 is stable.
DPS, CONFIRM_DPS = 80, 160
# Known upstream defect: flagged unitary, but with negative quantum dimensions
# (survey §1.1). Any other unitary entry failing the dimension check aborts.
KNOWN_PIVOTAL_ANOMALY = [7, 1, 2, 11, 2]

PROVENANCE = {
    "source_repository": "https://github.com/anyonwiki/AnyonWikiDatabase",
    "source_commit": COMMIT,
    "source_license": "MIT, Copyright (c) 2026 AnyonWiki",
    "source_files": "; ".join(f"{p} sha256={d}" for p, d in SHA256.items()),
    "projection": "tools/anyonwiki/gen_category_tables.py: exact qqb_id evaluated at 80 digits (confirmed at 160), correctly rounded to binary64, written as shortest round-trip decimals",
    "object_order": "fusion-ring element order, one-based: 1 = unit",
    "gauge": "source gauge verbatim; only the one-based to zero-based label shift is applied",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qqb_value(qqb, dps):
    """The exact algebraic number named by a qqb_id, to `dps` digits."""
    import mpmath

    coeffs, index = qqb.split("__")
    a = [int(c) for c in coeffs.split("_")]
    index = int(index)
    with mpmath.workdps(dps):
        if len(a) == 2:
            roots = [mpmath.mpf(-a[0]) / a[1]]
        else:
            roots = mpmath.polyroots(a[::-1], maxsteps=500, extraprec=4 * dps)
        tiny = mpmath.mpf(10) ** (-dps // 2)
        real = sorted(mpmath.re(r) for r in roots if abs(mpmath.im(r)) < tiny)
        cplx = sorted(
            (mpmath.mpc(r) for r in roots if abs(mpmath.im(r)) >= tiny),
            key=lambda z: (z.real, z.imag),
        )
        if len(real) + len(cplx) != len(a) - 1:
            sys.exit(f"{qqb}: root classification failed")
        ordered = [mpmath.mpc(r, 0) for r in real] + cplx
        z = ordered[index - 1]
        im = mpmath.mpf(0) if index <= len(real) else z.imag
        # Defensive: the value must be a root.
        p = sum(mpmath.mpf(c) * z**k for k, c in enumerate(a))
        if abs(p) > mpmath.mpf(10) ** (-dps // 2):
            sys.exit(f"{qqb}: not a root (residual {p})")
        return mpmath.mpc(z.real, im)


_F64 = {}


def to_f64(qqb):
    """Correctly rounded (re, im), confirmed stable at a higher precision."""
    if qqb not in _F64:
        z, z2 = qqb_value(qqb, DPS), qqb_value(qqb, CONFIRM_DPS)
        pair = (float(z.real), float(z.imag))
        if pair != (float(z2.real), float(z2.imag)):
            sys.exit(f"{qqb}: rounding not stable between {DPS} and {CONFIRM_DPS} digits")
        _F64[qqb] = pair
    return _F64[qqb]


def twist_f64(qqb):
    """Correctly rounded e^{2πiθ} for the twist fraction θ named by a qqb_id."""
    import mpmath

    def at(dps):
        theta = qqb_value(qqb, dps).real
        with mpmath.workdps(dps):
            z = mpmath.expjpi(2 * theta)
            return float(z.real), float(z.imag)

    pair = at(DPS)
    if pair != at(CONFIRM_DPS):
        sys.exit(f"{qqb}: twist rounding not stable between {DPS} and {CONFIRM_DPS} digits")
    return pair


def ulps(a, b):
    import struct

    def key(x):
        bits = struct.unpack("<q", struct.pack("<d", x))[0]
        return -(bits & 0x7FFF_FFFF_FFFF_FFFF) if bits < 0 else bits

    return abs(key(a) - key(b))


def unit_f_keys(entry, mult):
    """AnyonWiki key of F^{a ā a}_a[1, 1] for every object a (one-based)."""
    rank = len(mult)
    keys = []
    for a in range(rank):
        duals = [b for b in range(rank) if mult[a][b][0] == 1]
        if len(duals) != 1:
            sys.exit(f"{entry['uuid']}: object {a + 1} has no unique dual")
        abar = duals[0]
        keys.append(f"{a + 1}_{abar + 1}_{a + 1}_{a + 1}_1_1_1_1_1_1")
    return keys


def dims_match_exact(entry, mult):
    import mpmath

    for key, qd in zip(unit_f_keys(entry, mult), entry["quantum_dimensions"]):
        with mpmath.workdps(DPS):
            d = qqb_value(qd, DPS)
            f = qqb_value(entry["f_symbols"][key], DPS)
            if abs(d - 1 / abs(f)) > mpmath.mpf(10) ** (-DPS // 2):
                return False
    return True


def classify(entry, mult):
    """None for a supported entry, otherwise the Rust `UnsupportedReason`."""
    code = entry["anyonwiki_code"]
    if entry["base_field"] != "QQBar":
        sys.exit(f"{code}: only qqb_id (QQBar) entries are supported")
    if code[1] != 1:
        sys.exit(f"{code}: multiplicity {code[1]} is not supported")
    if not entry["is_unitary"]:
        return "NonUnitary"
    if not entry["in_unitary_gauge"]:
        sys.exit(f"{code}: unitary but not stored in a unitary gauge")
    if not dims_match_exact(entry, mult):
        if code[:5] != KNOWN_PIVOTAL_ANOMALY:
            sys.exit(f"{code}: unitary, but its dimensions are not |1/F|")
        return "PivotalNotUnitary"
    return None


def records(entry, mult):
    rank = len(mult)
    n_lines = [
        f"{a + 1} {b + 1} {c + 1} {mult[a][b][c]}"
        for a in range(rank) for b in range(rank) for c in range(rank)
        if mult[a][b][c] != 0
    ]
    f_lines, r_lines, check, values = [], [], [], {}
    for tag, symbols, numeric, lines in (
        ("F", entry["f_symbols"], entry["numeric_f_symbols"], f_lines),
        ("R", entry["r_symbols"], entry["numeric_r_symbols"], r_lines),
    ):
        for key in sorted(symbols):
            parts = key.split("_")
            if tag == "F":
                a, b, c, d, e, al, be, f, ga, de = parts
                labels = f"{a} {b} {c} {d} {al} {e} {be} {ga} {f} {de}"
            else:
                labels = " ".join(parts)
            qqb = symbols[key]
            re, im = to_f64(qqb)
            nre, nim = numeric[key]
            if ulps(re, nre) > 4 or ulps(im, nim) > 4:
                sys.exit(f"{entry['uuid']} {tag} {key}: {re!r},{im!r} is >4 ulp from numeric {nre!r},{nim!r}")
            lines.append(f"{labels} {re!r} {im!r}")
            values[qqb] = (re, im)
            check.append(f"{tag} {labels} {qqb} {re!r} {im!r} {nre!r} {nim!r}")
    for a, qqb in enumerate(entry["quantum_dimensions"], start=1):
        re, _ = to_f64(qqb)
        check.append(f"D {a} {qqb} {re!r}")
    for a, qqb in enumerate(entry["twists"] or [], start=1):
        re, im = twist_f64(qqb)
        check.append(f"T {a} {qqb} {re!r} {im!r}")
    text = lambda lines: "\n".join(lines) + "\n" if lines else ""
    return text(n_lines), text(f_lines), text(r_lines), check, values


def code_name(code):
    return "_".join(map(str, code))


def rust_str(s):
    return json.dumps(s, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", required=True)
    parser.add_argument("--crate", default=str(pathlib.Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    db, crate = pathlib.Path(args.database), pathlib.Path(args.crate)
    for rel, digest in SHA256.items():
        got = sha256(db / rel)
        if got != digest:
            sys.exit(f"{rel}: SHA-256 {got} != pinned {digest}")
    cats = json.load(open(db / CATEGORIES))["data"]
    rings = json.load(open(db / RINGS))["data"]

    entries = sorted(cats.values(), key=lambda e: e["anyonwiki_code"])
    index, shipped = [], []
    for entry in entries:
        mult = rings[entry["fusion_ring"]]["mult_tab"]
        reason = classify(entry, mult)
        index.append((entry, reason))
        if reason is None:
            shipped.append(entry)

    data = crate / DATA
    for sub in ("N", "F", "R"):
        (data / sub).mkdir(parents=True, exist_ok=True)
        for old in (data / sub).glob("*.txt"):
            old.unlink()
    f_files, n_files, rows, catalog = {}, {}, [], []
    gated_rows, gated_values = [], {}
    for entry in shipped:
        code = entry["anyonwiki_code"]
        ring = rings[entry["fusion_ring"]]
        n_text, f_text, r_text, check, values = records(entry, ring["mult_tab"])
        gated = code[0] > DEFAULT_MAX_RANK
        n_name = n_files.setdefault(n_text, f"N/{code_name(ring['anyonwiki_code'])}.txt")
        f_name = f_files.setdefault(f_text, f"F/{code_name(code)}.txt")
        r_name = f"R/{code_name(code)}.txt" if r_text else None
        for name, text in ((n_name, n_text), (f_name, f_text), (r_name, r_text)):
            if name:
                (data / name).write_text(text)
        braided = bool(r_text)
        if braided != bool(entry["is_braided"]):
            sys.exit(f"{code}: is_braided disagrees with r_symbols")
        block = [f"C {entry['uuid']} {','.join(map(str, code))}"]
        if gated:
            gated_rows.extend(block + [r for r in check if r[0] in "DT"])
            gated_values.update(values)
        else:
            rows.extend(block + check)
        symbols = "f_symbols, r_symbols" if braided else "f_symbols"
        provenance = dict(PROVENANCE)
        provenance["source_entry"] = (
            f"category {entry['uuid']} ({symbols}); fusion ring {ring['uuid']} (mult_tab)"
        )
        provenance["category_key"] = f"anyonwiki_code [{','.join(map(str, code))}]"
        r_field = f'Some(include_str!("../../{DATA}/{r_name}"))' if r_name else "None"
        catalog.append(
            (f'    #[cfg(feature = "{FEATURE}")]\n' if gated else "")
            + "    CategoryEntry {\n"
            f"        uuid: {rust_str(entry['uuid'])},\n"
            f"        code: {code},\n"
            f'        n: include_str!("../../{DATA}/{n_name}"),\n'
            f'        f: include_str!("../../{DATA}/{f_name}"),\n'
            f"        r: {r_field},\n"
            "        provenance: TableProvenance {\n"
            + "".join(
                f"            {k}: {rust_str(provenance[k])},\n"
                for k in (
                    "source_repository", "source_commit", "source_license",
                    "source_files", "source_entry", "projection", "category_key",
                    "object_order", "gauge",
                )
            )
            + "            projection_epoch: PROJECTION_EPOCH,\n"
            "        },\n"
            "    },\n"
        )

    import mpmath

    tool = f"tools/anyonwiki/gen_category_tables.py (Python {sys.version.split()[0]}, mpmath {mpmath.__version__}, {DPS}/{CONFIRM_DPS} digits, default rank cut {DEFAULT_MAX_RANK})"
    counts = collections.Counter(
        reason or ("Gated" if e["anyonwiki_code"][0] > DEFAULT_MAX_RANK else "Shipped")
        for e, reason in index
    )
    out = [
        f"// @generated by {tool}.",
        f"// AnyonWikiDatabase {COMMIT}. Do not edit: rerun the generator.",
        f"// Status counts: {dict(sorted(counts.items()))}.",
        "",
        "use super::provenance::{TableProvenance, PROJECTION_EPOCH};",
        "use super::{CategoryEntry, UnsupportedReason};",
        "",
        f"/// Every supported entry; those above rank {DEFAULT_MAX_RANK} only with the",
        f"/// `{FEATURE}` feature.",
        "pub(super) static SHIPPED: &[CategoryEntry] = &[",
        *catalog,
        "];",
        "",
        f"/// Supported entries above rank {DEFAULT_MAX_RANK}: shipped only with the",
        f"/// `{FEATURE}` feature.",
        "const GATED: Option<UnsupportedReason> = if cfg!(feature = \"" + FEATURE + "\") {",
        "    None",
        "} else {",
        "    Some(UnsupportedReason::NotShipped)",
        "};",
        "",
        "/// Every entry of the pinned database: uuid, code, and `None` when",
        "/// shipped, otherwise why not.",
        "pub(super) static INDEX: &[(&str, [u8; 7], Option<UnsupportedReason>)] = &[",
        *(
            f"    ({rust_str(e['uuid'])}, {e['anyonwiki_code']}, "
            + (f"Some(UnsupportedReason::{r})" if r
               else "GATED" if e["anyonwiki_code"][0] > DEFAULT_MAX_RANK else "None")
            + "),"
            for e, r in index
        ),
        "];",
        "",
    ]
    (crate / CATALOG).write_text("\n".join(out))
    header = [
        f"# AnyonWikiDatabase {COMMIT}; generated by {tool}",
        "# C uuid code                       -- one block per shipped entry",
        "# F/R labels qqb_id shipped_re shipped_im anyonwiki_numeric_re anyonwiki_numeric_im",
        "# D object qqb_id(dim) re          -- correctly rounded quantum dimension",
        "# T object qqb_id(theta) re im     -- correctly rounded exp(2 pi i theta)",
    ]
    (crate / CHECK).write_text("\n".join(header + rows) + "\n")
    unitary_header = [
        f"# AnyonWikiDatabase {COMMIT}; generated by {tool}",
        f"# Entries above rank {DEFAULT_MAX_RANK} (feature `{FEATURE}`).",
        "# V qqb_id re im                   -- every distinct exact F/R symbol, correctly rounded",
        "# C uuid code                       -- one block per entry",
        "# D object qqb_id(dim) re          -- correctly rounded quantum dimension",
        "# T object qqb_id(theta) re im     -- correctly rounded exp(2 pi i theta)",
    ]
    v_rows = [f"V {q} {re!r} {im!r}" for q, (re, im) in sorted(gated_values.items())]
    (crate / UNITARY_CHECK).write_text("\n".join(unitary_header + v_rows + gated_rows) + "\n")
    print(f"{dict(sorted(counts.items()))}; {len(n_files)} N, {len(f_files)} F, "
          f"{sum(1 for e in shipped if e['r_symbols'])} R files")


if __name__ == "__main__":
    main()
