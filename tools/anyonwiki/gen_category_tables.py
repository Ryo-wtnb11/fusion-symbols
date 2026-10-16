#!/usr/bin/env python3
"""Offline generator for the `tables` tables (fusion-symbols).

Authority: AnyonWikiDatabase (https://github.com/anyonwiki/AnyonWikiDatabase,
MIT) at commit badbfe299dd533c12a84e817f5890059705e0a3c. Nothing in the crate
runs this script; it needs Python 3 with mpmath (recorded run: Python 3.14.6,
mpmath 1.3.0).

Two steps:

  extract   Read the full database files (verified by SHA-256), keep only the
            requested category entry and its fusion ring, and write a small
            JSON extract that is committed. The 64 MB source is not committed.

              python3 tools/anyonwiki/gen_category_tables.py extract \
                  --database <checkout of AnyonWikiDatabase at the pinned commit> \
                  --code 2,1,0,2,1,2,1 \
                  --out data/anyonwiki-badbfe2/fibonacci.json

  generate  Evaluate every exact symbol (qqb_id: minimal polynomial + root
            index) to high precision, round each real and imaginary part
            correctly to binary64, and write the shipped tables in the
            record layout the crate parses, plus the check fixture.

              python3 tools/anyonwiki/gen_category_tables.py generate \
                  --extract data/anyonwiki-badbfe2/fibonacci.json \
                  --out-dir data/anyonwiki-badbfe2/fibonacci \
                  --check tests/fixtures/anyonwiki-fibonacci-check.txt

Shipped record layouts (one-based labels, multiplicity indices always 1):

  Nabc.txt  a b c Nabc
  F.txt     a b c d α e β μ f ν Re Im
  R.txt     a b c α μ Re Im

AnyonWiki keys map onto them as F "a_b_c_d_e_α_β_f_γ_δ" -> "a b c d α e β γ
f δ" and R "a_b_c_α_β" -> "a b c α β". Decimals are Python's shortest
round-trip repr, which Rust's correctly rounded `str::parse::<f64>` reads back
bit for bit.
"""

import argparse
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
KEPT = [
    "uuid", "anyonwiki_code", "fusion_ring", "base_field", "in_unitary_gauge",
    "is_braided", "is_pivotal", "is_unitary", "f_symbols", "r_symbols",
    "p_symbols", "quantum_dimensions", "twists", "numeric_f_symbols",
    "numeric_r_symbols", "references", "software", "tex_names",
]
# Working precision (decimal digits) and the higher precision used to confirm
# that the rounding to binary64 is stable.
DPS, CONFIRM_DPS = 80, 160


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(args):
    db = pathlib.Path(args.database)
    for rel, digest in SHA256.items():
        got = sha256(db / rel)
        if got != digest:
            sys.exit(f"{rel}: SHA-256 {got} != pinned {digest}")
    code = [int(x) for x in args.code.split(",")]
    cats = json.load(open(db / CATEGORIES))["data"]
    matches = [e for e in cats.values() if e["anyonwiki_code"] == code]
    if len(matches) != 1:
        sys.exit(f"code {code}: {len(matches)} matches")
    entry = matches[0]
    if entry["base_field"] != "QQBar":
        sys.exit("only qqb_id (QQBar) entries are supported")
    ring = json.load(open(db / RINGS))["data"][entry["fusion_ring"]]
    out = {
        "source": {
            "repository": "https://github.com/anyonwiki/AnyonWikiDatabase",
            "commit": COMMIT,
            "license": "MIT, Copyright (c) 2026 AnyonWiki",
            "files": [{"path": p, "sha256": d} for p, d in SHA256.items()],
        },
        "category": {k: entry[k] for k in KEPT if k in entry},
        "fusion_ring": {k: ring[k] for k in ("uuid", "anyonwiki_code", "mult_tab", "references")},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


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
        return z.real, im


def to_f64(qqb):
    """Correctly rounded (re, im), confirmed stable at a higher precision."""
    re, im = qqb_value(qqb, DPS)
    re2, im2 = qqb_value(qqb, CONFIRM_DPS)
    pair = (float(re), float(im))
    if pair != (float(re2), float(im2)):
        sys.exit(f"{qqb}: rounding not stable between {DPS} and {CONFIRM_DPS} digits")
    return pair


def twist_f64(qqb):
    """Correctly rounded e^{2πiθ} for the twist fraction θ named by a qqb_id."""
    import mpmath

    def at(dps):
        theta, _ = qqb_value(qqb, dps)
        with mpmath.workdps(dps):
            z = mpmath.expjpi(2 * theta)
            return float(z.real), float(z.imag)

    pair = at(DPS)
    if pair != at(CONFIRM_DPS):
        sys.exit(f"{qqb}: twist rounding not stable between {DPS} and {CONFIRM_DPS} digits")
    return pair


def generate(args):
    data = json.load(open(args.extract))
    cat, ring = data["category"], data["fusion_ring"]
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mult = ring["mult_tab"]
    rank = len(mult)
    n_lines = [
        f"{a + 1} {b + 1} {c + 1} {mult[a][b][c]}"
        for a in range(rank) for b in range(rank) for c in range(rank)
        if mult[a][b][c] != 0
    ]
    f_lines, r_lines, check = [], [], []
    for key in sorted(cat["f_symbols"]):
        a, b, c, d, e, al, be, f, ga, de = key.split("_")
        re, im = to_f64(cat["f_symbols"][key])
        labels = f"{a} {b} {c} {d} {al} {e} {be} {ga} {f} {de}"
        f_lines.append(f"{labels} {re!r} {im!r}")
        nre, nim = cat["numeric_f_symbols"][key]
        check.append(f"F {labels} {cat['f_symbols'][key]} {re!r} {im!r} {nre!r} {nim!r}")
    for key in sorted(cat["r_symbols"]):
        a, b, c, al, be = key.split("_")
        re, im = to_f64(cat["r_symbols"][key])
        labels = f"{a} {b} {c} {al} {be}"
        r_lines.append(f"{labels} {re!r} {im!r}")
        nre, nim = cat["numeric_r_symbols"][key]
        check.append(f"R {labels} {cat['r_symbols'][key]} {re!r} {im!r} {nre!r} {nim!r}")
    # Twists are not shipped in the tables (the crate derives them from F and
    # R, as TensorKitSectors does); these rows are the correctly rounded
    # constants the closed-form provider uses, and the check of the table's
    # derived twist.
    for a, qqb in enumerate(cat["twists"], start=1):
        re, im = twist_f64(qqb)
        check.append(f"T {a} {qqb} {re!r} {im!r}")
    (out / "Nabc.txt").write_text("\n".join(n_lines) + "\n")
    (out / "F.txt").write_text("\n".join(f_lines) + "\n")
    (out / "R.txt").write_text("\n".join(r_lines) + "\n")
    import mpmath

    header = [
        f"# AnyonWikiDatabase {COMMIT}, category {cat['uuid']}, code {cat['anyonwiki_code']}",
        f"# generated by tools/anyonwiki/gen_category_tables.py (Python {sys.version.split()[0]}, mpmath {mpmath.__version__}, {DPS}/{CONFIRM_DPS} digits)",
        "# F/R: tag labels qqb_id shipped_re shipped_im anyonwiki_numeric_re anyonwiki_numeric_im",
        "# T: T object qqb_id(theta) re im  -- correctly rounded exp(2 pi i theta)",
    ]
    pathlib.Path(args.check).write_text("\n".join(header + check) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--database", required=True)
    e.add_argument("--code", required=True)
    e.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--extract", required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--check", required=True)
    args = parser.parse_args()
    {"extract": extract, "generate": generate}[args.step](args)


if __name__ == "__main__":
    main()
