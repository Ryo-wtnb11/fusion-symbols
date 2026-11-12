//! Warm public-entry hits for every category kind, and label construction.
//! Not a CI gate. Labels are built outside the timed closures and every row is
//! a monomorphized closure.
//!
//! - `cargo bench --bench warm_hits`: Criterion timings.
//! - `FS_WARM_MODE=allocs cargo bench --bench warm_hits`: allocation calls of
//!   one warm call per row (0 except owned A/B and composed Product outputs,
//!   and the label-construction `irrep` rows).
//! - `FS_WARM_MODE=dist cargo bench --bench warm_hits`: the label-tier key
//!   hash distribution, beside quick_cache's default foldhash over the same
//!   keys written as the previous per-word key hash did.
//!
//! Distribution coverage: for each root A1, A2, A4, B2, C3, D4, D5, G2, F4,
//! E6, E7 and E8, six distinct labels: the unit, ω1, a complex conjugate pair
//! when the root has one (otherwise ω2 and ω_r), ω1 + ω_r and 2ω1, padded with
//! multiples of ω1 when the root has too few. Keys: every ordered sextet as a
//! Cartan F key, every ordered triple as an R key and as a CGC channel key,
//! every label as a module and a phase key, and every unordered pair in the
//! production word order with both Product projections. The SU(2) F form is
//! not covered: its packed Regge key is crate-private.
#[path = "../tests/common/mod.rs"]
mod common;
use common::counting_alloc;

/// The crate's own fingerprint source, so the report measures exactly the
/// production pipeline.
#[path = "../src/cache/fingerprint.rs"]
#[allow(dead_code)]
mod fingerprint;
mod lie {
    pub use fusion_symbols::lie::RootSystem;
}

use criterion::Criterion;
use fusion_symbols::category::{FusionCategory, GroupRepresentations};
use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep, RootSystem, U1Irrep, U1};
use fusion_symbols::product::Product;
use fusion_symbols::table::TableCategory;
use std::hint::black_box;

#[global_allocator]
static ALLOCATOR: counting_alloc::Counting = counting_alloc::Counting;

/// Times each row with Criterion, or else reports its warm allocations.
struct Runner(Option<Criterion>);

impl Runner {
    fn row<T>(&mut self, name: &str, query: impl Fn() -> T) {
        black_box(query());
        match &mut self.0 {
            Some(c) => {
                c.bench_function(name, |b| b.iter(|| black_box(query())));
            }
            None => {
                let (calls, bytes, _, value) = counting_alloc::measure(&query);
                black_box(value);
                println!("{name},{calls},{bytes}");
            }
        }
    }
}

/// N, CGC, R and F of `x`, its dual and the unit, and the construction of `x`.
fn lie_rows(r: &mut Runner, name: &str, g: &LieGroup, dynkin: &[i64]) -> LieIrrep {
    let x = g.irrep(dynkin).unwrap();
    let (d, one) = (g.dual(&x), g.unit());
    r.row(&format!("{name}/N"), || g.nsymbol(&x, &d, &one).unwrap());
    r.row(&format!("{name}/CGC"), || g.cgc(&x, &d, &one).unwrap());
    r.row(&format!("{name}/R"), || g.r_symbol(&x, &d, &one).unwrap());
    r.row(&format!("{name}/F"), || {
        g.f_symbol(&x, &d, &x, &x, &one, &one).unwrap()
    });
    r.row(&format!("{name}/irrep"), || g.irrep(dynkin).unwrap());
    x
}

/// The octet rows: a multiplicity-two vertex and the owned A/B outputs.
fn octet_rows(r: &mut Runner, name: &str, g: &LieGroup, adjoint: &[i64]) {
    let o = g.irrep(adjoint).unwrap();
    r.row(&format!("{name}/N-adj"), || g.nsymbol(&o, &o, &o).unwrap());
    r.row(&format!("{name}/CGC-adj"), || g.cgc(&o, &o, &o).unwrap());
    r.row(&format!("{name}/R-adj"), || g.r_symbol(&o, &o, &o).unwrap());
    r.row(&format!("{name}/F-adj"), || {
        g.f_symbol(&o, &o, &o, &o, &o, &o).unwrap()
    });
    r.row(&format!("{name}/A-adj"), || g.a_symbol(&o, &o, &o).unwrap());
    r.row(&format!("{name}/B-adj"), || g.b_symbol(&o, &o, &o).unwrap());
}

fn rows(r: &mut Runner) {
    let lie = |id: GroupId| LieGroup::new(id).unwrap();
    let su2 = lie(GroupId::su(2).unwrap());
    lie_rows(r, "SU2-j10", &su2, &[20]);
    let su3 = lie(GroupId::su(3).unwrap());
    lie_rows(r, "SU3-3", &su3, &[1, 0]);
    octet_rows(r, "SU3", &su3, &[1, 1]);
    let su4 = lie(GroupId::su(4).unwrap());
    lie_rows(r, "SU4-4", &su4, &[1, 0, 0]);
    octet_rows(r, "SU4", &su4, &[1, 0, 1]);
    lie_rows(r, "B2-5", &lie(GroupId::so(5).unwrap()), &[1, 0]);
    lie_rows(r, "C3-6", &lie(GroupId::sp(6).unwrap()), &[1, 0, 0]);
    lie_rows(r, "D4-8v", &lie(GroupId::spin(8).unwrap()), &[1, 0, 0, 0]);
    lie_rows(
        r,
        "D5-16",
        &lie(GroupId::spin(10).unwrap()),
        &[0, 0, 0, 1, 0],
    );
    lie_rows(r, "G2-7", &lie(GroupId::g2()), &[1, 0]);
    lie_rows(r, "F4-26", &lie(GroupId::f4()), &[0, 0, 0, 1]);
    let e6 = lie(GroupId::e6());
    let x = lie_rows(r, "E6-27", &e6, &[1, 0, 0, 0, 0, 0]);
    r.row("E6-27/FS", || e6.frobenius_schur_phase(&x).unwrap());
    lie_rows(r, "E7-56", &lie(GroupId::e7()), &[0, 0, 0, 0, 0, 0, 1]);
    lie_rows(r, "E8-248", &lie(GroupId::e8()), &[0, 0, 0, 0, 0, 0, 0, 1]);

    // A tuple product, a closed-form factor product and a composite cover.
    let p = Product::new((su3.clone(), su2.clone()));
    let x = (su3.irrep(&[1, 1]).unwrap(), su2.irrep(&[1]).unwrap());
    let one = p.unit();
    r.row("SU3xSU2/N", || p.nsymbol(&x, &x, &one).unwrap());
    r.row("SU3xSU2/R", || p.r_symbol(&x, &x, &one).unwrap());
    r.row("SU3xSU2/F", || {
        p.f_symbol(&x, &x, &x, &x, &one, &one).unwrap()
    });
    let p = Product::new((U1, su2.clone()));
    let x = (U1Irrep::new(1).unwrap(), su2.irrep(&[1]).unwrap());
    let d = p.dual(&x);
    let one = p.unit();
    r.row("U1xSU2/N", || p.nsymbol(&x, &d, &one).unwrap());
    r.row("U1xSU2/R", || p.r_symbol(&x, &d, &one).unwrap());
    r.row("U1xSU2/F", || {
        p.f_symbol(&x, &d, &x, &x, &one, &one).unwrap()
    });
    let cover = lie(GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 0).unwrap());
    let x = cover.irrep(&[1, 1, 1]).unwrap();
    let one = cover.unit();
    r.row("cover-SU2xSU3/N", || cover.nsymbol(&x, &x, &one).unwrap());
    r.row("cover-SU2xSU3/CGC", || cover.cgc(&x, &x, &one).unwrap());
    r.row("cover-SU2xSU3/R", || cover.r_symbol(&x, &x, &one).unwrap());
    r.row("cover-SU2xSU3/F", || {
        cover.f_symbol(&x, &x, &x, &x, &one, &one).unwrap()
    });
    r.row("cover-SU2xSU3/irrep", || cover.irrep(&[1, 1, 1]).unwrap());

    // Control: table categories enter no coefficient tier.
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    r.row("Fibonacci/N", || fib.nsymbol(&tau, &tau, &tau).unwrap());
    r.row("Fibonacci/R", || fib.r_symbol(&tau, &tau, &tau).unwrap());
    r.row("Fibonacci/F", || {
        fib.f_symbol(&tau, &tau, &tau, &tau, &tau, &tau).unwrap()
    });
}

/// The six distinct labels of the coverage note, from the public API.
fn label_set(id: GroupId) -> (RootSystem, Vec<Vec<i64>>) {
    let root = id.root_systems()[0];
    let g = LieGroup::new(id).unwrap();
    let r = root.rank();
    let omega = |i: usize, n: i64| {
        let mut w = vec![0; r];
        w[i] += n;
        w
    };
    let dual = |w: &[i64]| -> Vec<i64> { g.dual(&g.irrep(w).unwrap()).dynkin().collect() };
    let mut set = vec![vec![0; r], omega(0, 1)];
    match (0..r).find(|&i| dual(&omega(i, 1)) != omega(i, 1)) {
        Some(i) => set.extend([omega(i, 1), dual(&omega(i, 1))]),
        None => set.extend([omega(1.min(r - 1), 1), omega(r - 1, 1)]),
    }
    let mut sum = omega(0, 1);
    sum[r - 1] += 1;
    set.extend([sum, omega(0, 2)]);
    let mut out: Vec<Vec<i64>> = Vec::new();
    for w in set.into_iter().chain((3..).map(|n| omega(0, n))) {
        if !out.contains(&w) {
            out.push(w);
        }
        if out.len() == 6 {
            break;
        }
    }
    (root, out)
}

/// `(keys, distinct, used, ideal, max load, tag χ²)` at load ½.
fn spread(hashes: &[u64]) -> String {
    let n = hashes.len();
    let mut sorted = hashes.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let buckets = (2 * n).next_power_of_two();
    let mut load = vec![0u32; buckets];
    let mut tags = [0u32; 128];
    for &h in hashes {
        load[h as usize & (buckets - 1)] += 1;
        tags[(h >> 57) as usize] += 1;
    }
    let used = load.iter().filter(|&&x| x > 0).count();
    let ideal = buckets as f64 * (1.0 - (1.0 - 1.0 / buckets as f64).powi(n as i32));
    let expected = n as f64 / 128.0;
    let chi2: f64 = tags
        .iter()
        .map(|&t| (f64::from(t) - expected).powi(2) / expected)
        .sum();
    format!(
        "{n},{},{used},{ideal:.0},{},{chi2:.0}",
        sorted.len(),
        load.iter().max().unwrap()
    )
}

fn dist() {
    use fingerprint::{key_hash, label_hash, FoldBuild};
    use std::hash::{BuildHasher, Hash, Hasher};
    let foldhash = quick_cache::DefaultHashBuilder::default();
    // The per-word key hash the label tiers replaced, through foldhash.
    let previous = |prefix: &[u8], root: RootSystem, parts: &[&[i64]]| {
        let mut h = foldhash.build_hasher();
        for &b in prefix {
            b.hash(&mut h);
        }
        root.hash(&mut h);
        for x in parts.iter().flat_map(|p| p.iter()) {
            x.hash(&mut h);
        }
        h.finish()
    };
    println!("root,form,keys,distinct,buckets_used,ideal,max_load,tag_chi2,hasher");
    for id in [
        GroupId::su(2).unwrap(),
        GroupId::su(3).unwrap(),
        GroupId::su(5).unwrap(),
        GroupId::spin(5).unwrap(),
        GroupId::sp(6).unwrap(),
        GroupId::spin(8).unwrap(),
        GroupId::spin(10).unwrap(),
        GroupId::g2(),
        GroupId::f4(),
        GroupId::e6(),
        GroupId::e7(),
        GroupId::e8(),
    ] {
        let (root, labels) = label_set(id);
        let fp: Vec<u64> = labels.iter().map(|l| label_hash(l)).collect();
        let fold = |x: u64| FoldBuild.hash_one(x);
        let report = |form: &str, new: Vec<u64>, old: Vec<u64>| {
            println!("{root:?},{form},{},new", spread(&new));
            println!("{root:?},{form},{},foldhash", spread(&old));
        };
        let idx = 0..labels.len();
        let mut f = (Vec::new(), Vec::new());
        let mut triples = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        for k in 0..6usize.pow(6) {
            let s: [usize; 6] = std::array::from_fn(|i| k / 6usize.pow(i as u32) % 6);
            f.0.push(fold(key_hash(0x100, root, s.map(|i| fp[i]))));
            f.1.push(previous(&[1, 0], root, &s.map(|i| &labels[i][..])));
            if k < 216 {
                for (kind, t) in triples.iter_mut().enumerate() {
                    let s = [s[0], s[1], s[2]];
                    t.0.push(fold(key_hash(kind as u64, root, s.map(|i| fp[i]))));
                    t.1.push(previous(&[kind as u8], root, &s.map(|i| &labels[i][..])));
                }
            }
        }
        report("F", f.0, f.1);
        let [r, channel] = triples;
        report("R", r.0, r.1);
        report("CGC-channel", channel.0, channel.1);
        for (kind, form) in [(0u8, "CGC-module"), (2, "CGC-phase")] {
            report(
                form,
                idx.clone()
                    .map(|i| fold(key_hash(kind.into(), root, [fp[i]])))
                    .collect(),
                idx.clone()
                    .map(|i| previous(&[kind], root, &[&labels[i]]))
                    .collect(),
            );
        }
        let mut product = (Vec::new(), Vec::new());
        for i in idx.clone() {
            for j in idx.clone() {
                if labels[i] > labels[j] {
                    continue;
                }
                for flag in [false, true] {
                    let tag = 0x300 | u64::from(flag);
                    product.0.push(fold(key_hash(tag, root, [fp[i], fp[j]])));
                    product
                        .1
                        .push(foldhash.hash_one((root, &labels[i][..], &labels[j][..], flag)));
                }
            }
        }
        report("Product", product.0, product.1);
    }
}

fn main() {
    match std::env::var("FS_WARM_MODE").as_deref() {
        Ok("dist") => dist(),
        Ok("allocs") => {
            let _serial = counting_alloc::serial();
            println!("row,warm_alloc_calls,warm_alloc_bytes");
            rows(&mut Runner(None));
        }
        _ => {
            let mut runner = Runner(Some(Criterion::default().configure_from_args()));
            rows(&mut runner);
            if let Some(c) = runner.0 {
                c.final_summary();
            }
        }
    }
}
