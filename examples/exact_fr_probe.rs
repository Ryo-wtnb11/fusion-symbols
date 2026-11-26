//! Cost and agreement probe for exactly evaluated F and R blocks, one CSV row
//! per block on stdout. A measurement, not a test: it asserts nothing.
//!
//! ```text
//! cargo run --release --features exact-fr-probe --example exact_fr_probe -- \
//!     [--max-seconds S] [--only PAT]... [--skip PAT]... \
//!     [--b64-only | --exact-only] [--accum auto|big|check] [SPEC]...
//! ```
//!
//! A `SPEC` is `KIND:GROUP:LABELS`, labels as Dynkin labels `1,1` joined by
//! `/`. Kinds: `F` (six labels `a/b/c/d/e/f`, `F^{abc}_d[e,f]`), `R` (three,
//! `R^{ab}_c`), and for a sector `a/b/c` the two sextets of the A and B
//! symbols, `B` = `F^{a b b̄}_a[c, 𝟙]` and `A` = `F^{ā a b}_b[𝟙, c]`. Groups:
//! `suN`, `spN`, `spinN`, `soN`, `g2`, `f4`, `e6`, `e7`, `e8`, and the
//! product `su3xsu2` (labels concatenated in factor order). Without a `SPEC`
//! the built-in sweep runs; `--only`/`--skip` filter specs by substring.
//!
//! Every spec runs in a fresh child process (single-threaded, deterministic
//! values), killed after `--max-seconds` (default 600) and then reported as a
//! `SKIPPED` row. In the child, `cache::reset()` and a new `LieGroup` precede
//! each of the two cold measurements: the production binary64 query
//! (`b64_cold_s`, modules, channels and contraction included), then the exact
//! evaluation (`exact_gen_s`: modules, exact channels and common
//! denominators, of which `exact_capture_s` is copying exact coordinates out
//! of the descents and `exact_prep_s` the denominators and contents, the rest
//! being the production descent and elimination themselves; `exact_contract_s`: joins, integer accumulation, radicands
//! and the single rounding; of it, `exact_join_s` for the partial joins and
//! `exact_sum_s` for the final sum). Between them, untimed, the production
//! block's error bound is formed from its own (warm) rounded channels. No
//! disk tier is enabled. `--b64-only` and `--exact-only` measure one side
//! (the comparison columns stay empty); `--accum` selects the exact F joins
//! and final sum: `auto` (word joins into `i128` and stack-limb final sum
//! when their bounds hold, `join128+limbsN`; else `Q` joins, `qjoin+…`),
//! `big` (`Q` joins and `Q` sum, `qjoin+big`) or `check` (auto, then big;
//! the child panics and the row reads `FAILED` unless every `S` agrees).
//! Progress of every phase goes to stderr.
//!
//! Columns: `s_bits_*` are bit lengths of the exact integer numerators `S`
//! (`T` for R; per factor for products), `u_bits_*`/`v_bits_*` of the radicand
//! `U/V` passed to the rounding, `*_big_frac` the fraction of exact CGC
//! coordinates `x` and of accumulators `S` outside the inline word form of
//! `Q`, `max_ulp_nonzero` the largest ordered-binary64 step distance between
//! the production value and the rounded exact one over entries whose exact
//! value is nonzero, `max_abs_diff` the largest absolute difference over all
//! entries, `zero_exact_nonzero_prod` / `zero_exact_max_abs_prod` the count
//! and largest `|production|` of entries that are exactly zero but nonzero
//! in production, and `within_tol` whether every entry
//! satisfies `|current − exact_rounded| ≤ γ_k Σ|terms| + u |exact_rounded|`
//! (the forward bound of the §6.2 recipe, `k = 2(n + 7)` for F and
//! `2(n + 3)` for R with `n` summed terms; the last term is the rounding of
//! the exact value).

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use fusion_symbols::__exact_fr_probe as probe;
use fusion_symbols::cache;
use fusion_symbols::category::{FusionCategory, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep, RootSystem};

const HEADER: &str = "kind,group,labels,shape,entries,b64_cold_s,exact_gen_s,exact_capture_s,exact_prep_s,\
exact_contract_s,exact_join_s,exact_sum_s,exact_total_s,accum,s_bits_max,s_bits_med,u_bits_max,u_bits_med,v_bits_max,v_bits_med,x_big_frac,\
s_big_frac,max_ulp_nonzero,max_abs_diff,zero_exact_nonzero_prod,zero_exact_max_abs_prod,\
within_tol,status";

/// Sweep sectors `a/b/c`, smallest first. Each yields its B, A and R rows,
/// plus `F^{aaa}_a[a,a]` when `a = b = c`.
const SECTORS: &[(&str, &str)] = &[
    ("su3", "1,1/1,1/1,1"),
    ("su3xsu2", "1,1,1/1,1,1/1,1,2"),
    ("su3xsu2", "1,1,2/1,1,2/1,1,2"),
    ("g2", "1,0/1,0/1,0"),
    ("sp4", "1,1/1,1/2,0"),
    ("spin5", "1,1/1,1/0,2"),
    ("sp4", "1,1/1,1/1,1"),
    ("su4", "1,0,1/1,0,1/1,0,1"),
    ("f4", "0,0,0,1/0,0,0,1/0,0,0,1"),
    ("su3", "2,2/2,2/2,2"),
    ("g2", "2,0/2,0/2,0"),
    ("su3", "3,3/3,3/3,3"),
    ("e6", "0,1,0,0,0,0/0,1,0,0,0,0/0,1,0,0,0,0"),
    ("su4", "2,0,2/2,0,2/2,0,2"),
    ("e7", "0,0,0,0,0,0,1/0,0,0,0,0,0,1/1,0,0,0,0,0,0"),
    ("e7", "1,0,0,0,0,0,0/1,0,0,0,0,0,0/1,0,0,0,0,0,0"),
    ("e8", "0,0,0,0,0,0,0,1/0,0,0,0,0,0,0,1/0,0,0,0,0,0,0,1"),
    ("f4", "0,0,1,0/0,0,1,0/0,0,1,0"),
    ("su3", "6,6/6,6/6,6"),
    ("g2", "2,2/2,2/2,2"),
    ("e7", "0,0,0,0,0,1,0/0,0,0,0,0,1,0/0,0,0,0,0,1,0"),
];

/// Unit roundoff of binary64.
const U: f64 = f64::EPSILON / 2.0;

fn sweep() -> Vec<String> {
    let mut specs = Vec::new();
    for (group, labels) in SECTORS {
        for kind in ["B", "A", "R"] {
            specs.push(format!("{kind}:{group}:{labels}"));
        }
        let parts: Vec<&str> = labels.split('/').collect();
        if parts.iter().all(|x| *x == parts[0]) {
            let a = parts[0];
            specs.push(format!("F:{group}:{a}/{a}/{a}/{a}/{a}/{a}"));
        }
    }
    specs
}

fn group(tok: &str) -> Result<GroupId, String> {
    let n = |prefix: &str| {
        tok[prefix.len()..]
            .parse::<usize>()
            .map_err(|e| format!("group {tok}: {e}"))
    };
    let g = match tok {
        "g2" => return Ok(GroupId::g2()),
        "f4" => return Ok(GroupId::f4()),
        "e6" => return Ok(GroupId::e6()),
        "e7" => return Ok(GroupId::e7()),
        "e8" => return Ok(GroupId::e8()),
        "su3xsu2" => GroupId::cover(&[RootSystem::A(2), RootSystem::A(1)], 0),
        t if t.starts_with("spin") => GroupId::spin(n("spin")?),
        t if t.starts_with("sp") => GroupId::sp(n("sp")?),
        t if t.starts_with("su") => GroupId::su(n("su")?),
        t if t.starts_with("so") => GroupId::so(n("so")?),
        _ => return Err(format!("unknown group {tok}")),
    };
    g.map_err(|e| format!("group {tok}: {e:?}"))
}

fn dynkin(label: &str) -> Result<Vec<i64>, String> {
    label
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<i64>()
                .map_err(|e| format!("label {label}: {e}"))
        })
        .collect()
}

fn clean(s: &str) -> String {
    s.replace([',', '\n', '\r'], ";")
}

/// The first three columns, every measured column empty, then `status`.
fn blank_row(spec: &str, status: &str) -> String {
    let mut parts = spec.splitn(3, ':');
    let kind = parts.next().unwrap_or("");
    let group = parts.next().unwrap_or("");
    let labels = parts.next().unwrap_or("");
    let empty = HEADER.split(',').count() - 4;
    format!(
        "{kind},{group},\"{labels}\",{}{}",
        ",".repeat(empty),
        clean(status)
    )
}

fn median(mut v: Vec<u64>) -> String {
    if v.is_empty() {
        return String::new();
    }
    v.sort_unstable();
    v[v.len() / 2].to_string()
}

fn max(v: &[u64]) -> String {
    v.iter().max().map_or_else(String::new, u64::to_string)
}

fn frac(part: u64, whole: u64) -> String {
    if whole == 0 {
        String::new()
    } else {
        format!("{:.6}", part as f64 / whole as f64)
    }
}

/// Steps between `a` and `b` on the ordered binary64 line (`±0` coincide,
/// a pair across zero counts the steps through it). Only meaningful when the
/// exact value is nonzero: from `0` to `1e-17` is about `4e18` steps.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        i128::from(if bits < 0 { i64::MIN - bits } else { bits })
    };
    (key(a) - key(b)).unsigned_abs() as u64
}

/// The six (F) or three (R) objects of the query of `kind`.
fn objects(g: &LieGroup, kind: &str, raw: Vec<LieIrrep>) -> Result<Vec<LieIrrep>, String> {
    let n = raw.len();
    if matches!((kind, n), ("F", 6) | ("R", 3)) {
        return Ok(raw);
    }
    let one = g.unit();
    match (kind, raw.as_slice()) {
        ("B", [a, b, c]) => Ok(vec![
            a.clone(),
            b.clone(),
            g.dual(b),
            a.clone(),
            c.clone(),
            one,
        ]),
        ("A", [a, b, c]) => Ok(vec![
            g.dual(a),
            a.clone(),
            b.clone(),
            b.clone(),
            one,
            c.clone(),
        ]),
        _ => Err(format!("kind {kind} with {n} labels")),
    }
}

fn build(gid: &GroupId, kind: &str, labels: &str) -> Result<(LieGroup, Vec<LieIrrep>), String> {
    let g = LieGroup::new(gid.clone()).map_err(|e| e.to_string())?;
    let raw = labels
        .split('/')
        .map(|l| g.irrep(&dynkin(l)?).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let objects = objects(&g, kind, raw)?;
    Ok((g, objects))
}

/// Which sides a child measures, and how the exact F sum accumulates.
#[derive(Clone, Copy)]
struct Opts {
    b64: bool,
    exact: bool,
    accum: probe::Accum,
}

impl Opts {
    fn args(self) -> [&'static str; 2] {
        let mode = match (self.b64, self.exact) {
            (true, false) => "b64",
            (false, true) => "exact",
            _ => "both",
        };
        let accum = match self.accum {
            probe::Accum::Auto => "auto",
            probe::Accum::Big => "big",
            probe::Accum::Check => "check",
        };
        [mode, accum]
    }

    fn parse(mode: &str, accum: &str) -> Result<Self, String> {
        let (b64, exact) = match mode {
            "both" => (true, true),
            "b64" => (true, false),
            "exact" => (false, true),
            _ => return Err(format!("mode {mode}")),
        };
        let accum = match accum {
            "auto" => probe::Accum::Auto,
            "big" => probe::Accum::Big,
            "check" => probe::Accum::Check,
            _ => return Err(format!("--accum {accum}: auto, big or check")),
        };
        Ok(Self { b64, exact, accum })
    }
}

fn shape(dims: &[usize]) -> String {
    dims.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn measure(spec: &str, o: Opts) -> Result<String, String> {
    let mut parts = spec.splitn(3, ':');
    let (Some(kind), Some(gtok), Some(labels)) = (parts.next(), parts.next(), parts.next()) else {
        return Err(format!("bad spec {spec}"));
    };
    let gid = group(gtok)?;
    let err = |e: fusion_symbols::lie::LieError| e.to_string();

    // Cold production query, then its error bound from the warm channels.
    let production = if o.b64 {
        eprintln!("exact-fr-probe: {spec}: binary64 query");
        cache::reset();
        let (g, x) = build(&gid, kind, labels)?;
        let side = if kind == "R" {
            let t = Instant::now();
            let block = g.r_symbol(&x[0], &x[1], &x[2]).map_err(err)?;
            let secs = t.elapsed().as_secs_f64();
            eprintln!("exact-fr-probe: binary64 done in {secs:.3}s; error bound");
            let tol = probe::tol_r(&g, [&x[0], &x[1], &x[2]]).map_err(err)?;
            (secs, block.dims().to_vec(), block.as_slice().to_vec(), tol)
        } else {
            let t = Instant::now();
            let block = g
                .f_symbol(&x[0], &x[1], &x[2], &x[3], &x[4], &x[5])
                .map_err(err)?;
            let secs = t.elapsed().as_secs_f64();
            eprintln!("exact-fr-probe: binary64 done in {secs:.3}s; error bound");
            let tol = probe::tol_f(&g, [&x[0], &x[1], &x[2], &x[3], &x[4], &x[5]]).map_err(err)?;
            (secs, block.dims().to_vec(), block.as_slice().to_vec(), tol)
        };
        Some(side)
    } else {
        None
    };

    // Cold exact evaluation.
    let exact = if o.exact {
        eprintln!("exact-fr-probe: {spec}: exact evaluation");
        cache::reset();
        let (g, x) = build(&gid, kind, labels)?;
        let block = if kind == "R" {
            probe::exact_r(&g, [&x[0], &x[1], &x[2]])
        } else {
            probe::exact_f(&g, [&x[0], &x[1], &x[2], &x[3], &x[4], &x[5]], o.accum)
        }
        .map_err(err)?;
        Some(block)
    } else {
        None
    };

    let e = String::new;
    let mut row = vec![kind.to_string(), gtok.to_string(), format!("\"{labels}\"")];
    let dims = match (&production, &exact) {
        (Some((_, dims, ..)), _) => dims.clone(),
        (None, Some(x)) => x.dims.clone(),
        (None, None) => Vec::new(),
    };
    let entries: usize = dims.iter().product();
    row.push(shape(&dims));
    row.push(entries.to_string());
    row.push(
        production
            .as_ref()
            .map_or_else(e, |p| format!("{:.6e}", p.0)),
    );
    match &exact {
        Some(x) => {
            let st = &x.stats;
            row.extend([
                format!("{:.6e}", st.gen_secs),
                format!("{:.6e}", st.capture_secs),
                format!("{:.6e}", st.prep_secs),
                format!("{:.6e}", st.contract_secs),
                format!("{:.6e}", st.join_secs),
                format!("{:.6e}", st.sum_secs),
                format!("{:.6e}", st.gen_secs + st.contract_secs),
                st.accum.clone(),
                max(&st.s_bits),
                median(st.s_bits.clone()),
                max(&st.u_bits),
                median(st.u_bits.clone()),
                max(&st.v_bits),
                median(st.v_bits.clone()),
                frac(st.x_big, st.x_values),
                frac(st.s_big, st.s_values),
            ]);
        }
        None => row.extend(std::iter::repeat_with(e).take(16)),
    }
    let mut status = "OK";
    match (&production, &exact) {
        (Some((_, dims, current, tol)), Some(x)) => {
            if x.dims != *dims || x.values.len() != current.len() || tol.len() != current.len() {
                status = "DIMS_MISMATCH";
            }
            let (mut max_ulp, mut max_abs, mut within) = (0u64, 0.0f64, true);
            // Entries whose exact value is 0 but whose production value is not.
            let (mut zero_count, mut zero_max) = (0u64, 0.0f64);
            for ((&c, &v), &t) in current.iter().zip(&x.values).zip(tol) {
                let diff = (c - v).abs();
                if v == 0.0 {
                    if c != 0.0 {
                        zero_count += 1;
                        zero_max = zero_max.max(c.abs());
                    }
                } else {
                    max_ulp = max_ulp.max(ulps(c, v));
                }
                max_abs = max_abs.max(diff);
                within &= diff <= t + U * v.abs();
            }
            row.extend([
                max_ulp.to_string(),
                format!("{max_abs:.3e}"),
                zero_count.to_string(),
                format!("{zero_max:.3e}"),
                within.to_string(),
            ]);
        }
        _ => row.extend(std::iter::repeat_with(e).take(5)),
    }
    row.push(status.to_string());
    debug_assert_eq!(row.len(), HEADER.split(',').count());
    Ok(row.join(","))
}

fn run_child(spec: &str, opts: Opts, limit: Duration) -> String {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => return blank_row(spec, &format!("ERROR: {e}")),
    };
    let mut child = match Command::new(exe)
        .arg("--one")
        .arg(spec)
        .args(opts.args())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return blank_row(spec, &format!("ERROR: spawn {e}")),
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                let row = out.trim_end();
                return if status.success() && !row.is_empty() {
                    row.to_string()
                } else {
                    blank_row(spec, &format!("FAILED: {status}"))
                };
            }
            Ok(None) if start.elapsed() > limit => {
                let _ = child.kill();
                let _ = child.wait();
                return blank_row(spec, &format!("SKIPPED: over {}s", limit.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return blank_row(spec, &format!("ERROR: wait {e}")),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [flag, spec, mode, accum] = args.as_slice() {
        if flag == "--one" {
            let row = Opts::parse(mode, accum).and_then(|o| measure(spec, o));
            println!(
                "{}",
                row.unwrap_or_else(|e| blank_row(spec, &format!("ERROR: {e}")))
            );
            return;
        }
    }
    let (mut limit, mut only, mut skip, mut specs) = (600u64, Vec::new(), Vec::new(), Vec::new());
    let (mut mode, mut accum) = ("both", String::from("auto"));
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--max-seconds" => {
                limit = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--max-seconds takes a whole number of seconds");
            }
            "--only" => only.push(it.next().expect("--only takes a pattern")),
            "--skip" => skip.push(it.next().expect("--skip takes a pattern")),
            "--b64-only" => mode = "b64",
            "--exact-only" => mode = "exact",
            "--accum" => accum = it.next().expect("--accum takes auto, big or check"),
            _ => specs.push(arg),
        }
    }
    let opts = Opts::parse(mode, &accum).unwrap_or_else(|e| panic!("{e}"));
    if specs.is_empty() {
        specs = sweep();
    }
    println!("{HEADER}");
    for spec in specs {
        if (!only.is_empty() && !only.iter().any(|p| spec.contains(p.as_str())))
            || skip.iter().any(|p| spec.contains(p.as_str()))
        {
            continue;
        }
        println!("{}", run_child(&spec, opts, Duration::from_secs(limit)));
    }
}
