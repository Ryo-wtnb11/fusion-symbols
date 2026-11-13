//! Small `prefill` requests against the serial entry loop over the same
//! items: empty, resident-only, single tiny, many tiny, skewed and
//! reduced-budget requests, tiny cold requests on a team token, and a child
//! token inside a team section. Not a CI gate; run with `cargo bench --bench
//! prefill_small`. Cold rows reset the caches outside the timed routine.
//!
//! Set `FS_PREFILL_BUDGET=reduced` to run only the reduced-budget rows: the
//! budgets are process-wide and fixed before the first cache access.

use criterion::{BatchSize, Criterion};
use fusion_symbols::cache::{
    self, configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier,
};
use fusion_symbols::category::{FusionCategory, GroupRepresentations};
use fusion_symbols::lie::{prefill, GroupId, LieError, LieGroup, LieIrrep, PrefillRequest};
use std::hint::black_box;
use std::time::Duration;
use threadteam::{Exec, Pool, Team};

enum Item {
    Cgc([LieIrrep; 3]),
    F([LieIrrep; 6]),
    R([LieIrrep; 3]),
    Fusion([LieIrrep; 2]),
    Phase(LieIrrep),
}

struct Row {
    group: LieGroup,
    items: Vec<Item>,
    request: PrefillRequest,
}

impl Row {
    fn new(group: &LieGroup, items: Vec<Item>) -> Self {
        let mut request = PrefillRequest::new(group);
        for item in &items {
            match item {
                Item::Cgc([a, b, c]) => request.cgc(a, b, c),
                Item::F([a, b, c, d, e, f]) => request.f_symbol(a, b, c, d, e, f),
                Item::R([a, b, c]) => request.r_symbol(a, b, c),
                Item::Fusion([a, b]) => request.fusion(a, b),
                Item::Phase(a) => request.frobenius_schur_phase(a),
            };
        }
        Self {
            group: group.clone(),
            items,
            request,
        }
    }

    /// The serial entries over the items, attempting every item and keeping
    /// the first error, as `prefill` does.
    fn entries(&self) -> Result<(), LieError> {
        let g = &self.group;
        let mut first = Ok(());
        for item in &self.items {
            let result = match item {
                Item::Cgc([a, b, c]) => g.cgc(a, b, c).map(drop),
                Item::F([a, b, c, d, e, f]) => g.f_symbol(a, b, c, d, e, f).map(drop),
                Item::R([a, b, c]) => g.r_symbol(a, b, c).map(drop),
                Item::Fusion([a, b]) => g.fusion(a, b).map(drop),
                Item::Phase(a) => g.frobenius_schur_phase(a).map(drop),
            };
            if first.is_ok() {
                first = result;
            }
        }
        first
    }

    fn prefill(&self, exec: Exec<'_>) -> Result<(), LieError> {
        prefill(&self.request, exec)
    }
}

fn su3() -> LieGroup {
    LieGroup::new(GroupId::su(3).unwrap()).unwrap()
}

fn irr(g: &LieGroup, d: &[i64]) -> LieIrrep {
    g.irrep(d).unwrap()
}

/// Every channel `a ⊗ b → c` with `a`, `b` among the trivial, fundamental,
/// antifundamental and symmetric-square irreps of SU(3).
fn tiny_channels(g: &LieGroup) -> Vec<Item> {
    let small = [[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]].map(|d| irr(g, &d));
    let mut items = Vec::new();
    for a in &small {
        for b in &small {
            for c in g.fusion(a, b).unwrap() {
                items.push(Item::Cgc([a.clone(), b.clone(), c]));
            }
        }
    }
    items
}

/// One heavy F block on the adjoint, then the tiny channels.
fn skewed(g: &LieGroup) -> Vec<Item> {
    let adj = irr(g, &[1, 1]);
    let mut items = vec![Item::F([(); 6].map(|_| adj.clone()))];
    items.extend(tiny_channels(g));
    items
}

/// Every item kind over small irreps.
fn mixed(g: &LieGroup) -> Vec<Item> {
    let [one, f, fb, adj] = [[0, 0], [1, 0], [0, 1], [1, 1]].map(|d| irr(g, &d));
    let mut items = tiny_channels(g);
    items.extend([
        Item::F([
            f.clone(),
            fb.clone(),
            f.clone(),
            f.clone(),
            one.clone(),
            f.clone(),
        ]),
        Item::F([
            f.clone(),
            f.clone(),
            fb.clone(),
            f.clone(),
            adj.clone(),
            one.clone(),
        ]),
        Item::R([f.clone(), fb.clone(), adj.clone()]),
        Item::Fusion([f.clone(), adj.clone()]),
        Item::Phase(f),
        Item::Phase(adj),
    ]);
    items
}

fn team() -> Team {
    Pool::builder()
        .threads(2)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(2)
        .unwrap()
}

/// `f` timed with the caches reset before every call, outside the timing.
fn cold(c: &mut Criterion, name: &str, f: impl Fn() -> Result<(), LieError>) {
    c.bench_function(name, |b| {
        b.iter_batched(
            cache::reset,
            |()| black_box(f()).unwrap(),
            BatchSize::PerIteration,
        )
    });
}

/// `f` timed on whatever the caches hold after one untimed call.
fn warm(c: &mut Criterion, name: &str, f: impl Fn() -> Result<(), LieError>) {
    cache::reset();
    f().unwrap();
    c.bench_function(name, |b| b.iter(|| black_box(f()).unwrap()));
}

fn default_rows(c: &mut Criterion) {
    let g = su3();
    let team = team();

    let empty = Row::new(&g, Vec::new());
    warm(c, "empty/entries", || empty.entries());
    warm(c, "empty/prefill_serial", || empty.prefill(Exec::serial()));

    let resident = Row::new(&g, mixed(&g));
    warm(c, "resident/entries", || resident.entries());
    warm(c, "resident/prefill_serial", || {
        resident.prefill(Exec::serial())
    });

    let rows = [
        (
            "single_tiny",
            Row::new(
                &g,
                vec![Item::Cgc([[1, 0], [0, 1], [1, 1]].map(|d| irr(&g, &d)))],
            ),
        ),
        ("many_tiny", Row::new(&g, tiny_channels(&g))),
        ("skewed", Row::new(&g, skewed(&g))),
    ];
    for (label, row) in &rows {
        cold(c, &format!("{label}/entries"), || row.entries());
        cold(c, &format!("{label}/prefill_serial"), || {
            row.prefill(Exec::serial())
        });
        cold(c, &format!("{label}/prefill_team"), || {
            row.prefill(team.exec(2))
        });
    }

    // Job 0 of a two-job team section runs the request on its child token.
    let many = &rows[1].1;
    let in_section = |f: &(dyn Fn(Exec<'_>) -> Result<(), LieError> + Sync)| {
        let out = std::sync::Mutex::new(Ok(()));
        team.exec(2).fork(2, |i, child| {
            if i == 0 {
                *out.lock().unwrap() = f(child);
            }
        });
        out.into_inner().unwrap()
    };
    cold(c, "child_team/entries", || in_section(&|_| many.entries()));
    cold(c, "child_team/prefill_child", || {
        in_section(&|child| many.prefill(child))
    });
}

/// Nothing fits, so every call regenerates its items.
fn reduced_rows(c: &mut Criterion) {
    configure_cache_budgets(
        CoefficientCacheBudgets::default()
            .with_limit(CoefficientCacheTier::Cgc, 1 << 10)
            .with_limit(CoefficientCacheTier::FSymbol, 64)
            .with_limit(CoefficientCacheTier::RSymbol, 64),
    )
    .unwrap();
    let g = su3();
    let row = Row::new(&g, mixed(&g));
    warm(c, "reduced_budget/entries", || row.entries());
    warm(c, "reduced_budget/prefill_serial", || {
        row.prefill(Exec::serial())
    });
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    if std::env::var("FS_PREFILL_BUDGET").as_deref() == Ok("reduced") {
        reduced_rows(&mut c);
    } else {
        default_rows(&mut c);
    }
    c.final_summary();
}
