//! Negative controls for the public `category::residual` functions: each
//! residual is zero on the shipped Fibonacci category and nonzero, at the
//! labels that use the damaged symbol, once one symbol is damaged through a
//! wrapper that implements the public `FusionCategory` trait.
//! Frobenius-Schur and twist have no residual function, so they have no control here.
use fusion_symbols::{
    category::{
        f_unitarity_residual, hexagon_residual, pentagon_residual, BraidingStyle, FusionCategory,
        FusionStyle,
    },
    table::{TableCategory, TableError, TableFusion, TableObject},
};
use num_complex::Complex64;

type O = TableObject;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Fault {
    None,
    /// `F^{τττ}_τ[1,1]` shifted by 1/4: breaks pentagon and unitarity.
    FEntry,
    /// `F^{τττ}_τ` scaled by 3/2: unitary rows no longer have norm 1.
    FScale,
    /// `R^{ττ}_τ` negated: breaks the hexagon.
    RSign,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Damaged {
    inner: TableCategory,
    fault: Fault,
}

impl Damaged {
    fn new(fault: Fault) -> Self {
        Self {
            inner: TableCategory::fibonacci(),
            fault,
        }
    }
    fn one(&self) -> O {
        self.inner.object(1).unwrap()
    }
    fn tau(&self) -> O {
        self.inner.object(2).unwrap()
    }
}

impl FusionCategory for Damaged {
    type Object = O;
    type Scalar = Complex64;
    type FSymbol = Complex64;
    type RSymbol = Complex64;
    type Fusion<'a> = TableFusion<'a>;
    type Error = TableError;
    fn fusion_style(&self) -> FusionStyle {
        self.inner.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyle {
        self.inner.braiding_style()
    }
    fn unit(&self) -> O {
        self.inner.unit()
    }
    fn contains(&self, a: &O) -> bool {
        self.inner.contains(a)
    }
    fn dual(&self, a: &O) -> O {
        self.inner.dual(a)
    }
    fn dim(&self, a: &O) -> f64 {
        self.inner.dim(a)
    }
    fn frobenius_schur_phase(&self, a: &O) -> Result<Complex64, TableError> {
        self.inner.frobenius_schur_phase(a)
    }
    fn frobenius_schur_indicator(&self, a: &O) -> Complex64 {
        self.inner.frobenius_schur_indicator(a)
    }
    fn twist(&self, a: &O) -> Complex64 {
        self.inner.twist(a)
    }
    fn fusion<'a>(&'a self, a: &O, b: &O) -> Result<TableFusion<'a>, TableError> {
        self.inner.fusion(a, b)
    }
    fn nsymbol(&self, a: &O, b: &O, c: &O) -> Result<u32, TableError> {
        self.inner.nsymbol(a, b, c)
    }
    fn f_symbol(&self, a: &O, b: &O, c: &O, d: &O, e: &O, f: &O) -> Result<Complex64, TableError> {
        let x = self.inner.f_symbol(a, b, c, d, e, f)?;
        let tau = self.tau();
        if [a, b, c, d] != [&tau; 4] {
            return Ok(x);
        }
        Ok(match self.fault {
            Fault::FEntry if *e == self.one() && *f == self.one() => x + 0.25,
            Fault::FScale => x * 1.5,
            _ => x,
        })
    }
    fn r_symbol(&self, a: &O, b: &O, c: &O) -> Result<Complex64, TableError> {
        let x = self.inner.r_symbol(a, b, c)?;
        let tau = self.tau();
        Ok(if self.fault == Fault::RSign && [a, b, c] == [&tau; 3] {
            -x
        } else {
            x
        })
    }
}

const TOL: f64 = 1e-9;

#[test]
fn unbroken_category_has_zero_residuals_everywhere() {
    let cat = Damaged::new(Fault::None);
    let labels = [cat.one(), cat.tau()];
    for a in &labels {
        for b in &labels {
            for c in &labels {
                assert!(hexagon_residual(&cat, a, b, c).unwrap() < TOL);
                for d in &labels {
                    assert!(pentagon_residual(&cat, a, b, c, d).unwrap() < TOL);
                    assert!(f_unitarity_residual(&cat, a, b, c, d).unwrap() < TOL);
                }
            }
        }
    }
}

#[test]
fn f_unitarity_residual_reports_a_nonunitary_block_only_at_its_labels() {
    let cat = Damaged::new(Fault::FScale);
    let (one, tau) = (cat.one(), cat.tau());
    // 3/2 scaling: M M^dagger = 9/4 on the unit-row entries, so the residual is at least 1.
    assert!(f_unitarity_residual(&cat, &tau, &tau, &tau, &tau).unwrap() > 1.0);
    // The block for (tau, tau, 1, tau) has no damaged entry.
    assert!(f_unitarity_residual(&cat, &tau, &tau, &one, &tau).unwrap() < TOL);
}

#[test]
fn f_unitarity_residual_reports_a_shifted_entry() {
    let cat = Damaged::new(Fault::FEntry);
    let (one, tau) = (cat.one(), cat.tau());
    assert!(f_unitarity_residual(&cat, &tau, &tau, &tau, &tau).unwrap() > 0.1);
    assert!(f_unitarity_residual(&cat, &tau, &tau, &one, &tau).unwrap() < TOL);
}

#[test]
fn pentagon_residual_reports_a_shifted_f_entry_only_where_it_is_used() {
    let cat = Damaged::new(Fault::FEntry);
    let (one, tau) = (cat.one(), cat.tau());
    assert!(pentagon_residual(&cat, &tau, &tau, &tau, &tau).unwrap() > 0.1);
    // Every F in this pentagon has a unit outer label, so none is the damaged one.
    assert!(pentagon_residual(&cat, &one, &one, &one, &one).unwrap() < TOL);
}

#[test]
fn hexagon_residual_reports_a_sign_error_in_r_only_where_it_is_used() {
    let cat = Damaged::new(Fault::RSign);
    let (one, tau) = (cat.one(), cat.tau());
    assert!(hexagon_residual(&cat, &tau, &tau, &tau).unwrap() > 0.1);
    // R with a unit leg is untouched.
    assert!(hexagon_residual(&cat, &one, &tau, &tau).unwrap() < TOL);
}
