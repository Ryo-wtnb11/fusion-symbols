/// A non-negative half-integer, such as an SU(2) spin `j`, stored as `2j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HalfInt(u32);

impl HalfInt {
    /// The value `n / 2`.
    pub const fn half(n: u32) -> Self {
        Self(n)
    }
    /// The integer `n`, or `None` when `2n` exceeds `u32::MAX`.
    pub const fn integer(n: u32) -> Option<Self> {
        match n.checked_mul(2) {
            Some(doubled) => Some(Self(doubled)),
            None => None,
        }
    }
    /// The doubled value `2j`.
    pub const fn doubled(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for HalfInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_multiple_of(2) {
            write!(f, "{}", self.0 / 2)
        } else {
            write!(f, "{}/2", self.0)
        }
    }
}
