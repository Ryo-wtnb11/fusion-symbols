/// A non-negative half-integer, such as an SU(2) spin `j`, stored as `2j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HalfInt(u32);

/// An invalid denominator or an unrepresentable doubled value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalfIntError {
    /// The denominator is neither `1` nor `2`; fractions are not reduced.
    Denominator,
    /// Twice the requested value exceeds `u32::MAX`.
    Overflow,
}

impl std::fmt::Display for HalfIntError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Denominator => "half-integer denominator must be 1 or 2",
            Self::Overflow => "doubled half-integer exceeds u32::MAX",
        })
    }
}

impl std::error::Error for HalfIntError {}

impl HalfInt {
    /// The value `num / den`, with denominator `1` or `2`.
    ///
    /// # Errors
    ///
    /// [`HalfIntError::Denominator`] for any other denominator, including
    /// unreduced fractions such as `2/4`; [`HalfIntError::Overflow`] when
    /// `den == 1` and `2 * num` exceeds `u32::MAX`.
    pub const fn new(num: u32, den: u32) -> Result<Self, HalfIntError> {
        match den {
            1 => match num.checked_mul(2) {
                Some(doubled) => Ok(Self(doubled)),
                None => Err(HalfIntError::Overflow),
            },
            2 => Ok(Self(num)),
            _ => Err(HalfIntError::Denominator),
        }
    }

    /// The value `n / 2`, given its doubled representation.
    pub const fn from_doubled(n: u32) -> Self {
        Self(n)
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
