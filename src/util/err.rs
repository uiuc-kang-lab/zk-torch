use std::fmt;

#[derive(Debug, Clone)]
pub struct CQOutOfRangeError {
  pub input: i128,
}
