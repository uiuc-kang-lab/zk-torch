/*
 * Iteration utilities:
 * The functions are used for iterating over arrays and vectors.
 * Each function/macro has CPU and GPU implementations.
 */
use ndarray::ArrayD;
#[cfg(feature = "gpu")]
use rayon::prelude::*;

#[cfg(feature = "gpu")]
pub fn array_into_iter<T: Send + Sync>(x: &ArrayD<T>) -> impl ParallelIterator<Item = &T> {
  x.into_par_iter()
}

#[cfg(not(feature = "gpu"))]
pub fn array_into_iter<T>(x: &ArrayD<T>) -> impl Iterator<Item = &T> {
  x.into_iter()
}

#[cfg(feature = "gpu")]
pub fn vec_iter<T: Send + Sync>(x: &Vec<T>) -> impl ParallelIterator<Item = &T> {
  x.par_iter()
}

#[cfg(not(feature = "gpu"))]
pub fn vec_iter<T>(x: &Vec<T>) -> impl Iterator<Item = &T> {
  x.iter()
}

#[macro_export]
macro_rules! ndarr_azip {
  ($($arg:tt)*) => {
    #[cfg(feature = "gpu")]
    {
      par_azip!($($arg)*)
    }
    #[cfg(not(feature = "gpu"))]
    {
      azip!($($arg)*)
    }
  };
}
