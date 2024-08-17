/*
 * Serialization utilities for converting between serde and ark_serialize.
 * And other file I/O utilities.
 */
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};

// For serialization, ArrayD uses serde while G1Affine uses ark_serialize.
// In order to bridge between the two, the following code snippet is used:
// https://github.com/arkworks-rs/algebra/issues/178#issuecomment-1413219278
pub fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  let mut bytes = vec![];
  a.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
  s.serialize_bytes(&bytes)
}

pub fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
  let a = A::deserialize_compressed_unchecked(s.as_slice());
  a.map_err(serde::de::Error::custom)
}

pub fn measure_file_size(file_path: &str) -> u64 {
  let file = File::open(file_path).unwrap();
  let metadata = file.metadata().unwrap();
  let file_size_bytes = metadata.len();
  println!("{} size: {}", file_path, format_file_size(file_size_bytes));
  file_size_bytes
}

pub fn format_file_size(bytes: u64) -> String {
  const KB: f64 = 1024.0;
  const MB: f64 = KB * 1024.0;
  const GB: f64 = MB * 1024.0;

  if bytes as f64 >= GB {
    format!("{:.2} GB", bytes as f64 / GB)
  } else if bytes as f64 >= MB {
    format!("{:.2} MB", bytes as f64 / MB)
  } else if bytes as f64 >= KB {
    format!("{:.2} KB", bytes as f64 / KB)
  } else {
    format!("{} bytes", bytes)
  }
}

pub fn hash_str(s: &str) -> String {
  let mut hasher = DefaultHasher::new();
  s.hash(&mut hasher);
  let hash_value = hasher.finish();
  hash_value.to_string()
}

pub fn file_exists(path: &str) -> bool {
  fs::metadata(path).is_ok()
}
