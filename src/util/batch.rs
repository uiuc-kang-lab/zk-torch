use crate::basic_block::BatchCounters;

pub fn get_batch_key(tag: &String, batch_no: usize) -> String {
  format!("{}_{}", tag, batch_no)
}

pub fn update_batch_counters(batch_counters: &mut BatchCounters, tag: &String, batch_limit: usize) -> String {
  let mut binding = batch_counters.borrow_mut();
  let (batch_no, _) = binding
    .entry(tag.to_string())
    .and_modify(|value| {
      let (no, ct) = value;
      *value = if *ct < batch_limit { (*no, *ct + 1) } else { (*no + 1, 1) }
    })
    .or_insert((0, 1));
  get_batch_key(tag, *batch_no)
}
