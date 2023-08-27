pub mod cnd_model;
#[allow(dead_code)]
pub mod model;
#[allow(dead_code)]
#[allow(clippy::approx_constant)]
mod test_data;
pub mod tokenizer;

use std::collections::HashMap;
pub type F32VecMap<'a> = HashMap<&'a str, Vec<f32>>;
