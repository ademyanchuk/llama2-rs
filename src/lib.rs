pub mod cnd_model;
pub mod cnd_weights;
#[allow(dead_code)]
pub mod model;
pub mod qmodel;
pub mod sampler;
#[allow(dead_code)]
#[allow(clippy::approx_constant)]
mod test_data;

use std::collections::HashMap;
pub type F32VecMap<'a> = HashMap<&'a str, Vec<f32>>;
