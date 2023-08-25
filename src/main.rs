#[allow(dead_code)]
mod model;
#[allow(dead_code)]
#[allow(clippy::approx_constant)]
mod test_data;
mod tokenizer;

use anyhow::{Ok, Result};

use ndarray::{Array, Axis};
use rand::{rngs::StdRng, SeedableRng};

use model::Transformer;
use tokenizer::Tokenizer;

fn main() -> Result<()> {
    let path = "stories15M.bin";
    let start = "One day, Lily met a Shoggoth";
    let num_samples = 2;
    let max_new_tokens = 100; // number of tokens generated in each sample
    let temperature = 1.0; // 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    let top_k = 300; // retain only the top_k most likely tokens, clamp others to have 0 probability

    let seed = [13; 32];
    let mut rng = StdRng::from_seed(seed);

    let transformer = Transformer::from(path)?;
    let enc = Tokenizer::default();

    let start_ids = enc.encode(start, true, false);
    let start_ids: Vec<_> = start_ids.into_iter().map(|i| i as usize).collect();
    let x = Array::from(start_ids).insert_axis(Axis(0));
    for _ in 0..num_samples {
        let y = transformer.generate(
            &mut rng,
            x.clone(),
            max_new_tokens,
            temperature,
            Some(top_k),
        );
        // Ensure there's at least one row
        if let Some(first_row) = y.outer_iter().next() {
            let vec_u32: Vec<u32> = first_row.iter().map(|&x| x as u32).collect();
            println!("{}", enc.decode(&vec_u32));
            println!("---------------")
        } else {
            println!("The 'y' array is empty!");
        }
    }
    Ok(())
}
