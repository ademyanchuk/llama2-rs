use candle_core::{DType, Error, Result, Tensor, D};
use rand::{distributions::Distribution, SeedableRng};

/// Logits Sampler to use on output returned by transformer
/// provides simple max prob sampling, temperature sampling
/// and "nucleus" (aka topp) sampling
/// adopted from (https://github.com/huggingface/candle/blob/main/candle-transformers/src/generation/mod.rs#L4)
pub struct LogitsSampler {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    topp: Option<f64>,
}

impl LogitsSampler {
    pub fn new(seed: u64, temperature: Option<f64>, topp: Option<f64>) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            topp,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let temperature = self.temperature.unwrap_or(0.);
        // sampling strategies temperature and topp
        let next_token = if temperature > 0. {
            let prs = candle_nn::ops::softmax(&(&logits / temperature)?, D::Minus1)?;
            let prs: Vec<f32> = prs.to_vec1()?;
            if let Some(topp) = self.topp {
                sample_topp(prs, topp, &mut self.rng) // sampling with topp filtering
            } else {
                let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
                distr.sample(&mut self.rng) as u32 // sampling from the full probabilities vector
            }
        } else {
            // return token id with max probability, no sampling
            let logits_v: Vec<f32> = logits.to_vec1()?;
            logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        Ok(next_token)
    }
}

/// top-p sampling (or "nucleus sampling") samples from the smallest set of
/// tokens that exceed probability topp. This way we never sample tokens that
/// have very low probabilities and are less likely to go "off the rails".
/// from (https://github.com/karpathy/llama2.c/blob/master/run.c)
fn sample_topp(probs: Vec<f32>, topp: f64, rng: &mut rand::rngs::StdRng) -> u32 {
    assert!(probs.len() > 1);
    let cutoff = (1.0 - topp) / (probs.len() - 1) as f64;
    // enumerate to keep original indices, collect only probs >= cutoff
    let mut prob_index: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p as f64 >= cutoff)
        .map(|(idx, &p)| (idx, p))
        .collect();
    prob_index.sort_unstable_by(|(_, u), (_, v)| v.total_cmp(u)); // non-increasing order

    // build a prefix-sum vector with accumulated probs, then pick up to cumulative p <= topp
    let cumulative = prob_index.iter().scan(0.0f32, |acc, item| {
        *acc += item.1;
        Some(*acc)
    });

    let prob_index: Vec<_> = prob_index
        .iter()
        .zip(cumulative)
        .take_while(|&(_, cum_prob)| cum_prob as f64 <= topp)
        .map(|(item, _)| item)
        .collect();
    // now we can sample
    let weights: Vec<_> = prob_index.iter().map(|item| item.1).collect();
    let distr = rand::distributions::WeightedIndex::new(weights).unwrap();
    let idx = distr.sample(rng);
    prob_index[idx].0 as u32
}
