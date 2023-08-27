// Same llama.c version of Transformer, but built with HF candle
use candle_core::{Device, Result, Tensor};
use candle_nn::{ops::silu, Linear, Module};

use crate::F32VecMap;
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        mut weights: F32VecMap,
        device: &Device,
    ) -> Result<FeedForward> {
        let w1 = Tensor::from_vec(
            weights.remove("w1").expect("w1 is expected"),
            (hidden_dim, in_dim),
            device,
        )?;
        let w1 = Linear::new(w1, None);
        let w2 = Tensor::from_vec(
            weights.remove("w2").expect("w2 is expected"),
            (in_dim, hidden_dim),
            device,
        )?;
        let w2 = Linear::new(w2, None);
        let w3 = Tensor::from_vec(
            weights.remove("w3").expect("w3 is expected"),
            (hidden_dim, in_dim),
            device,
        )?;
        let w3 = Linear::new(w3, None);
        Ok(FeedForward { w1, w2, w3 })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?;
        self.w2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use approx::*;
    use std::collections::HashMap;

    use crate::cnd_model::*;

    fn approx_eq_nested_vec(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, epsilon: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }

        for (sub_a, sub_b) in a.iter().zip(b.iter()) {
            if sub_a.len() != sub_b.len() {
                return false;
            }

            for (val_a, val_b) in sub_a.iter().zip(sub_b.iter()) {
                if !abs_diff_eq!(val_a, val_b, epsilon = epsilon) {
                    return false;
                }
            }
        }

        true
    }

    #[test]
    fn test_feed_forward() -> Result<()> {
        let d = &Device::Cpu;
        // init FeedForward block
        let mut weights: F32VecMap = HashMap::new();
        weights.insert(
            "w1",
            vec![
                -0.4764f32, 0.0453, 0.1539, 0.0752, -0.2386, 0.3814, 0.1513, -0.3863, 0.4562, 0.2769,
                0.4002, -0.1647,
            ],
        );
        weights.insert(
            "w2",
            vec![
                0.2170f32, 0.3585, -0.2992, 0.3554, -0.4850, 0.2447, 0.1820, 0.2602, 0.0146, 0.1802,
                -0.4978, -0.0919,
            ],
        );
        weights.insert(
            "w3",
            vec![
                -0.4622f32, -0.5098, 0.4391, 0.4349, -0.4857, 0.3582, 0.2414, 0.3671, 0.2596, 0.2129,
                0.0142, 0.1426,
            ],
        );
        let ff = FeedForward::new(3, 4, weights, d)?;
        // predefined input
        let x = Tensor::from_vec(
            vec![0.0874f32, -0.7098, -1.6503, -0.5212, -1.3321, -0.5542],
            (2, 3),
            d,
        )?;
        // get result and compare with predefined out
        let y = ff.forward(&x)?;
        let expect = Tensor::from_vec(
            vec![-0.0112f32, 0.0036, -0.0521, 0.0489, -0.0182, 0.0356],
            (2, 3),
            d,
        )?;
        assert!(approx_eq_nested_vec(&y.to_vec2()?, &expect.to_vec2()?, 1e-4));
        Ok(())
    }
}
