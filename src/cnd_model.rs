// Same llama.c version of Transformer, but built with HF candle
use candle_core::{Result, Tensor};
use candle_nn::linear_no_bias as linear;
use candle_nn::{ops::silu, Linear, Module, VarBuilder};

use crate::model::ModelArgs;
//Blocks
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn new(w1: Linear, w2: Linear, w3: Linear) -> FeedForward {
        FeedForward { w1, w2, w3 }
    }
    pub fn from(vb: VarBuilder, args: &ModelArgs) -> Result<FeedForward> {
        let (in_dim, hidden_dim) = (args.dim, args.hidden_dim);
        let w1 = linear(in_dim, hidden_dim, vb.pp("w1"))?; // in_dim, out_dim as expected, but this function wants a tensor(out_dim, in_dim)
        let w2 = linear(hidden_dim, in_dim, vb.pp("w2"))?;
        let w3 = linear(in_dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self::new(w1, w2, w3))
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?;
        self.w2.forward(&x)
    }
}

// Functions

#[cfg(test)]
mod tests {
    use approx::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    use crate::{cnd_model::*, model::ModelArgsBuilder};

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
        let args = ModelArgsBuilder::new().dim(3).hidden_dim(4).build();
        let d = &Device::Cpu;
        // init FeedForward block
        let mut weights = HashMap::new();
        weights.insert(
            "w1.weight".to_string(),
            Tensor::from_vec(
                vec![
                    -0.4764f32, 0.0453, 0.1539, 0.0752, -0.2386, 0.3814, 0.1513, -0.3863, 0.4562,
                    0.2769, 0.4002, -0.1647,
                ],
                (args.hidden_dim, args.dim),
                d,
            )?,
        );
        weights.insert(
            "w2.weight".to_string(),
            Tensor::from_vec(
                vec![
                    0.2170f32, 0.3585, -0.2992, 0.3554, -0.4850, 0.2447, 0.1820, 0.2602, 0.0146,
                    0.1802, -0.4978, -0.0919,
                ],
                (args.dim, args.hidden_dim),
                d,
            )?,
        );
        weights.insert(
            "w3.weight".to_string(),
            Tensor::from_vec(
                vec![
                    -0.4622f32, -0.5098, 0.4391, 0.4349, -0.4857, 0.3582, 0.2414, 0.3671, 0.2596,
                    0.2129, 0.0142, 0.1426,
                ],
                (args.hidden_dim, args.dim),
                d,
            )?,
        );
        let vb = VarBuilder::from_tensors(weights, DType::F32, d);
        let ff = FeedForward::from(vb, &args)?;
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
        assert!(approx_eq_nested_vec(
            &y.to_vec2()?,
            &expect.to_vec2()?,
            1e-4
        ));
        Ok(())
    }
}
