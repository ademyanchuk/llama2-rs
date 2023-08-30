use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::linear_no_bias as linear;
use candle_nn::ops::{silu, softmax};
use candle_nn::{Linear, Module, VarBuilder};

use crate::model::ModelArgs;
// Same llama.c version of Transformer,
// but built with HF candle (mostly copied from candle examples)
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

pub struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    mask: Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
}

impl Attention {
    pub fn from(vb: VarBuilder, args: &ModelArgs) -> Result<Self> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let n_rep = n_heads / n_kv_heads;
        let head_dim = args.dim / n_heads;
        let wq = linear(args.dim, n_heads * head_dim, vb.pp("wq"))?;
        let wk = linear(args.dim, n_kv_heads * head_dim, vb.pp("wk"))?;
        let wv = linear(args.dim, n_kv_heads * head_dim, vb.pp("wv"))?;
        let wo = linear(n_heads * head_dim, args.dim, vb.pp("wo"))?;
        let mask = triu_mask(args.max_seq_len, &vb.device().clone())?;
        Ok(Attention {
            wq,
            wk,
            wv,
            wo,
            mask,
            n_heads,
            n_kv_heads,
            n_rep,
            head_dim,
        })
    }
    pub fn forward(&self, x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        // QKV
        let xq = self.wq.forward(x)?;
        let xk = self.wk.forward(x)?;
        let xv = self.wv.forward(x)?;

        let xq = xq.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?;
        let xk = xk.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;
        let xv = xv.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;

        // RoPE relative positional embeddings
        let xq = apply_rotary_emb(&xq, freqs_cos, freqs_sin)?;
        let xk = apply_rotary_emb(&xk, freqs_cos, freqs_sin)?;

        // grouped multiquery attention: expand out keys and values
        let xk = repeat_kv(xk, self.n_rep)?; // (bs, seqlen, n_heads, head_dim)
        let xv = repeat_kv(xv, self.n_rep)?;

        // make heads into a batch dimension
        let xq = xq.transpose(1, 2)?.contiguous()?; // (bs, n_heads, seqlen, head_dim)
        let xk = xk.transpose(1, 2)?.contiguous()?;
        let xv = xv.transpose(1, 2)?.contiguous()?;

        let scores = (xq.matmul(&xk.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = scores.broadcast_add(&self.mask.i((.., .., ..seq_len, ..seq_len))?)?;
        let scores = softmax(&scores, D::Minus1)?;
        // out will be (bs, n_heads, seqlen, head_dim)
        let out = scores.matmul(&xv.contiguous()?)?;
        // restore time as batch dimension and concat heads
        let out = out.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&out)
    }
}

// Functions
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // candle implementation requires cos and sin to have 3 dims, last one is 1
    let (b_sz, seq_len, h, n_embd) = x.dims4()?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let cos = cos.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, seq_len, 1, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, seq_len, h, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let dst0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let dst1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[&dst0, &dst1], D::Minus1)?.reshape((b_sz, seq_len, h, n_embd))?;
    Ok(rope)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, seq_len, n_kv_head, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(3)?
            .expand((b_sz, seq_len, n_kv_head, n_rep, head_dim))?
            .reshape((b_sz, seq_len, n_kv_head * n_rep, head_dim))?;
        Ok(x)
    }
}

fn triu_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; t]; t];
    for (i, row) in mask.iter_mut().enumerate().take(t) {
        for val in row.iter_mut().take(i+1) {
            *val = 0.0;
        }
    }
    let mask = Tensor::from_iter(mask.into_iter().flatten(), device)?;
    mask.reshape((1, 1, t, t))
}

#[cfg(test)]
mod tests {
    use approx::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    use crate::{cnd_model::*, model::ModelArgsBuilder, test_data::*};

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

    fn approx_eq_vec(a: &Vec<f32>, b: &Vec<f32>, eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (val_a, val_b) in a.iter().zip(b.iter()) {
            if !abs_diff_eq!(val_a, val_b, epsilon = eps) {
                return false;
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
    #[test]
    fn test_attention() -> Result<()> {
        let device = &Device::Cpu;
        let args = ModelArgs::new(8, 12, 2, None, 256, 256, 1e-4, 32);
        // need it here to build appropriate tensors for Linear layers
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let head_dim = args.dim / n_heads;
        // init attention
        let mut weights_data = HashMap::new();
        let wq = Tensor::from_vec(ATT_WQ.to_vec(), (n_heads * head_dim, args.dim), device)?;
        let wk = Tensor::from_vec(ATT_WK.to_vec(), (n_kv_heads * head_dim, args.dim), device)?;
        let wv = Tensor::from_vec(ATT_WV.to_vec(), (n_kv_heads * head_dim, args.dim), device)?;
        let wo = Tensor::from_vec(ATT_WO.to_vec(), (args.dim, n_heads * head_dim), device)?;
        weights_data.insert("wq.weight".to_string(), wq);
        weights_data.insert("wk.weight".to_string(), wk);
        weights_data.insert("wv.weight".to_string(), wv);
        weights_data.insert("wo.weight".to_string(), wo);
        let vb = VarBuilder::from_tensors(weights_data, DType::F32, device);
        let att = Attention::from(vb, &args)?;

        // init inputs
        let seq_len = 4_usize;
        let inp = Tensor::from_slice(ATT_INP_FLAT, (2, seq_len, args.dim), device)?;
        let freqs_cos = Tensor::from_slice(
            ATT_FREQ_COS_FLAT,
            (args.max_seq_len, args.dim / args.n_heads / 2, 1),
            device,
        )?;
        let freqs_cos = freqs_cos.i((..seq_len, .., ..))?;
        let freqs_sin = Tensor::from_slice(
            ATT_FREQ_SIN_FLAT,
            (args.max_seq_len, args.dim / args.n_heads / 2, 1),
            device,
        )?;
        let freqs_sin = freqs_sin.i((..seq_len, .., ..))?;
        // check results
        let result = att.forward(&inp, &freqs_cos, &freqs_sin)?;
        assert_eq!(result.dims3()?, (2, seq_len, args.dim));
        assert!(approx_eq_vec(
            &result.flatten_all()?.to_vec1()?,
            &ATT_OUT_FLAT.to_vec(),
            1e-3
        ));
        Ok(())
    }
    #[test]
    fn test_apply_rotary_emb() -> Result<()> {
        let dev = &Device::Cpu;
        let shape = (2, 3, 2, 4);
        let xq = Tensor::from_slice(XQ_DATA, shape, dev)?;
        let xk = Tensor::from_slice(XK_DATA, shape, dev)?;
        let freq_cos = Tensor::from_vec(
            vec![-0.1339f32, -1.4408, -0.7710, 0.4526, -3.0065, 2.3243],
            (3, 2, 1),
            dev,
        )?;
        let freqs_sin = Tensor::from_vec(
            vec![0.3798f32, -1.3930, -0.0854, 0.7161, 2.4592, -1.0601],
            (3, 2, 1),
            dev,
        )?;
        let xq_out = apply_rotary_emb(&xq, &freq_cos, &freqs_sin)?;
        assert_eq!(xq_out.dims4()?, shape);
        assert!(approx_eq_vec(
            &xq_out.flatten_all()?.to_vec1()?,
            &XQ_OUT_EXP_DATA.to_vec(),
            1e-3
        ));
        let xk_out = apply_rotary_emb(&xk, &freq_cos, &freqs_sin)?;
        assert_eq!(xq_out.dims4()?, shape);
        assert!(approx_eq_vec(
            &xk_out.flatten_all()?.to_vec1()?,
            &XK_OUT_EXP_DATA.to_vec(),
            1e-3
        ));
        Ok(())
    }
    #[test]
    fn test_repeat_kv() -> Result<()> {
        let x = Tensor::from_slice(REP_KV_INP, (2, 3, 2, 3), &Device::Cpu)?;
        let repeated = repeat_kv(x, 2)?;
        assert_eq!(repeated.dims4()?, (2, 3, 4, 3));
        assert_eq!(
            &repeated.flatten_all()?.to_vec1::<f32>()?,
            &REP_KV_OUT.to_vec()
        );
        Ok(())
    }
    #[test]
    fn test_triu_mask() -> Result<()> {
        let mask = triu_mask(3, &Device::Cpu)?;
        assert_eq!(mask.dims4()?, (1, 1, 3, 3));
        assert_eq!(
            mask.flatten_all()?.to_vec1::<f32>()?,
            vec![
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
        );
        Ok(())
    }
}
