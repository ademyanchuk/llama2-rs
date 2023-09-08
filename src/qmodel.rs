use std::collections::HashMap;

/// Quantized version of the model architecture
/// it supposed to be loaded from the version 2 of Karapathy's export
/// version2_export here https://github.com/karpathy/llama2.c/blob/master/export.py
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::ops::{silu, softmax};
use candle_nn::{
    embedding, linear_no_bias as linear, rms_norm, Embedding, Module, RmsNorm, VarBuilder,
};

use crate::{
    cnd_model::{apply_rotary_emb, repeat_kv},
    model::ModelArgs,
};

// TODO: Implement reading from v2 version of Karpathy's export
// this will allow to use dummy model for tests
pub struct TransformerWeights {
    q8_weights: HashMap<String, QTensor>,
    f32_weights: HashMap<String, Tensor>,
}
impl TransformerWeights {
    pub fn remove_q8(&mut self, name: &str) -> Result<QTensor> {
        match self.q8_weights.remove(name) {
            None => candle_core::bail!("cannot find tensor with name '{name}'"),
            Some(weight) => Ok(weight),
        }
    }
    pub fn remove_f32(&mut self, name: &str) -> Result<Tensor> {
        match self.f32_weights.remove(name) {
            None => candle_core::bail!("cannot find tensor with name '{name}'"),
            Some(weight) => Ok(weight),
        }
    }
}

// Blocks
pub struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    n_heads: usize,
    n_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
}

impl Attention {
    pub fn from(
        weights: &mut TransformerWeights,
        args: &ModelArgs,
        block_id: usize,
    ) -> Result<Self> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let n_rep = n_heads / n_kv_heads;
        let head_dim = args.dim / n_heads;
        let wq = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.wq"))?);
        let wk = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.wk"))?);
        let wv = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.wv"))?);
        let wo = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.wo"))?);
        Ok(Attention {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            n_rep,
            head_dim,
        })
    }
    pub fn forward(
        &self,
        x: &Tensor,
        freqs_cos: &Tensor,
        freqs_sin: &Tensor,
        block_idx: usize,
        cache: &mut Vec<Option<(Tensor, Tensor)>>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        // QKV
        let xq = self.wq.forward(x)?;
        let xk = self.wk.forward(x)?;
        let xv = self.wv.forward(x)?;

        let xq = xq.reshape((b_sz, seq_len, self.n_heads, self.head_dim))?;
        let xk = xk.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;
        let mut xv = xv.reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?;

        // RoPE relative positional embeddings
        let xq = apply_rotary_emb(&xq, freqs_cos, freqs_sin)?;
        let mut xk = apply_rotary_emb(&xk, freqs_cos, freqs_sin)?;

        if let Some((cache_k, cache_v)) = &cache[block_idx] {
            xk = Tensor::cat(&[cache_k, &xk], 1)?.contiguous()?;
            xv = Tensor::cat(&[cache_v, &xv], 1)?.contiguous()?;
        }
        cache[block_idx] = Some((xk.clone(), xv.clone()));

        // grouped multiquery attention: expand out keys and values
        let xk = repeat_kv(xk, self.n_rep)?; // (bs, seq_len+cache_len, n_heads, head_dim)
        let xv = repeat_kv(xv, self.n_rep)?;

        // make heads into a batch dimension
        let xq = xq.transpose(1, 2)?.contiguous()?; // (bs, n_heads, seq_len, head_dim)
        let xk = xk.transpose(1, 2)?.contiguous()?; // (bs, n_heads, seq_len+cache_len, head_dim)
        let xv = xv.transpose(1, 2)?.contiguous()?; // (bs, n_heads, seq_len+cache_len, head_dim)

        // (bs, n_heads, seq_len, seq_len+cache_len)
        let scores = (xq.matmul(&xk.t()?)? / (self.head_dim as f64).sqrt())?;

        // we don't need to use mask for inference, because
        // 1. during prompt pass we have seq_len > 1, but we only care about last token, and it can attend everything before
        // 2. during next steps, we iterate token by token and given some pos_idx and seq_len == 1, we will index into pos_idx
        // row of the mask and take up to pos_idx + 1 columns from this row, effectively making the commented op identity op.
        // let scores = scores.broadcast_add(&self.mask.i((.., .., ..seq_len, ..seq_len))?)?;
        let scores = softmax(&scores, D::Minus1)?;
        // out will be (bs, n_heads, seq_len, head_dim)
        let out = scores.matmul(&xv.contiguous()?)?;
        // restore time as batch dimension and concat heads
        let out = out.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::*;
    use candle_core::{quantized::k_quants::BlockQ8_0, Device};

    #[test]
    fn test_attention() -> Result<()> {
        let device = &Device::Cpu;
        let args = ModelArgs::new(8, 12, 2, None, 256, 256, 1e-4, 32, true);
        // need it here to build appropriate tensors for Linear layers
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let head_dim = args.dim / n_heads;
        let wq = Tensor::from_vec(ATT_WQ.to_vec(), (n_heads * head_dim, args.dim), device)?;
        let wq = QTensor::quantize::<BlockQ8_0>(&wq)?;
        let wk = Tensor::from_vec(ATT_WK.to_vec(), (n_kv_heads * head_dim, args.dim), device)?;
        let wk = QTensor::quantize::<BlockQ8_0>(&wk)?;
        let wv = Tensor::from_vec(ATT_WV.to_vec(), (n_kv_heads * head_dim, args.dim), device)?;
        let wv = QTensor::quantize::<BlockQ8_0>(&wv)?;
        let wo = Tensor::from_vec(ATT_WO.to_vec(), (args.dim, n_heads * head_dim), device)?;
        let wo = QTensor::quantize::<BlockQ8_0>(&wo)?;
        let mut q8_weights = HashMap::new();
        q8_weights.insert("layers.0.wq".to_string(), wq);
        q8_weights.insert("layers.0.wk".to_string(), wk);
        q8_weights.insert("layers.0.wv".to_string(), wv);
        q8_weights.insert("layers.0.wo".to_string(), wo);
        let mut weights = TransformerWeights {q8_weights, f32_weights: HashMap::new()};
        let att = Attention::from(&mut weights, &args, 0)?;
        Ok(())
    }
}
