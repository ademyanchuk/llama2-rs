// I reuse a lot of conveniences from candle llama2 example
// Here is everything about reading from .bin Karpathy's exports
use anyhow::Result; // some functions mix tensor and reading potential errors, that;s why anyhow
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::model::{ModelArgs, ModelArgsBuilder};

pub(crate) fn read_i32<R: std::io::Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_tensor<R: std::io::Read, S: Into<Shape>>(
    r: &mut R,
    shape: S,
    dev: &Device,
) -> Result<Tensor> {
    let shape = shape.into();
    let mut data_t = vec![0f32; shape.elem_count()];
    r.read_f32_into::<LittleEndian>(&mut data_t)?;
    let tensor = Tensor::from_vec(data_t, shape, dev)?;
    Ok(tensor)
}

// ModelArgs from reader
impl ModelArgs {
    pub fn from_reader<R: std::io::Read>(r: &mut R) -> Result<ModelArgs> {
        // for the legacy export of models, we need to check a sign of vocab size
        // see here https://github.com/karpathy/llama2.c/blob/master/export.py
        let mut shared_classifier = true;
        let dim = read_i32(r)? as usize;
        let hidden_dim = read_i32(r)? as usize;
        let n_layers = read_i32(r)? as usize;
        let n_heads = read_i32(r)? as usize;
        let n_kv_heads = read_i32(r)? as usize;
        let mut vocab_size = read_i32(r)?;
        if vocab_size < 0 {
            shared_classifier = false;
            vocab_size = -vocab_size;
        }
        let vocab_size = vocab_size as usize;
        let max_seq_len = read_i32(r)? as usize;
        // Build args
        Ok(ModelArgsBuilder::new()
            .dim(dim)
            .hidden_dim(hidden_dim)
            .n_layers(n_layers)
            .n_heads(n_heads)
            .n_kv_heads(n_kv_heads)
            .vocab_size(vocab_size)
            .max_seq_len(max_seq_len)
            .shared_classifier(shared_classifier)
            .build())
    }
}

// Weights, their loading from reader, and producing VarBuilder
// Currently only legacy version of .bin exports
pub struct TransformerWeights {
    // token embedding table
    embed: Tensor, // (vocab_size, dim)
    // weights for rmsnorms
    attn_norm: Tensor, // (layer, dim) rmsnorm weights
    ffd_norm: Tensor,  // (layer, dim)
    // weights for matmuls
    wq: Tensor, // (layer, dim, dim) <=> (layer, out_dim, in_dim)
    wk: Tensor, // (layer, n_kv_heads * head_dim, dim) IMPORTANT if n_kv_heads != n_heads
    wv: Tensor, // (layer, n_kv_heads * head_dim, dim)
    wo: Tensor, // (layer, dim, dim)
    // weights for ffd
    w1: Tensor, // (layer, hidden_dim, dim)
    w2: Tensor, // (layer, dim, hidden_dim)
    w3: Tensor, // (layer, hidden_dim, dim)
    // final rmsnorm
    final_norm: Tensor, // (dim,)
    lm_head: Tensor,    // (vocab_size, dim)
    // freq_cis for RoPE relatively positional embeddings
    freqs_cos: Tensor, // (seq_len, head_size/2)
    freqs_sin: Tensor, // (seq_len, head_size/2)
}

impl TransformerWeights {
    pub fn from_reader<R: std::io::Read>(
        r: &mut R,
        c: &ModelArgs,
        dev: &Device,
    ) -> Result<TransformerWeights> {
        let embed = read_tensor(r, (c.vocab_size, c.dim), dev)?;
        let attn_norm = read_tensor(r, (c.n_layers, c.dim), dev)?;
        // careful with wk and wv, in case n_kv_heads != n_heads we need right shape
        let n_heads = c.n_heads;
        let n_kv_heads = c.n_kv_heads.unwrap_or(n_heads);
        let head_dim = c.dim / n_heads; // note: dim = head_dim * n_heads
                                        // read in the rest of blocks' weights
        let wq = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let wk = read_tensor(r, (c.n_layers, n_kv_heads * head_dim, c.dim), dev)?;
        let wv = read_tensor(r, (c.n_layers, n_kv_heads * head_dim, c.dim), dev)?;
        let wo = read_tensor(r, (c.n_layers, c.dim, c.dim), dev)?;
        let ffd_norm = read_tensor(r, (c.n_layers, c.dim), dev)?;
        let w1 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let w2 = read_tensor(r, (c.n_layers, c.dim, c.hidden_dim), dev)?;
        let w3 = read_tensor(r, (c.n_layers, c.hidden_dim, c.dim), dev)?;
        let final_norm = read_tensor(r, c.dim, dev)?;
        let freqs_cos = read_tensor(r, (c.max_seq_len, head_dim / 2), dev)?;
        let freqs_sin = read_tensor(r, (c.max_seq_len, head_dim / 2), dev)?;
        let lm_head: Tensor = if !c.shared_classifier {
            read_tensor(r, (c.vocab_size, c.dim), dev)?
        } else {
            embed.clone()
        };
        Ok(TransformerWeights {
            embed,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffd_norm,
            w1,
            w2,
            w3,
            final_norm,
            lm_head,
            freqs_cos,
            freqs_sin,
        })
    }
    // copied from candle examples to reuse VarBuilder in model loading
    pub fn var_builder(
        &self,
        cfg: &ModelArgs,
        device: &Device,
    ) -> candle_core::error::Result<VarBuilder<'static>> {
        // TODO: As of 2023-08-04, gemm is slower than expected when multiplying a matrix of
        // size (1, k) with the transpose of a matrix of size (k, n) as it ends up transposing the
        // second matrix back. We detect this case here and as a temporary hack make the weight
        // matrix column major rather than row major. This ends up speeding up text generation from
        // 120 token/s to 220 token/s on a Ryzen 2600X.
        let tr = device.is_cpu() && !candle_core::utils::has_mkl();
        let tr = |x: Tensor| if tr { x.t()?.contiguous()?.t() } else { Ok(x) };
        let mut ws = HashMap::new();
        let mut insert = |name: &str, t: Tensor| {
            ws.insert(name.to_string(), t);
        };
        insert("freqs_cos", self.freqs_cos.clone());
        insert("freqs_sin", self.freqs_sin.clone());
        insert("embed.weight", self.embed.clone());
        insert("lm_head.weight", tr(self.lm_head.clone())?);
        insert("final_norm.weight", self.final_norm.clone());
        for layer in 0..cfg.n_layers {
            ws.insert(
                format!("layers.{layer}.attn.wq.weight"),
                tr(self.wq.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.attn.wk.weight"),
                tr(self.wk.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.attn.wv.weight"),
                tr(self.wv.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.attn.wo.weight"),
                tr(self.wo.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.ffd.w1.weight"),
                tr(self.w1.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.ffd.w2.weight"),
                tr(self.w2.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.ffd.w3.weight"),
                tr(self.w3.i(layer)?)?,
            );
            ws.insert(
                format!("layers.{layer}.attn_norm.weight"),
                self.attn_norm.i(layer)?,
            );
            ws.insert(
                format!("layers.{layer}.ffd_norm.weight"),
                self.ffd_norm.i(layer)?,
            );
        }
        let vb = VarBuilder::from_tensors(ws, DType::F32, device);
        Ok(vb)
    }
}
