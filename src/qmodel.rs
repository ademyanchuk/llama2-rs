use candle_core::quantized::k_quants::BlockQ8_0;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::{silu, softmax};
use candle_nn::{
    embedding, linear_no_bias as linear, rms_norm, Embedding, Module, RmsNorm, VarBuilder,
};
use std::collections::HashMap;
use std::io::{self, SeekFrom};

use crate::cnd_weights::{read_i32, read_tensor};
use crate::model::ModelArgsBuilder;
use crate::{
    cnd_model::{apply_rotary_emb, repeat_kv},
    model::ModelArgs,
};

/// Quantized version of the model architecture
/// it supposed to be loaded from the version 1 of Karapathy's export
/// and convert to QTensor appropriate weights
/// version1_export here https://github.com/karpathy/llama2.c/blob/master/export.py

// Model args from reader, V1 (llama2.c repo)
impl ModelArgs {
    pub fn from_reader_v1<R: io::Read + io::Seek>(r: &mut R) -> anyhow::Result<ModelArgs> {
        // 1. Check magic: should be uint32 of "ak42" in ASCII
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf);
        if magic != 0x616b3432 {
            anyhow::bail!("magic doesn't match!");
        }
        // 2. read version, must be version 1
        let version = read_i32(r)?;
        if version != 1 {
            anyhow::bail!("export file version must be 1");
        }
        // 3. read model arguments
        let dim = read_i32(r)? as usize;
        let hidden_dim = read_i32(r)? as usize;
        let n_layers = read_i32(r)? as usize;
        let n_heads = read_i32(r)? as usize;
        let n_kv_heads = read_i32(r)? as usize;
        let vocab_size = read_i32(r)? as usize;
        let max_seq_len = read_i32(r)? as usize;
        // 4. read shared classifier 'B' - 1 byte
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf)?;
        let shared_classifier = buf[0] != 0;
        // 4. we know the size of header is 256 bytes (spec of llama2.c v2 export)
        // need to move reader ahead to read weights later
        r.seek(SeekFrom::Start(256))?;
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

// TODO: Implement reading from v1 version of Karpathy's export
// this will allow to use dummy model for tests
pub struct TransformerWeights {
    q8_weights: HashMap<String, QTensor>,
    f32_weights: HashMap<String, Tensor>,
}
impl TransformerWeights {
    pub fn from_reader<R: io::Read>(
        r: &mut R,
        args: &ModelArgs,
        device: &Device,
    ) -> anyhow::Result<TransformerWeights> {
        // initialize hashmaps
        let mut q8_weights: HashMap<String, QTensor> = HashMap::new();
        let mut f32_weights: HashMap<String, Tensor> = HashMap::new();
        // all f32 weights were written first (norm layers attn, ffd, final norm)
        let attn_norm = read_tensor(r, (args.n_layers, args.dim), device)?;
        let ffd_norm = read_tensor(r, (args.n_layers, args.dim), device)?;
        for layer in 0..args.n_layers {
            f32_weights.insert(format!("layers.{layer}.attn_norm"), attn_norm.i(layer)?);
            f32_weights.insert(format!("layers.{layer}.ffd_norm"), ffd_norm.i(layer)?);
        }
        let final_norm = read_tensor(r, args.dim, device)?;
        f32_weights.insert("final_norm".to_string(), final_norm);
        // next embeddings
        let embed = read_tensor(r, (args.vocab_size, args.dim), device)?;

        // linear layers which supposed to be quantized
        // careful with wk and wv, in case n_kv_heads != n_heads we need right shape
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let head_dim = args.dim / n_heads; // note: dim = head_dim * n_heads
                                           // read in the rest of blocks' weights
                                           // attention weights first
        let wq = read_tensor(r, (args.n_layers, args.dim, args.dim), device)?; // assume? out_dim, in_dim as in linear layers for now
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.wq"),
                QTensor::quantize::<BlockQ8_0>(&wq.i(layer)?)?,
            );
        }
        let wk = read_tensor(r, (args.n_layers, n_kv_heads * head_dim, args.dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.wk"),
                QTensor::quantize::<BlockQ8_0>(&wk.i(layer)?)?,
            );
        }
        let wv = read_tensor(r, (args.n_layers, n_kv_heads * head_dim, args.dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.wv"),
                QTensor::quantize::<BlockQ8_0>(&wv.i(layer)?)?,
            );
        }
        let wo = read_tensor(r, (args.n_layers, args.dim, args.dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.wo"),
                QTensor::quantize::<BlockQ8_0>(&wo.i(layer)?)?,
            );
        }
        // feed forward weights
        let w1 = read_tensor(r, (args.n_layers, args.hidden_dim, args.dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.w1"),
                QTensor::quantize::<BlockQ8_0>(&w1.i(layer)?)?,
            );
        }
        let w2 = read_tensor(r, (args.n_layers, args.dim, args.hidden_dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.w2"),
                QTensor::quantize::<BlockQ8_0>(&w2.i(layer)?)?,
            );
        }
        let w3 = read_tensor(r, (args.n_layers, args.hidden_dim, args.dim), device)?;
        for layer in 0..args.n_layers {
            q8_weights.insert(
                format!("layers.{layer}.w3"),
                QTensor::quantize::<BlockQ8_0>(&w3.i(layer)?)?,
            );
        }
        // check if not shared classifier and read head of not
        let lm_head: Tensor = if !args.shared_classifier {
            read_tensor(r, (args.vocab_size, args.dim), device)?
        } else {
            embed.clone()
        };
        // postpone inserting in case we need this tensor
        f32_weights.insert("embed".to_string(), embed);
        q8_weights.insert(
            "lm_head".to_string(),
            QTensor::quantize::<BlockQ8_0>(&lm_head)?,
        );

        anyhow::Ok(TransformerWeights {
            q8_weights,
            f32_weights,
        })
    }
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
        cache: &mut [Option<(Tensor, Tensor)>],
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
    use std::{env, fs::File, io::Seek};

    use super::*;
    use candle_core::Device;

    #[test]
    fn test_v1_args_reader() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        // args = ModelArgs(dim=64, n_layers=2, n_heads=2, vocab_size=128, multiple_of=64, max_seq_len=32)
        // shared classifier == true, hidden_dim == 192
        assert_eq!(args.dim, 64);
        assert_eq!(args.n_layers, 2);
        assert_eq!(args.n_heads, 2);
        assert_eq!(args.n_kv_heads, Some(2));
        assert_eq!(args.vocab_size, 128);
        assert_eq!(args.hidden_dim, 192);
        assert_eq!(args.max_seq_len, 32);
        assert!(args.shared_classifier);
        let current_position = file.seek(SeekFrom::Current(0))?;
        assert_eq!(current_position, 256);
        Ok(())
    }
    #[test]
    fn test_read_and_quantize_weights() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let _ = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        assert!(true);
        Ok(())
    }
}
