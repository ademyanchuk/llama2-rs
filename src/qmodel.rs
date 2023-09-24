use anyhow::bail;
use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::k_quants::BlockQ8_0;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::{silu, softmax};
use candle_nn::{Embedding, LayerNorm, Module};
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

// Model args from reader, V1 and V3 (llama2.c repo)
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
        if ![1, 3].contains(&version) {
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
            .version(version)
            .build())
    }
}

// Weights Struct and reading from v1 version of Karpathy's export
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
        match args.version {
            1 => Self::from_reader_v1(r, args, device),
            3 => Self::from_reader_v3(r, args, device),
            _ => bail!("Trying to load quantized weights from unsupported export version, must be v1 or v3")
        }
    }
    fn from_reader_v1<R: io::Read>(
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

    fn from_reader_v3<R: io::Read>(
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

        // next goes quantized layers
        // we need to de-quantize embedding layer to use in the model!
        let embed = read_q80_tensor(r, vec![args.vocab_size, args.dim])?;

        // linear layers which supposed to be quantized
        // careful with wk and wv, in case n_kv_heads != n_heads we need right shape
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads.unwrap_or(n_heads);
        let head_dim = args.dim / n_heads; // note: dim = head_dim * n_heads
                                           // read in the rest of blocks' weights
                                           // attention weights first

        // assume? out_dim, in_dim as in linear layers for now
        for layer in 0..args.n_layers {
            let wq = read_q80_tensor(r, vec![args.dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.wq"), wq);
        }
        for layer in 0..args.n_layers {
            let wk = read_q80_tensor(r, vec![n_kv_heads * head_dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.wk"), wk);
        }
        for layer in 0..args.n_layers {
            let wv = read_q80_tensor(r, vec![n_kv_heads * head_dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.wv"), wv);
        }
        for layer in 0..args.n_layers {
            let wo = read_q80_tensor(r, vec![args.dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.wo"), wo);
        }
        // feed forward weights
        for layer in 0..args.n_layers {
            let w1 = read_q80_tensor(r, vec![args.hidden_dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.w1"), w1);
        }
        for layer in 0..args.n_layers {
            let w2 = read_q80_tensor(r, vec![args.dim, args.hidden_dim])?;
            q8_weights.insert(format!("layers.{layer}.w2"), w2);
        }
        for layer in 0..args.n_layers {
            let w3 = read_q80_tensor(r, vec![args.hidden_dim, args.dim])?;
            q8_weights.insert(format!("layers.{layer}.w3"), w3);
        }

        let embed_f32 = embed.dequantize(device)?;
        // check if not shared classifier and read head of not
        let lm_head: QTensor = if !args.shared_classifier {
            read_q80_tensor(r, vec![args.vocab_size, args.dim])?
        } else {
            embed
        };
        // postpone inserting in case we need this tensor
        f32_weights.insert("embed".to_string(), embed_f32);
        q8_weights.insert("lm_head".to_string(), lm_head);

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

fn read_q80_tensor<R: io::Read>(r: &mut R, dims: Vec<usize>) -> Result<QTensor> {
    let dtype = GgmlDType::Q8_0;
    let numel = dims.iter().product::<usize>();
    let size_in_bytes = numel * dtype.type_size() / dtype.blck_size();
    let mut raw_data = vec![0u8; size_in_bytes];
    r.read_exact(&mut raw_data)?;
    qtensor_from_ggml(dtype, &raw_data, dims)
}

// Model
pub struct Transformer {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    output: QMatMul,
    freqs_cos: Tensor,
    freqs_sin: Tensor,
    cache: Vec<Option<(Tensor, Tensor)>>,
}
impl Transformer {
    pub fn from(weights: &mut TransformerWeights, args: &ModelArgs) -> Result<Transformer> {
        let tok_embeddings = Embedding::new(weights.remove_f32("embed")?, args.dim);
        let output = QMatMul::from_qtensor(weights.remove_q8("lm_head")?);
        let norm = LayerNorm::rms_norm(weights.remove_f32("final_norm")?, args.norm_eps as f64);
        let layers: Vec<_> = (0..args.n_layers)
            .map(|i| TransformerBlock::from(weights, args, i).unwrap())
            .collect();
        let (freqs_cos, freqs_sin) =
            precompute_freqs_cis(args.dim / args.n_heads, args.max_seq_len, 10000.0)?;
        let freqs_cos = freqs_cos.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_sin = freqs_sin.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        Ok(Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
            freqs_cos,
            freqs_sin,
            cache: vec![None; args.n_layers],
        })
    }
    pub fn forward(&mut self, x: &Tensor, pos_idx: usize) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        let mut h = self.tok_embeddings.forward(x)?;
        // index into correct position of freqs here
        let freqs_cos = self.freqs_cos.i((pos_idx..pos_idx + seq_len, .., ..))?;
        let freqs_sin = self.freqs_sin.i((pos_idx..pos_idx + seq_len, .., ..))?;
        for (block_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &freqs_cos, &freqs_sin, block_idx, &mut self.cache)?;
        }
        h = self.norm.forward(&h)?;
        let logits = self.output.forward(&h)?;
        logits.to_dtype(DType::F32)
    }
}

// Blocks
pub struct TransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    attn_norm: LayerNorm,
    ffd_norm: LayerNorm,
}
impl TransformerBlock {
    pub fn from(
        weights: &mut TransformerWeights,
        args: &ModelArgs,
        block_id: usize,
    ) -> Result<TransformerBlock> {
        let attention = Attention::from(weights, args, block_id)?;
        let feed_forward = FeedForward::from(weights, block_id)?;
        let attn_norm = LayerNorm::rms_norm(
            weights.remove_f32(&format!("layers.{block_id}.attn_norm"))?,
            args.norm_eps as f64,
        );
        let ffd_norm = LayerNorm::rms_norm(
            weights.remove_f32(&format!("layers.{block_id}.ffd_norm"))?,
            args.norm_eps as f64,
        );
        Ok(TransformerBlock {
            attention,
            feed_forward,
            attn_norm,
            ffd_norm,
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
        let residual = x;
        let x = (residual
            + self.attention.forward(
                &self.attn_norm.forward(x)?,
                freqs_cos,
                freqs_sin,
                block_idx,
                cache,
            )?)?;
        let residual = &x;
        residual + self.feed_forward.forward(&self.ffd_norm.forward(&x)?)
    }
}
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

pub struct FeedForward {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl FeedForward {
    pub fn new(w1: QMatMul, w2: QMatMul, w3: QMatMul) -> FeedForward {
        FeedForward { w1, w2, w3 }
    }
    pub fn from(weights: &mut TransformerWeights, block_id: usize) -> Result<FeedForward> {
        let w1 = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.w1"))?);
        let w2 = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.w2"))?);
        let w3 = QMatMul::from_qtensor(weights.remove_q8(&format!("layers.{block_id}.w3"))?);
        Ok(Self::new(w1, w2, w3))
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?;
        self.w2.forward(&x)
    }
}
// Functions
fn precompute_freqs_cis(
    head_dim: usize,
    max_seq_len: usize,
    freq_base: f32,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), &Device::Cpu)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use std::{env, fs::File, io::Seek};

    use crate::test_data::{
        QATT_INP, QATT_OUT, QBLOCK_OUT, QFFD_IN, QFFD_OUT, QMOD_OUT, TB_FREQS_COS, TB_FREQS_SIN,
    };

    use super::*;
    use approx::abs_diff_eq;
    use candle_core::Device;

    fn approx_eq_vec(a: &Vec<f32>, b: &Vec<f32>, eps: f32) -> bool {
        println!("{a:?}\n{b:?}");
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
    #[test]
    fn test_attention() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let seq_len = 4;
        let device = &Device::Cpu;
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let attn = Attention::from(&mut weights, &args, 0)?;

        let (freqs_cos, freqs_sin) =
            precompute_freqs_cis(args.dim / args.n_heads, args.max_seq_len, 10000.0)?;
        let freqs_cos = freqs_cos.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_sin = freqs_sin.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_cos = freqs_cos.i((..seq_len, .., ..))?;
        let freqs_sin = freqs_sin.i((..seq_len, .., ..))?;

        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None];
        let input = Tensor::from_slice(QATT_INP, (2, seq_len, args.dim), device)?;

        let output = attn.forward(&input, &freqs_cos, &freqs_sin, 0, &mut cache)?;
        assert_eq!(output.dims3()?, (2, seq_len, args.dim));
        let last_step = output.i((.., seq_len - 1, ..))?;
        let expect = Tensor::from_slice(QATT_OUT, (2, seq_len, args.dim), device)?;
        let expect = expect.i((.., seq_len - 1, ..))?;
        assert!(approx_eq_vec(
            &last_step.flatten_all()?.to_vec1()?,
            &expect.flatten_all()?.to_vec1()?,
            1e-1 // we compare the result from quantized model with reference from f32 model
        ));

        Ok(())
    }
    #[test]
    fn test_attention_with_cache() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let seq_len = 4;
        let device = &Device::Cpu;
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let attn = Attention::from(&mut weights, &args, 0)?;

        let (freqs_cos, freqs_sin) =
            precompute_freqs_cis(args.dim / args.n_heads, args.max_seq_len, 10000.0)?;
        let freqs_cos = freqs_cos.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_sin = freqs_sin.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_cos = freqs_cos.i((..seq_len, .., ..))?;
        let freqs_sin = freqs_sin.i((..seq_len, .., ..))?;

        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None];
        let input = Tensor::from_slice(QATT_INP, (2, seq_len, args.dim), device)?;
        let expect = Tensor::from_slice(QATT_OUT, (2, seq_len, args.dim), device)?;
        for step in 0..seq_len {
            let out = attn.forward(
                &input.i((.., step..step + 1, ..))?.contiguous()?,
                &freqs_cos.i((step..step + 1, .., ..))?,
                &freqs_sin.i((step..step + 1, .., ..))?,
                0,
                &mut cache,
            )?;
            let exp_step = expect.i((.., step..step + 1, ..))?;
            assert!(approx_eq_vec(
                &out.flatten_all()?.to_vec1()?,
                &exp_step.flatten_all()?.to_vec1()?,
                1e-1
            ));
        }
        Ok(())
    }
    #[test]
    fn test_feed_forward_from() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let _ = FeedForward::from(&mut weights, 0)?;
        assert!(true);
        Ok(())
    }
    #[test]
    fn test_feed_forward_forward() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let ffd = FeedForward::from(&mut weights, 0)?;
        let seq_len = 4;
        let device = &Device::Cpu;
        let x = Tensor::from_slice(QFFD_IN, (2, seq_len, args.dim), device)?;
        let expect = Tensor::from_slice(QFFD_OUT, (2, seq_len, args.dim), device)?;
        let y = ffd.forward(&x)?;
        assert_eq!(y.dims3()?, (2, seq_len, args.dim));
        assert!(approx_eq_vec(
            &y.flatten_all()?.to_vec1()?,
            &expect.flatten_all()?.to_vec1()?,
            1e-1 // we compare the result from quantized model with reference from f32 model
        ));
        Ok(())
    }
    #[test]
    fn test_block_from() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let _ = TransformerBlock::from(&mut weights, &args, 0)?;
        assert!(true);
        Ok(())
    }
    #[test]
    fn test_block_forward() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let block = TransformerBlock::from(&mut weights, &args, 0)?;

        let seq_len = 4;
        let device = &Device::Cpu;
        let (freqs_cos, freqs_sin) =
            precompute_freqs_cis(args.dim / args.n_heads, args.max_seq_len, 10000.0)?;
        let freqs_cos = freqs_cos.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_sin = freqs_sin.reshape((args.max_seq_len, args.dim / args.n_heads / 2, 1))?;
        let freqs_cos = freqs_cos.i((..seq_len, .., ..))?;
        let freqs_sin = freqs_sin.i((..seq_len, .., ..))?;

        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None];
        let input = Tensor::from_slice(QATT_INP, (2, seq_len, args.dim), device)?;

        let output = block.forward(&input, &freqs_cos, &freqs_sin, 0, &mut cache)?;
        assert_eq!(output.dims3()?, (2, seq_len, args.dim));
        let last_step = output.i((.., seq_len - 1, ..))?;
        let expect = Tensor::from_slice(QBLOCK_OUT, (2, seq_len, args.dim), device)?;
        let expect = expect.i((.., seq_len - 1, ..))?;
        assert!(approx_eq_vec(
            &last_step.flatten_all()?.to_vec1()?,
            &expect.flatten_all()?.to_vec1()?,
            1e-1 // we compare the result from quantized model with reference from f32 model
        ));

        Ok(())
    }
    #[test]
    fn test_model_from() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let _ = Transformer::from(&mut weights, &args)?;
        assert!(true);
        Ok(())
    }
    #[test]
    fn test_model_forward() -> anyhow::Result<()> {
        let path = env::current_dir()
            .unwrap()
            .join("tests")
            .join("data")
            .join("test_qmodel.bin");
        let mut file = File::open(path)?;
        let args = ModelArgs::from_reader_v1(&mut file).expect("failed to read header");
        let mut weights = TransformerWeights::from_reader(&mut file, &args, &Device::Cpu)?;
        let mut model = Transformer::from(&mut weights, &args)?;
        let seq_len = 4;
        let device = &Device::Cpu;
        let input =
            Tensor::from_slice(&[13u32, 24, 54, 63, 77, 104, 42, 26], (2, seq_len), device)?;

        let output = model.forward(&input, 0)?;
        assert_eq!(output.dims3()?, (2, seq_len, args.vocab_size));
        let last_step = output.i((.., seq_len - 1, ..))?;
        let expect = Tensor::from_slice(QMOD_OUT, (2, 1, args.vocab_size), device)?;
        assert!(approx_eq_vec(
            &last_step.flatten_all()?.to_vec1()?,
            &expect.flatten_all()?.to_vec1()?,
            1e0 // we compare the result from quantized model with reference from f32 model
                // given that qmodel produce reasonable output from generate function, implementation
                // seems to be correct, TODO: maybe find reference quantized model?
        ));
        Ok(())
    }
    #[test]
    fn test_precompute_frecs_cis() -> Result<()> {
        let (dim, n_heads, max_seq_len) = (8, 2, 32);
        let (freqs_cos, freqs_sin) = precompute_freqs_cis(dim / n_heads, max_seq_len, 10000.0)?;
        assert_eq!(freqs_cos.dims(), &[max_seq_len, dim / n_heads / 2]);
        assert_eq!(freqs_sin.dims(), &[max_seq_len, dim / n_heads / 2]);
        assert!(approx_eq_vec(
            &TB_FREQS_COS.to_vec(),
            &freqs_cos.flatten_all()?.to_vec1::<f32>()?,
            1e-3
        ));
        assert!(approx_eq_vec(
            &TB_FREQS_SIN.to_vec(),
            &freqs_sin.flatten_all()?.to_vec1::<f32>()?,
            1e-3
        ));
        Ok(())
    }
}
