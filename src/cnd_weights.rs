// I reuse a lot of conveniences from candle llama2 example
// Here is everything about reading from .bin Karpathy's exports
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use candle_nn::VarBuilder;

use crate::model::{ModelArgs, ModelArgsBuilder};

fn read_i32<R: std::io::Read>(r: &mut R) -> Result<i32> {
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
        Ok(ModelArgsBuilder::new()
            .dim(read_i32(r)? as usize)
            .hidden_dim(read_i32(r)? as usize)
            .n_layers(read_i32(r)? as usize)
            .n_heads(read_i32(r)? as usize)
            .n_kv_heads(read_i32(r)? as usize)
            .vocab_size(read_i32(r)? as usize)
            .max_seq_len(read_i32(r)? as usize)
            .build())
    }
}
