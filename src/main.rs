#[allow(dead_code)]
mod model;
#[allow(dead_code)]
#[allow(clippy::approx_constant)]
mod test_data;

use anyhow::{Ok, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;

use model::Embedding;

fn main() -> Result<()> {
    // Open the file
    let mut f = File::open("stories15M.bin")?;
    let mut buffer = [0; 28]; // 7 integers * 4 bytes each

    // Read the first 28 bytes (header)
    f.read_exact(&mut buffer)?;

    // Convert bytes to integers
    let mut cursor = std::io::Cursor::new(buffer);
    let dim = cursor.read_i32::<LittleEndian>()?;
    let hidden_dim = cursor.read_i32::<LittleEndian>()?;
    let n_layers = cursor.read_i32::<LittleEndian>()?;
    let n_heads = cursor.read_i32::<LittleEndian>()?;
    let n_kv_heads = cursor.read_i32::<LittleEndian>()?;
    let vocab_size = cursor.read_i32::<LittleEndian>()?;
    let max_seq_len = cursor.read_i32::<LittleEndian>()?;
    println!("dim: {}, hidden_dim: {}, n_layers: {}, n_heads: {}, n_kv_heads: {}, vocab_size: {}, max_seq_len: {}", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len);

    let n_weights = vocab_size * dim;
    let mut buffer = vec![0; n_weights as usize * 4]; // 4 bytes per float

    // Read the appropriate number of bytes from the file.
    f.read_exact(&mut buffer)?;

    // Convert bytes to floats
    let mut cursor = std::io::Cursor::new(buffer);
    let mut emb_weights = Vec::new();
    for _ in 0..n_weights {
        let weight = cursor.read_f32::<LittleEndian>()?;
        emb_weights.push(weight);
    }
    let _tok_embedding = Embedding::new(emb_weights, vocab_size as usize, dim as usize);
    Ok(())
}
