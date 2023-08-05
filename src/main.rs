use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{s, Array, Array1, Array2, Dim, IxDynImpl, Zip};
use std::fs::File;
use std::io::prelude::*;
pub struct Embedding {
    weight: Array2<f32>,
}
impl Embedding {
    pub fn new(data: Vec<f32>, num_embeddings: usize, embedding_dim: usize) -> Embedding {
        // default from_shape_vec is row-major
        let weight = Array2::from_shape_vec((num_embeddings, embedding_dim), data)
            .expect("Data shape does not match the embedding matrix dimensions");

        Embedding { weight }
    }

    pub fn forward<T: ndarray::Dimension<Larger = Dim<IxDynImpl>>>(
        &self,
        input: Array<usize, T>,
    ) -> Array<f32, T::Larger> {
        let original_shape = input.shape().to_vec();
        let flattened_input = input
            .into_shape((original_shape.iter().product(),))
            .unwrap();

        let mut output_shape: Vec<usize> = original_shape.clone();
        output_shape.push(self.weight.shape()[1]);
        let mut output = Array::zeros(output_shape.as_slice());

        let mut flattened_output = output
            .view_mut()
            .into_shape((original_shape.iter().product(), self.weight.shape()[1]))
            .unwrap();
        Zip::from(flattened_output.rows_mut())
            .and(&flattened_input)
            .par_for_each(|mut output_row, &index| {
                output_row.assign(&self.weight.row(index));
            });
        // for (i, index) in flattened_input.iter().enumerate() {
        //     flattened_output.row_mut(i).assign(&self.weight.row(*index));
        // }

        output
    }
}

pub struct Linear {
    weight: Array2<f32>,
    bias: Option<Array1<f32>>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(
        weight_data: Vec<f32>,
        bias_data: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Linear {
        let weight = Array2::from_shape_vec((out_features, in_features), weight_data)
            .expect("Data shape does not match the linear weight dimensions");
        let bias = bias_data.map(|bias_vec| {
            Array1::from_shape_vec(out_features, bias_vec)
                .expect("Data shape does not match the linear bias dimensions")
        });
        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
    pub fn forward<T: ndarray::Dimension<Larger = Dim<IxDynImpl>>>(
        &self,
        input: Array<f32, T>,
    ) -> Array<f32, T::Larger> {
        // check dimensions, panic if not correct
        if input.shape().last().unwrap_or(&0) != &self.in_features {
            panic!("The last dimension of the input must be equal to in_features");
        }
        // build input shape
        let original_shape = input.shape().to_vec();
        // compute output on flattened input
        let flattened_input = input
            .into_shape((
                original_shape.iter().rev().skip(1).product(), // takes all but last shapes and multiplies them
                self.in_features,
            ))
            .unwrap();

        let mut output = flattened_input.dot(&self.weight.t());
        if let Some(bias) = &self.bias {
            output += bias;
        }
        // reshape to the final output shape
        let mut output_shape = original_shape;
        let last_ix = output_shape.len() - 1;
        output_shape[last_ix] = self.out_features;
        output.into_shape(output_shape).unwrap()
    }
}

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
    let tok_embedding = Embedding::new(emb_weights, vocab_size as usize, dim as usize);
    println!("{}", tok_embedding.weight.slice(s![..3, ..5]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_new_with_valid_inputs() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let num_embeddings = 2;
        let embedding_dim = 2;
        let embedding = Embedding::new(data, num_embeddings, embedding_dim);
        assert_eq!(embedding.weight.shape(), &[num_embeddings, embedding_dim]);
    }

    #[test]
    #[should_panic(expected = "Data shape does not match the embedding matrix dimensions")]
    fn test_new_with_invalid_inputs() {
        let data = vec![1.0, 2.0, 3.0];
        let num_embeddings = 2;
        let embedding_dim = 2;
        let _embedding = Embedding::new(data, num_embeddings, embedding_dim);
    }

    #[test]
    fn test_weights() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let num_embeddings = 2;
        let embedding_dim = 2;
        let embedding = Embedding::new(data.clone(), num_embeddings, embedding_dim);
        for (i, weight) in embedding.weight.iter().enumerate() {
            assert_eq!(*weight, data[i]);
        }
    }
    #[test]
    fn test_new_with_zero_dimensions() {
        let data: Vec<f32> = vec![];
        let num_embeddings = 0;
        let embedding_dim = 2;
        let embedding = Embedding::new(data.clone(), num_embeddings, embedding_dim);
        assert_eq!(embedding.weight, Array2::zeros((0, 2)));

        let num_embeddings = 2;
        let embedding_dim = 0;
        let embedding = Embedding::new(data, num_embeddings, embedding_dim);
        assert_eq!(embedding.weight, Array2::zeros((2, 0)));
    }
    #[test]
    fn test_new_with_large_input() {
        let data: Vec<f32> = vec![1.0; 10_000];
        let num_embeddings = 100;
        let embedding_dim = 100;
        let embedding = Embedding::new(data, num_embeddings, embedding_dim);
        assert_eq!(embedding.weight.shape(), &[num_embeddings, embedding_dim]);
    }
    #[test]
    fn test_embedding_forward_shape() {
        let num_embeddings = 5;
        let embedding_dim = 3;
        let weight_data = vec![0.0; num_embeddings * embedding_dim];
        let embedding = Embedding::new(weight_data, num_embeddings, embedding_dim);

        let input_data = vec![1usize, 3, 4];
        let input_len = input_data.len();
        let input = Array::from_shape_vec(ndarray::IxDyn(&[input_len]), input_data).unwrap();
        let output = embedding.forward(input);
        assert_eq!(output.dim(), ndarray::IxDyn(&[input_len, embedding_dim]));
    }
    #[test]
    fn test_embedding_forward_values() {
        let num_embeddings = 5;
        let embedding_dim = 3;
        let weight_data = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        ];
        let embedding = Embedding::new(weight_data, num_embeddings, embedding_dim);

        let input = Array::from_shape_vec(ndarray::IxDyn(&[3]), vec![1usize, 3, 4]).unwrap();
        let output = embedding.forward(input.clone());

        let expected_output = Array::from_shape_vec(
            ndarray::IxDyn(&[3, embedding_dim]),
            vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        )
        .unwrap();
        assert_eq!(output, expected_output);
    }
    #[test]
    fn test_embedding_forward_shape_2d() {
        let num_embeddings = 5;
        let embedding_dim = 3;
        let weight_data = vec![0.0; num_embeddings * embedding_dim];
        let embedding = Embedding::new(weight_data, num_embeddings, embedding_dim);

        let input_data = vec![1usize, 3, 4, 2, 0, 1];
        let rows = 2;
        let cols = input_data.len() / rows;
        let input = Array::from_shape_vec(ndarray::IxDyn(&[rows, cols]), input_data).unwrap();

        let output = embedding.forward(input.clone());

        assert_eq!(output.dim(), ndarray::IxDyn(&[rows, cols, embedding_dim]));
    }
    #[test]
    fn test_linear_new_with_bias() {
        let weight_data = vec![1.0, 2.0, 3.0, 4.0];
        let bias_data = vec![0.1, 0.2];
        let linear = Linear::new(weight_data.clone(), Some(bias_data.clone()), 2, 2);

        assert_eq!(
            linear.weight,
            Array2::from_shape_vec((2, 2), weight_data).unwrap()
        );
        assert_eq!(
            linear.bias.unwrap(),
            Array1::from_shape_vec((2,), bias_data).unwrap()
        );
    }

    #[test]
    fn test_linear_new_without_bias() {
        let weight_data = vec![1.0, 2.0, 3.0, 4.0];
        let linear = Linear::new(weight_data.clone(), None, 2, 2);

        assert_eq!(
            linear.weight,
            Array2::from_shape_vec((2, 2), weight_data).unwrap()
        );
        assert!(linear.bias.is_none());
    }
    #[test]
    fn test_linear_2d_input_no_bias() {
        let in_features = 3;
        let out_features = 2;
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
        let linear = Linear::new(weight, None, in_features, out_features);
        let input =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let output = linear.forward(input);
        let expected_output =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![14.0, 32.0, 32.0, 77.0]).unwrap(); // shape [2, 2]
        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-6);
    }
    #[test]
    fn test_linear_2d_input_with_bias() {
        let in_features = 3;
        let out_features = 2;
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
        let bias = vec![0.5, -0.5];
        let linear = Linear::new(weight, Some(bias), in_features, out_features);
        let input =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let output = linear.forward(input);
        let expected_output =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![14.5, 31.5, 32.5, 76.5]).unwrap(); // shape [2, 2]
        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-6);
    }
    #[test]
    fn test_linear_3d_input_no_bias() {
        let in_features = 3;
        let out_features = 2;
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2, 3]
        let linear = Linear::new(weight, None, in_features, out_features);
        let input = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 2, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let output = linear.forward(input);
        let expected_output = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 2, 2]),
            vec![14.0, 32.0, 32.0, 77.0, 50.0, 122.0, 68.0, 167.0],
        )
        .unwrap(); // shape [2, 2, 2]
        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-6);
    }
}
