use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{s, Array, Array1, Array2, ArrayD, ArrayView, Axis, Dim, IxDyn, IxDynImpl, Zip};
use std::fs::File;
use std::io::prelude::*;

pub struct ModelArgs {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: Option<usize>,
    pub vocab_size: isize,
    pub multiple_of: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub dropout: f32,
}

impl Default for ModelArgs {
    fn default() -> Self {
        ModelArgs {
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: None,
            vocab_size: -1,
            multiple_of: 256,
            norm_eps: 1e-5,
            max_seq_len: 2048,
            dropout: 0.0,
        }
    }
}

// Blocks
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        multiple_of: usize,
        w1_weights: Vec<f32>,
        w2_weights: Vec<f32>,
        w3_weights: Vec<f32>,
    ) -> FeedForward {
        let hidden_dim = 2 * hidden_dim / 3;
        let hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

        let w1 = Linear::new(w1_weights, None, in_dim, hidden_dim);
        let w2 = Linear::new(w2_weights, None, hidden_dim, in_dim);
        let w3 = Linear::new(w3_weights, None, in_dim, hidden_dim);

        FeedForward { w1, w2, w3 }
    }

    pub fn forward<T: ndarray::Dimension<Larger = Dim<IxDynImpl>>>(
        &self,
        input: Array<f32, T>,
    ) -> Array<f32, T::Larger> {
        let x1 = self.w1.forward(&input);
        let x1 = silu_inplace(x1);
        let x2 = self.w3.forward(&input);
        let result = x1 * x2;
        self.w2.forward(&result)
    }
}

// Layers
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
        input: &Array<f32, T>,
    ) -> Array<f32, T::Larger> {
        // check dimensions, panic if not correct
        if input.shape().last().unwrap_or(&0) != &self.in_features {
            panic!("The last dimension of the input must be equal to in_features");
        }
        // build input shape
        let original_shape = input.shape().to_vec();
        // compute output on flattened input
        let flattened_input = input
            .view()
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
pub struct RMSNorm {
    weight: Array1<f32>,
    eps: f32,
}
impl RMSNorm {
    pub fn new(dim: usize, eps: f32, weight_data: Vec<f32>) -> RMSNorm {
        let weight = Array1::from_shape_vec((dim,), weight_data)
            .expect("RMSNorm data shape does not match the weight dimensions");
        RMSNorm { weight, eps }
    }
    pub fn forward<T: ndarray::Dimension<Larger = Dim<IxDynImpl>> + ndarray::RemoveAxis>(
        &self,
        mut input: Array<f32, T>,
    ) -> Array<f32, T> {
        if input.shape().last().unwrap_or(&0) != self.weight.shape().last().unwrap() {
            panic!("The last dimension of the input must be equal to dimension of RMSNorm weight");
        }
        // prep dims
        let last_idx = input.ndim() - 1;
        let mut mean_shape = input.shape().to_vec();
        mean_shape[last_idx] = 1;

        // compute the mean of squares across the last dimension
        let mean = (&input * &input).mean_axis(Axis(input.ndim() - 1)).unwrap();
        let mut mean = mean.into_shape(mean_shape).unwrap(); // keep dim == True
        mean += self.eps; // add epsilon for numerical stability

        // compute the reciprocal square-root
        mean.mapv_inplace(|v| v.sqrt().recip());
        // normalize the input
        input *= &mean;
        input *= &self.weight;
        input
    }
}
// Functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_inplace<T: ndarray::Dimension>(mut input: Array<f32, T>) -> Array<f32, T> {
    input.mapv_inplace(|x| x * sigmoid(x));
    input
}

fn reshape_for_broadcast<'a>(
    freqs_cis: &'a Array2<f32>,
    x: &ArrayD<f32>,
) -> ArrayView<'a, f32, IxDyn> {
    // Ensure the shape of freqs_cis matches the second and last dimension of x
    assert!(x.ndim() > 2);
    assert_eq!(
        freqs_cis.shape(),
        &[x.shape()[1], *x.shape().last().unwrap()]
    );

    // Calculate the new shape
    let shape: Vec<usize> = x
        .shape()
        .iter()
        .enumerate()
        .map(
            |(i, &dim)| {
                if i == 1 || i == x.ndim() - 1 {
                    dim
                } else {
                    1
                }
            },
        )
        .collect();

    // Reshape freqs_cis and return
    freqs_cis.view().into_shape(shape).unwrap()
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
    use ndarray::arr1;

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
        let output = linear.forward(&input);
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
        let output = linear.forward(&input);
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
        let output = linear.forward(&input);
        let expected_output = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 2, 2]),
            vec![14.0, 32.0, 32.0, 77.0, 50.0, 122.0, 68.0, 167.0],
        )
        .unwrap(); // shape [2, 2, 2]
        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-6);
    }
    #[test]
    fn test_rmsnorm_new() {
        // create an instance of RMSNorm
        let norm = RMSNorm::new(4, 0.01, vec![1.0, 1.0, 1.0, 1.0]);

        // check that the properties were correctly set
        assert_eq!(norm.eps, 0.01);
        assert_eq!(norm.weight, arr1(&[1.0, 1.0, 1.0, 1.0]));
    }
    #[test]
    fn test_rmsnorm_forward() {
        let rms_norm = RMSNorm::new(3, 1e-5, vec![1.0, 1.0, 1.0]);

        let input =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let output = rms_norm.forward(input);

        let expected_output = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![0.4629, 0.9258, 1.3887, 0.7895, 0.9869, 1.1843],
        )
        .unwrap();

        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-4);
    }
    #[test]
    #[should_panic(
        expected = "The last dimension of the input must be equal to dimension of RMSNorm weight"
    )]
    fn rmsnorm_forward_panic_test() {
        let rmsnorm = RMSNorm::new(3, 1e-5, vec![1.0, 1.0, 1.0]);
        let input =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 4.0, 5.0]).unwrap();
        let _ = rmsnorm.forward(input);
    }
    #[test]
    fn test_silu_inplace() {
        let input = Array2::from_shape_vec(
            (2, 3),
            vec![0.3659, -1.6326, -0.5668, -0.7105, 0.7709, 0.1407],
        )
        .unwrap();
        let expected_output = Array2::from_shape_vec(
            (2, 3),
            vec![0.2161, -0.2669, -0.2052, -0.2341, 0.5271, 0.0753],
        )
        .unwrap();

        let output = silu_inplace(input);

        assert_abs_diff_eq!(output, expected_output, epsilon = 1e-4);
    }
    #[test]
    fn test_feedforward() {
        // 1. Initialize FeedForward with predefined weights (came from PyTorch implementation)
        let w1_weights = vec![
            -0.4764, 0.0453, 0.1539, 0.0752, -0.2386, 0.3814, 0.1513, -0.3863, 0.4562, 0.2769,
            0.4002, -0.1647,
        ];
        let w2_weights = vec![
            0.2170, 0.3585, -0.2992, 0.3554, -0.4850, 0.2447, 0.1820, 0.2602, 0.0146, 0.1802,
            -0.4978, -0.0919,
        ];
        let w3_weights = vec![
            -0.4622, -0.5098, 0.4391, 0.4349, -0.4857, 0.3582, 0.2414, 0.3671, 0.2596, 0.2129,
            0.0142, 0.1426,
        ];
        let feed_forward = FeedForward::new(3, 4, 4, w1_weights, w2_weights, w3_weights);

        // 2. Provide it with a predefined input
        let input = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![0.0874, -0.7098, -1.6503, -0.5212, -1.3321, -0.5542],
        )
        .unwrap();

        // 3. Get the result and compare it with expected output.
        let result = feed_forward.forward(input);
        let expected_output = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![-0.0112, 0.0036, -0.0521, 0.0489, -0.0182, 0.0356],
        )
        .unwrap();
        assert_abs_diff_eq!(result, expected_output, epsilon = 1e-4);
    }
    #[test]
    fn test_reshape_for_broadcast_shapes() {
        let x = Array::ones(vec![3, 5, 6]);
        let freqs_cis = Array2::<f32>::ones((5, 6));

        let reshaped = reshape_for_broadcast(&freqs_cis, &x);
        assert_eq!(reshaped.shape(), &[1, 5, 6]);
    }
    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_dim() {
        let x = Array::ones(vec![5, 6]); // x has only 2 dimensions
        let freqs_cis = Array2::<f32>::ones((5, 6));

        let _ = reshape_for_broadcast(&freqs_cis, &x);
    }

    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_shape_mismatch_1() {
        let x = Array::ones(vec![3, 4, 6]); // Second dimension doesn't match
        let freqs_cis = Array2::<f32>::ones((5, 6));

        let _ = reshape_for_broadcast(&freqs_cis, &x);
    }

    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_shape_mismatch_last() {
        let x = Array::ones(vec![3, 5, 7]); // Last dimension doesn't match
        let freqs_cis = Array2::<f32>::ones((5, 6));

        let _ = reshape_for_broadcast(&freqs_cis, &x);
    }
}
