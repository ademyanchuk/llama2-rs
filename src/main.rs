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

fn reshape_for_broadcast(x_shape: &[usize], freq_shape: &[usize]) -> Vec<usize> {
    // Assert the shapes to ensure compatibility
    assert!(x_shape.len() > 2);
    assert_eq!(freq_shape[0], x_shape[1]);
    assert_eq!(freq_shape[1], *x_shape.last().unwrap());

    x_shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            if i == 1 || i == x_shape.len() - 1 {
                dim
            } else {
                1
            }
        })
        .collect()
}

fn reshape_to_complex(tensor: &Array<f32, IxDyn>) -> ArrayView<'_, f32, IxDyn> {
    let mut new_shape: Vec<usize> = tensor.shape().to_vec();

    // The last dimension is reshaped to become two dimensions: [n/2, 2]
    let last_dim = new_shape.pop().unwrap();
    new_shape.push(last_dim / 2);
    new_shape.push(2);

    tensor.view().into_shape(new_shape).unwrap()
}

fn apply_rotary_emb(
    xq: &ArrayD<f32>,
    xk: &ArrayD<f32>,
    freqs_cos: &Array2<f32>,
    freqs_sin: &Array2<f32>,
) -> (ArrayD<f32>, ArrayD<f32>) {
    // Reshape xq and xk to match the complex representation
    let xq_view = reshape_to_complex(xq);
    let xk_view = reshape_to_complex(xk);

    let xq_r = xq_view.index_axis(Axis(xq_view.ndim() - 1), 0);
    let xq_i = xq_view.index_axis(Axis(xq_view.ndim() - 1), 1);

    let xk_r = xk_view.index_axis(Axis(xk_view.ndim() - 1), 0);
    let xk_i = xk_view.index_axis(Axis(xk_view.ndim() - 1), 1);

    // Reshape freqs_cos and freqs_sin for broadcasting
    let new_shape_cos = reshape_for_broadcast(xq_r.shape(), freqs_cos.shape());
    let reshaped_freqs_cos = freqs_cos.view().into_shape(new_shape_cos).unwrap();

    let new_shape_sin = reshape_for_broadcast(xq_i.shape(), freqs_sin.shape());
    let reshaped_freqs_sin = freqs_sin.view().into_shape(new_shape_sin).unwrap();

    // Apply rotation using real numbers
    let xq_out_r = &xq_r * &reshaped_freqs_cos - &xq_i * &reshaped_freqs_sin;
    let xq_out_i = &xq_r * &reshaped_freqs_sin + &xq_i * &reshaped_freqs_cos;

    let xk_out_r = &xk_r * &reshaped_freqs_cos - &xk_i * &reshaped_freqs_sin;
    let xk_out_i = &xk_r * &reshaped_freqs_sin + &xk_i * &reshaped_freqs_cos;

    // Flatten dimensions starting from 3d
    let stacked = ndarray::stack(Axis(xq_out_r.ndim()), &[xq_out_r.view(), xq_out_i.view()])
        .expect("stack works with equal dims");
    let new_shape = stacked.shape().to_vec();
    let last_dim = new_shape.iter().skip(3).product();
    let mut new_shape: Vec<usize> = new_shape.into_iter().take(3).collect();
    new_shape.push(last_dim);

    let xq_out = stacked.into_shape(new_shape.as_slice()).unwrap();
    println!("{:?}", xq_out);

    let stacked = ndarray::stack(Axis(xk_out_r.ndim()), &[xk_out_r.view(), xk_out_i.view()])
        .expect("stack works with equal dims");
    let new_shape = stacked.shape().to_vec();
    let last_dim = new_shape.iter().skip(3).product();
    let mut new_shape: Vec<usize> = new_shape.into_iter().take(3).collect();
    new_shape.push(last_dim);

    let xk_out = stacked.into_shape(new_shape.as_slice()).unwrap();

    (
        xq_out.into_dimensionality::<IxDyn>().unwrap(),
        xk_out.into_dimensionality::<IxDyn>().unwrap(),
    )
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
    use ndarray::{arr1, ArrayD, ShapeBuilder};

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
        let x_shape = &[3, 5, 6];
        let freq_shape = &[5, 6];

        let new_shape = reshape_for_broadcast(x_shape, freq_shape);
        assert_eq!(new_shape, &[1, 5, 6]);
    }
    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_dim() {
        let x_shape = &[5, 6]; // only 2 dimensions
        let freq_shape = &[5, 6];

        let _ = reshape_for_broadcast(x_shape, freq_shape);
    }

    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_shape_mismatch_1() {
        let x_shape = &[3, 4, 6]; //second dimension doesn't match
        let freq_shape = &[5, 6];

        let _ = reshape_for_broadcast(x_shape, freq_shape);
    }

    #[test]
    #[should_panic]
    fn test_reshape_for_broadcast_panic_shape_mismatch_last() {
        let x_shape = &[3, 5, 7]; // last dimension doesn't match
        let freq_shape = &[5, 6];

        let _ = reshape_for_broadcast(x_shape, freq_shape);
    }
    #[test]
    fn test_reshape_to_complex() {
        let tensor = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
        let reshaped = reshape_to_complex(&tensor);
        assert_eq!(reshaped.shape(), &[2, 2, 2]);

        let tensor = ArrayD::from_elem(IxDyn(&[3, 6, 8]), 1.0f32);
        let reshaped = reshape_to_complex(&tensor);
        assert_eq!(reshaped.shape(), &[3, 6, 4, 2]);
    }
    #[test]
    fn test_apply_rotary_emb_shapes() {
        // Generate random input arrays with the specified shapes
        let xq = ArrayD::from_shape_fn(vec![2, 5, 4, 8], |_| rand::random::<f32>());
        let xk = ArrayD::from_shape_fn(vec![2, 5, 4, 8], |_| rand::random::<f32>());
        let freqs_cos = Array2::from_shape_fn((5, 4), |_| rand::random::<f32>());
        let freqs_sin = Array::from_shape_fn((5, 4), |_| rand::random::<f32>());

        let (output_xq, output_xk) = apply_rotary_emb(&xq, &xk, &freqs_cos, &freqs_sin);

        // Check shapes of the output arrays
        assert_eq!(output_xq.shape(), &[2, 5, 4, 8]);
        assert_eq!(output_xk.shape(), &[2, 5, 4, 8]);
    }
    #[test]
    fn test_apply_rotary_emb_values() {
        let xq = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3, 2, 4]),
            vec![
                -0.8394, -1.4532, -1.1078, 0.6352, 1.1605, 1.8578, -1.0871, -1.3556, 0.3745,
                -1.3348, -1.8303, -1.7940, -0.1393, -0.8891, 0.8550, 0.6536, 1.3476, -1.7046,
                -0.5971, -0.6485, -0.3225, 0.4260, -0.7996, 0.0668, 0.6275, -2.7466, 0.3128,
                -0.0060, 0.1096, -0.3225, 0.4355, -1.3955, 0.6312, -0.1921, 1.6291, -0.4094,
                0.7282, -0.2979, 0.3601, 1.0889, -1.1238, 0.8396, -0.1531, 0.3422, -1.1257, 0.0461,
                -0.8512, 1.0879,
            ],
        )
        .unwrap();
        let xk = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3, 2, 4]),
            vec![
                2.6858e-01,
                -1.1336e+00,
                1.7634e+00,
                6.3016e-01,
                -2.3757e-01,
                -1.0991e+00,
                4.3722e-01,
                8.3467e-01,
                -1.0983e+00,
                -8.2900e-02,
                -8.3557e-02,
                9.5412e-01,
                8.1022e-01,
                1.2607e-01,
                -9.8301e-01,
                -3.0040e-01,
                5.7840e-01,
                -2.3034e+00,
                2.5555e+00,
                1.2410e+00,
                -6.8892e-01,
                -4.0045e-01,
                -1.3077e-01,
                9.0362e-01,
                5.1952e-01,
                -1.4140e+00,
                2.1658e-03,
                -1.1250e+00,
                -2.1708e+00,
                5.1018e-01,
                -1.1051e+00,
                8.1955e-01,
                -4.8645e-01,
                6.7161e-01,
                1.6593e+00,
                2.0753e-01,
                2.0263e+00,
                1.1484e+00,
                1.8454e+00,
                2.6848e-01,
                1.5497e-01,
                1.0212e+00,
                -5.1874e-02,
                7.7246e-01,
                5.1927e-01,
                6.3070e-02,
                -2.1352e-02,
                -8.1618e-01,
            ],
        )
        .unwrap();
        let freqs_cos = Array2::from_shape_vec(
            (3, 2),
            vec![-0.1339, -1.4408, -0.7710, 0.4526, -3.0065, 2.3243],
        )
        .unwrap();
        let freqs_sin = Array2::from_shape_vec(
            (3, 2),
            vec![0.3798, -1.3930, -0.0854, 0.7161, 2.4592, -1.0601],
        )
        .unwrap();
        let xq_out_expected = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3, 2, 4]),
            vec![
                6.6427e-01,
                -1.2423e-01,
                2.4810e+00,
                6.2798e-01,
                -8.6092e-01,
                1.9201e-01,
                -3.2210e-01,
                3.4675e+00,
                -4.0276e-01,
                9.9708e-01,
                4.5629e-01,
                -2.1226e+00,
                3.1440e-02,
                6.9736e-01,
                -8.1089e-02,
                9.0804e-01,
                1.4047e-01,
                8.4391e+00,
                -2.0752e+00,
                -8.7430e-01,
                -7.7912e-02,
                -2.0740e+00,
                -1.7876e+00,
                1.0030e+00,
                9.5910e-01,
                6.0602e-01,
                -4.5903e-01,
                -4.2717e-01,
                1.0781e-01,
                8.4801e-02,
                -2.5715e+00,
                1.4040e+00,
                -5.0307e-01,
                9.4194e-02,
                1.0305e+00,
                9.8130e-01,
                -5.8686e-01,
                1.6744e-01,
                -6.1679e-01,
                7.5075e-01,
                1.3138e+00,
                -5.2879e+00,
                6.8969e-03,
                9.5780e-01,
                3.2709e+00,
                -2.9069e+00,
                -8.2521e-01,
                3.4310e+00,
            ],
        )
        .unwrap();
        let xk_out_expected = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3, 2, 4]),
            vec![
                0.3945, 0.2538, -1.6629, -3.3645, 0.4492, 0.0569, 0.5328, -1.8116, 0.8397, 0.1577,
                -0.7211, 0.3720, -0.6139, -0.1664, -0.2298, -0.8399, 3.9257, 8.3476, 7.2554,
                0.1753, 3.0560, -0.4903, 0.6540, 2.2389, 0.4675, 0.3866, -1.5703, 1.6178, 0.0969,
                -0.8927, 2.7338, 0.3586, 0.4324, -0.4762, 0.6024, 1.2822, -1.4641, -1.0585, 0.6429,
                1.4430, -2.9772, -2.6890, 0.6983, 1.8505, -1.7163, 1.0874, -0.9149, -1.8745,
            ],
        )
        .unwrap();
        let (output_xq, output_xk) = apply_rotary_emb(&xq, &xk, &freqs_cos, &freqs_sin);
        assert_abs_diff_eq!(
            output_xq,
            xq_out_expected,
            epsilon = 1e-4
        );
        assert_abs_diff_eq!(
            output_xk,
            xk_out_expected,
            epsilon = 1e-4
        );
    }
}
