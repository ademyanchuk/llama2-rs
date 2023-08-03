use ndarray::parallel::prelude::*;
use ndarray::{Array, Array2, Axis, Dim, IxDynImpl, Zip};
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

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let input = Array::from_shape_vec((input_data.len(),), input_data).unwrap();
        let output = embedding.forward(ndarray::ArrayD::from(input));
        assert_eq!(output.dim(), (input.dim(), embedding_dim));
    }
}
