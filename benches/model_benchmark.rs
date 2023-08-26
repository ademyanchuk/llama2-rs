use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::Rng;

use llama2_rs::model::Linear;

fn generate_random_array<I>(shape: I) -> Array<f32, I::Dim>
where
    I: ndarray::ShapeBuilder,
{
    let distr = Uniform::from(0.0..1.0); // generates values between 0.0 and 1.0
    Array::random(shape, distr)
}
fn generate_random_vec(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen::<f32>()).collect()
}

pub fn benchmark_linear_forward(c: &mut Criterion) {
    c.bench_function("linear_forward", |b| {
        // SETUP
        let in_features = 256;
        let out_features = 512;
        let weight = generate_random_vec(in_features * out_features); // shape [in_features, out_features]
        let linear = Linear::new(weight, None, in_features, out_features);
        let input = generate_random_array(ndarray::IxDyn(&[8, 64, 256]));

        b.iter(|| {
            // BENCHMARK ONLY THIS
            linear.forward(black_box(&input))
        });
    });
}

criterion_group!(benches, benchmark_linear_forward);
criterion_main!(benches);
