use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::Rng;

use candle_core::{Device, Tensor};
use candle_nn::Module;

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
        let in_features = 512;
        let out_features = 1024;
        let weight = generate_random_vec(in_features * out_features); // shape [in_features, out_features]
        let linear = Linear::new(weight, None, in_features, out_features);
        let input = generate_random_array(ndarray::IxDyn(&[8, 64, in_features]));

        b.iter(|| {
            // BENCHMARK ONLY THIS
            linear.forward(black_box(&input))
        });
    });
}

pub fn benchmark_candle_linear_forward(c: &mut Criterion) {
    c.bench_function("candle_linear_forward", |b| {
        // SETUP
        let device = Device::Cpu;
        let in_features = 512;
        let out_features = 1024;
        let weight = Tensor::rand(0.0, 1.0, (out_features, in_features), &device).unwrap();
        let linear = candle_nn::Linear::new(weight, None);
        let input = Tensor::rand(0.0, 1.0, (8, 64, in_features), &device).unwrap();

        b.iter(|| {
            // BENCHMARK ONLY THIS
            linear.forward(black_box(&input)).unwrap()
        });
    });
}

criterion_group!(
    benches,
    benchmark_linear_forward,
    benchmark_candle_linear_forward
);
criterion_main!(benches);
