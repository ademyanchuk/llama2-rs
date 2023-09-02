use std::fs::File;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use candle_core::{Device, Tensor};
use candle_nn::Module;

use llama2_rs::model::Linear;
use llama2_rs::{cnd_model, cnd_weights, model};

fn model_config() -> Criterion {
    Criterion::default().measurement_time(std::time::Duration::new(20, 0))
}

fn default_config() -> Criterion {
    Criterion::default()
}

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

pub fn benchmark_ndarray_model(c: &mut Criterion) {
    c.bench_function("ndarray_model", |b| {
        // Setup
        let path = "stories15M.bin";
        let transformer = model::Transformer::from(path).expect("model load failed");
        let shape = (1, 256); // max_seq_len
        let hi = transformer.args.vocab_size;
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);

        b.iter(|| {
            let x = Array::from_shape_fn(shape, |_| rng.gen_range(0..hi)).into_dyn();
            transformer.forward(black_box(x))
        })
    });
}

pub fn benchmark_candle_model(c: &mut Criterion) {
    c.bench_function("candle_model", |b| {
        // Setup
        let path = "stories15M.bin";
        let dev = &Device::Cpu;
        let mut f = File::open(path).expect("stories.bin file is expected");
        let args = model::ModelArgs::from_reader(&mut f).expect("read model args failed");
        let ws = cnd_weights::TransformerWeights::from_reader(&mut f, &args, dev)
            .expect("read model weights failed");
        let vb = ws.var_builder(&args, dev).expect("var builder failed");
        let trns = cnd_model::Transformer::from(vb, &args).expect("model load failed");
        let shape = (1, 256);
        let hi = args.vocab_size as u32;
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);

        b.iter(|| {
            let x: Vec<_> = (0..256).map(|_| rng.gen_range(0..hi)).collect();
            let x = Tensor::from_vec(x, shape, dev).expect("x failed");
            trns.forward(black_box(&x))
        })
    });
}

criterion_group! {
    name = default_benches;
    config = default_config();
    targets = benchmark_linear_forward, benchmark_candle_linear_forward
}

criterion_group! {
    name = model_benches;
    config = model_config();
    targets = benchmark_ndarray_model, benchmark_candle_model
}
criterion_main!(default_benches, model_benches);
