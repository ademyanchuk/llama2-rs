use std::fs::File;
use std::io::Write;
use std::time::Instant;

use anyhow::{Ok, Result};

use candle_core::{Device, IndexOp, Tensor};

use llama2_rs::cnd_model::Transformer;
use llama2_rs::cnd_weights::TransformerWeights;
use llama2_rs::model::ModelArgs;
use llama2_rs::sampler::LogitsSampler;
use tokenizers::Tokenizer;

fn generate() -> Result<()> {
    // hardcode for now, TODO: CLI as in original repo
    let path = "stories15M.bin";
    let tok_path = "tokenizer.json";
    let prompt = "One day, Lily met a";
    let max_new_tokens = 100; // number of tokens generated in each sample
    let temperature = 1.0; // 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    let top_p = 0.9; // nuclear sampling (sample from only top_p probabilities)

    // Load the model
    let device = &Device::Cpu;
    let mut f = File::open(path)?;
    let args = ModelArgs::from_reader(&mut f)?;
    let ws = TransformerWeights::from_reader(&mut f, &args, device)?;
    let vb = ws.var_builder(&args, device)?;
    let transformer = Transformer::from(vb, &args)?;

    // tokenizer and sampler
    // To avoid rewriting the tokenizer myself, I utilize hugging face tokenizer library
    // We need tokenizer.json from https://huggingface.co/hf-internal-testing/llama-tokenizer/tree/main
    // or alternatively can use api as in candle llama-2 example
    let enc = Tokenizer::from_file(tok_path).expect("tokenizer loading failed");
    let mut sampler = LogitsSampler::new(13, Some(temperature), Some(top_p));

    print!("{}", prompt);

    let mut tokens = enc
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let start = Instant::now(); // Start timing
    let mut step = 0;
    loop {
        if step >= max_new_tokens {
            break;
        }
        // make sure we have context length >= model's max sequence length
        let start_i = tokens.len().saturating_sub(args.max_seq_len);
        let context = &tokens[start_i..];
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        let logits = transformer.forward(&input)?;
        // only last time step
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        // sample, decode, print
        let next_token_id = sampler.sample(&logits)?;
        // we want to stop on BOS, so the story doesn't end abruptly
        if next_token_id == 1 {
            break;
        }
        tokens.push(next_token_id);
        // From candle examples (using it here, to make output "interactive")
        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Some(text) = enc.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        step += 1;
    }
    let dt = start.elapsed();
    if step > 1 {
        println!(
            "\n{} tokens generated ({:.2} token/s)\n",
            step,
            step as f64 / dt.as_secs_f64(),
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    generate()
}
