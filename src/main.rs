use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Ok, Result};
use clap::{Parser, Subcommand};

use candle_core::{Device, IndexOp, Tensor};

use llama2_rs::cnd_model::{self, Transformer};
use llama2_rs::cnd_weights::TransformerWeights;
use llama2_rs::model::ModelArgs;
use llama2_rs::qmodel;
use llama2_rs::sampler::LogitsSampler;
use tokenizers::Tokenizer;

enum TransformerModel {
    F32Model(cnd_model::Transformer),
    Q80Model(qmodel::Transformer),
}

fn temp_in_range(s: &str) -> Result<f64, String> {
    let temp: f64 = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a floating point number"))?;
    if temp < 0.0 {
        Err("temperature not in range [0,inf]".to_string())
    } else {
        Result::Ok(temp)
    }
}
fn topp_in_range(s: &str) -> Result<f64, String> {
    let topp: f64 = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a floating point number"))?;
    if (0.0..=1.0).contains(&topp) {
        Result::Ok(topp)
    } else {
        Err("top-p value not in range [0,1]".to_string())
    }
}
fn validate_path(val: &str) -> Result<PathBuf, String> {
    let path = Path::new(val);
    if path.exists() {
        Result::Ok(path.to_path_buf())
    } else {
        Err(format!("The path does not exist: {:?}", path))
    }
}
#[derive(Parser, Debug, Clone)]
struct GenerateCmd {
    /// temperature in [0,inf], 1.0 = no change, < 1.0 = less random, > 1.0 = more random
    #[arg(long, short, default_value_t = 1.0, value_parser = temp_in_range)]
    temperature: f64,

    /// p value in top-p (nucleus) sampling in [0,1]
    #[arg(short, default_value_t = 0.9, value_parser = topp_in_range)]
    p: f64,

    /// number of steps to run for, default 256. 0 = max_seq_len
    #[arg(long, short, default_value_t = 256)]
    num_steps: usize,

    /// input prompt
    #[arg(long, short, default_value = "")]
    input: String,
}
#[derive(Parser, Debug, Clone)]
struct ChatCmd {}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Generate(GenerateCmd),
    Chat(ChatCmd),
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// model.bin path, currently you need to provide v0 model for f32 weights,
    /// and v1 model for quantized weights [we quantize them on load]
    #[arg(value_parser = validate_path)]
    path: PathBuf,
    /// if you want to quantize model weights [model must be exported via v1 export]
    #[arg(long, short, default_value = "false")]
    quantize: bool,
    /// mode: generate|chat, default: generate
    #[command(subcommand)]
    mode: Option<Task>,
}

fn load_args<R: io::Read + io::Seek>(r: &mut R, quantize: bool) -> ModelArgs {
    if quantize {
        ModelArgs::from_reader_v1(r).expect("Make sure to provide v1 exported model!")
    } else {
        ModelArgs::from_reader(r).expect("Make sure to provide v0 exported model!")
    }
}
fn load_model<R: io::Read>(
    r: &mut R,
    args: &ModelArgs,
    quantize: bool,
    device: &Device,
) -> Result<TransformerModel> {
    if quantize {
        let mut weights = qmodel::TransformerWeights::from_reader(r, args, device)?;
        Ok(TransformerModel::Q80Model(qmodel::Transformer::from(
            &mut weights,
            args,
        )?))
    } else {
        let weights = TransformerWeights::from_reader(r, args, device)?;
        let vb = weights.var_builder(args, device)?;
        Ok(TransformerModel::F32Model(Transformer::from(vb, args)?))
    }
}
fn print_token(next_token: u32, tokenizer: &Tokenizer) {
    // Extracting the last token as a string is complicated, here we just apply some simple
    // heuristics as it seems to work well enough for this example. See the following for more
    // details:
    // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
    if let Some(text) = tokenizer.id_to_token(next_token) {
        let text = text.replace('▁', " ").replace("<0x0A>", "\n");
        let ascii = text
            .strip_prefix("<0x")
            .and_then(|t| t.strip_suffix('>'))
            .and_then(|t| u8::from_str_radix(t, 16).ok());
        match ascii {
            None => print!("{text}"),
            Some(ascii) => {
                if let Some(chr) = char::from_u32(ascii as u32) {
                    if chr.is_ascii() {
                        print!("{chr}")
                    }
                }
            }
        }
        let _ = io::stdout().flush();
    }
}

fn generate<P: AsRef<Path>>(args: &GenerateCmd, model_path: P, quantize: bool) -> Result<()> {
    // hardcode for now, TODO: CLI as in original repo
    let tok_path = "tokenizer.json";
    let prompt = args.input.clone();
    let mut max_new_tokens = args.num_steps; // number of tokens generated in each sample
    let temperature = args.temperature; // 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    let top_p = args.p; // nuclear sampling (sample from only top_p probabilities)

    // Load the model
    let device = &Device::Cpu;
    let mut f = File::open(model_path)?;
    let model_args = load_args(&mut f, quantize);
    let mut model = load_model(&mut f, &model_args, quantize, device)?;

    // update max_new_tokens to be in range
    if max_new_tokens == 0 || max_new_tokens > model_args.max_seq_len {
        max_new_tokens = model_args.max_seq_len;
    }

    // tokenizer and sampler
    // To avoid rewriting the tokenizer myself, I utilize hugging face tokenizer library
    // We need tokenizer.json from https://huggingface.co/hf-internal-testing/llama-tokenizer/tree/main
    // or alternatively can use api as in candle llama-2 example
    let enc = Tokenizer::from_file(tok_path).expect("tokenizer loading failed");
    let mut sampler = LogitsSampler::new(
        13,
        Some(temperature),
        if top_p > 0.0 { Some(top_p) } else { None }, // switch top-p off if value is 0.0
    );

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
        let context_size = if step > 0 { 1 } else { tokens.len() };
        let start_i = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_i..];
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        let logits = match &mut model {
            TransformerModel::Q80Model(qmodel) => qmodel.forward(&input, step)?,
            TransformerModel::F32Model(f32_model) => f32_model.forward(&input, step)?,
        };
        // only last time step
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        // sample, decode, print
        let next_token_id = sampler.sample(&logits)?;
        // we want to stop on BOS, so the story doesn't end abruptly
        if next_token_id == 1 {
            break;
        }
        // this way we peak next_token as input for the next iteration
        step += context.len();

        tokens.push(next_token_id);
        // From candle examples (using it here, to make output "interactive")
        print_token(next_token_id, &enc);
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
    let args = Args::parse();
    let path = args.path;
    match &args.mode {
        None => {
            let cmd = GenerateCmd::parse();
            generate(&cmd, path, args.quantize)
        }
        Some(Task::Generate(cmd)) => generate(cmd, path, args.quantize),
        Some(Task::Chat(_)) => todo!(),
    }
}
