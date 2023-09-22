[![test](https://github.com/ademyanchuk/llama2-rs/actions/workflows/test.yaml/badge.svg)](https://github.com/ademyanchuk/llama2-rs/actions/workflows/test.yaml)
[![codecov](https://codecov.io/github/ademyanchuk/llama2-rs/graph/badge.svg?token=FAWFPD6BQF)](https://codecov.io/github/ademyanchuk/llama2-rs)
# Llama2.c Text Generation in Rust

The project is inspired by famous llama2.c by Andrej Karpathy. Make sure to go ahead, and read all or some parts of his repo amazing [readme](https://github.com/karpathy/llama2.c/tree/master). You will need to go [there](https://github.com/karpathy/llama2.c/tree/master#models) and download and/or export some model files to play with before using this rust port of his project. This rust implementation, I am working on, is purely for learning how modern transformers work and practicing programming in Rust at the same time. This repository contains a Rust implementation for text generation using the Llama2 type of model architecture. It provides capabilities for generating text based on prompts.

## Features:
- Text generation using the Llama2 model.
- Command-line interface for setting various parameters like temperature, top-p, and number of tokens to generate.
- Flexible subcommands for different tasks (`generate`, `chat`).
- Quantize model weights on load (would require a machine with 32GB of RAM available to load and quantize weights of Llama-2-7b model, but generating process will be using only about 8GB).

## Dependencies:

This project depends on several external libraries including:
- `std`
- `anyhow`
- `clap`
- `candle_core`
- `llama2_rs`
- `tokenizers`

## How to Use:

1. **Setup**:
   - Ensure Rust is installed on your system.
   - Clone the repository and navigate to the root directory.
   - Make sure to have .bin type of model as mentioned in the introduction (exported v0 - for f32 weights, exported v1 - for quantized weights, the project performs quantization on model load).
   - Run `cargo build --release` to compile the program.

2. **Generate Text**:

   Generate text based on a given prompt and other parameters:
   `./target/release/your_binary_name model_v1.bin -q generate --input "Your prompt here" --temperature 1.0 --p 0.9 --num-steps 256`

   Parameters:
- path to model as positional argument.
- `--quantized`: bool flag, if weights to be quantized (works only with v1 exported model).
- `generate|chat`: task to perform.
- `--input`: The input prompt to start the text generation.
- `--temperature`: Adjusts the randomness of predictions. Default is `1.0`.
- `--p`: The top-p (nucleus) sampling value, in range [0,1]. Default is `0.9`.
- `--num-steps`: Number of steps to run for the generation. Default is `256`.

1. **Generate Example**:
   `cargo run --release -- stories110M.bin generate -i "One day, Alex went to"`
    
    One day, Alex went to the park with his mom. He saw a beautiful black bird perched on a tree branch. 
    Alex was so excited to see the bird and started to clap his hands. His mom smiled and said, "That bird looks so graceful, Alex!"
    Alex watched the bird for a few minutes and then he asked his mom, "Can I sing a song to the bird?" His mom said, "Yes, why not?"
    So Alex started to sing a little song about the bird. The bird flew away, but then came back again and started to clap its wings! Alex was so surprised that he started to clap too. 
    The bird was so graceful, and it danced around the branch as Alex clapped his hands. After Alex was done clapping, the bird flew away again. Alex smiled and said, "That was so much fun!"
    186 tokens generated (41.93 token/s)

2. **Chat**:

This functionality is under development.

## Limitations and Known Issues:
- The `chat` subcommand is not yet implemented.
- Tokenizer path is currently hardcoded.

## Contribution:
Feel free to contribute to the repository by submitting pull requests or by reporting issues.

## License:
MIT

## Acknowledgements:
Special thanks to the developers of `candle` and `tokenizers` for their immensely valuable libraries.