[![test](https://github.com/ademyanchuk/llama2-rs/actions/workflows/test.yaml/badge.svg)](https://github.com/ademyanchuk/llama2-rs/actions/workflows/test.yaml)
[![codecov](https://codecov.io/github/ademyanchuk/llama2-rs/graph/badge.svg?token=FAWFPD6BQF)](https://codecov.io/github/ademyanchuk/llama2-rs)
# Llama2.c Text Generation in Rust

The project is inspired by famous llama2.c by Andrej Karpathy. Make sure to go ahead, and read all or some parts of his repo amazing [readme](https://github.com/karpathy/llama2.c/tree/master). You will need to go [there](https://github.com/karpathy/llama2.c/tree/master#models) and download and/or export some model files to play with before using this rust port of his project. This rust implementation, I am working on, is purely for learning how modern transformers work and practicing programming in Rust at the same time. This repository contains a Rust implementation for text generation using the Llama2 type of model architecture. It provides capabilities for generating text based on prompts.

## Features:
- Text generation using the Llama2 model.
- Command-line interface for setting various parameters like temperature, top-p, and number of tokens to generate.
- Flexible subcommands for different tasks (`generate`, `chat`).
- Quantize model weights on load (would require a machine with 32GB of RAM available to load and quantize weights of Llama-2-7b model, but generating process will be using only about 8GB).
- Note, if you use my [fork](https://github.com/ademyanchuk/llama2.c) of llama2.c, you can export the model into v3 version, just run the same export command with `--version 3` flag, and you'll be able to export model into quantized version directly, and load it faster and using less RAM later.

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

2. **Common CLI arguments and Generate Text**:

   Generate text based on a given prompt and other parameters:
   `./target/release/your_binary_name model_v1.bin --temperature 1.0 --p 0.9 --num-steps 256 -q generate --input "Your prompt here"`

   Common model Parameters:
- path to model as positional argument.
- `--temperature`: Adjusts the randomness of predictions. Default is `1.0`.
- `--p`: The top-p (nucleus) sampling value, in range [0,1]. Default is `0.9`.
- `--num-steps`: Number of steps to run for the generation. Default is `256`.
- `--quantized`: bool flag, if weights to be quantized (works only with v1 and v3 exported model).

- `generate|chat`: task to perform.
- `generate` take as optional argument:
    - `--input`: The input prompt to start the text generation.


3. **Generate Example**:

   `cargo run --release -- stories110M.bin generate -i "One day, Alex went to"`
    
    One day, Alex went to the park with his mom. He saw a beautiful black bird perched on a tree branch. 
    Alex was so excited to see the bird and started to clap his hands. His mom smiled and said, "That bird looks so graceful, Alex!"
    Alex watched the bird for a few minutes and then he asked his mom, "Can I sing a song to the bird?" His mom said, "Yes, why not?"
    So Alex started to sing a little song about the bird. The bird flew away, but then came back again and started to clap its wings! Alex was so surprised that he started to clap too. 
    The bird was so graceful, and it danced around the branch as Alex clapped his hands. After Alex was done clapping, the bird flew away again. Alex smiled and said, "That was so much fun!"
    186 tokens generated (41.93 token/s)

4. **Chat**:

Major goal of chat functionality is to give a way to chat with meta llama-2 model.
Consider downloading llama2-7b-chat files (e.g. from hugging face hub), you'll need to go through access procedure required by Meta. Once you have files locally, run:

```python export.py llama2_7b_chat_v3.bin --version 3 --meta-llama path/to/llama_chat_folder```

to quantize model weights.

Make sure you run this command from my [fork](https://github.com/ademyanchuk/llama2.c) of llama2.c.
If you don't have it locally, just clone the repo and cd into it. Do not forget to put quantized weights `llama2_7b_chat_v3.bin` file to the root folder of this rust project.

To chat, run:

```cargo run --release -- .\llama2_7b_chat_v3.bin -n 0 -q chat```

This functionality is not yet properly tested, consider it as a proof of concept, not a stable feature

`chat` takes as optional arguments:
- `--sys-prompt`: system prompt
- `--user-prompt`: first user message to the assistant

## Limitations and Known Issues:
- Chat functionality will work properly only with chat version of llama2 (e.g. llama2-7b-chat).
- This implementation uses hugging face tokenizers crate, and `tokenizer.json` is currently used as the only tokenizer. The project doesn't support training a custom tokenizer.

## Contribution:
Feel free to contribute to the repository by submitting pull requests or by reporting issues.

## License:
MIT

## Acknowledgements:
Special thanks to the developers of `candle` and `tokenizers` for their immensely valuable libraries.