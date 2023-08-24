use std::path::Path;

use sentencepiece::SentencePieceProcessor;

pub const TOKENIZER_MODEL: &str = "tokenizer.model";

// First we just have really simple implementation
// which only works the llama sentencepiece tokenizer model
pub struct Tokenizer {
    sp_model: SentencePieceProcessor,
    bos_id: u32,
    eos_id: u32,
}

impl Tokenizer {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Tokenizer {
        let sp_model = SentencePieceProcessor::open(model_path)
            .expect("expect the path to the sentencepiece model");
        let bos_id = sp_model.bos_id().unwrap();
        let eos_id = sp_model.eos_id().unwrap();
        Tokenizer {
            sp_model,
            bos_id,
            eos_id,
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Tokenizer::new(TOKENIZER_MODEL)
    }
}
