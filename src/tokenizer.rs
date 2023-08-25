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
    pub fn encode(&self, s: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens = self
            .sp_model
            .encode(s)
            .unwrap()
            .into_iter()
            .map(|p| p.id)
            .collect::<Vec<_>>();
        if bos {
            tokens.insert(0, self.bos_id);
        }
        if eos {
            tokens.push(self.eos_id);
        }
        tokens
    }
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.sp_model.decode_piece_ids(tokens).unwrap()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Tokenizer::new(TOKENIZER_MODEL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let _ = Tokenizer::new(TOKENIZER_MODEL);
    }

    #[test]
    fn test_encode_basic() {
        let tokenizer = Tokenizer::new(TOKENIZER_MODEL);
        let tokens = tokenizer.encode("Hello, world!", false, false);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_encode_with_flags() {
        let tokenizer = Tokenizer::new(TOKENIZER_MODEL);

        // Test with BOS flag
        let tokens_bos = tokenizer.encode("Hello", true, false);
        assert_eq!(tokens_bos[0], tokenizer.bos_id);

        // Test with EOS flag
        let tokens_eos = tokenizer.encode("Hello", false, true);
        assert_eq!(*tokens_eos.last().unwrap(), tokenizer.eos_id);

        // Test with both BOS and EOS flags
        let tokens_both = tokenizer.encode("Hello", true, true);
        assert_eq!(tokens_both[0], tokenizer.bos_id);
        assert_eq!(*tokens_both.last().unwrap(), tokenizer.eos_id);
    }

    #[test]
    fn test_encode_empty_string() {
        let tokenizer = Tokenizer::new(TOKENIZER_MODEL);
        let tokens = tokenizer.encode("", false, false);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_decode() {
        let tokenizer = Tokenizer::new(TOKENIZER_MODEL);
        let original = "Hello, world!";
        let tokens = tokenizer.encode(original, false, false);
        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_default() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.encode("Testing default", false, false);
        assert!(!tokens.is_empty());
    }
}
