use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn get_exe_path() -> String {
    std::env::var("EXE_PATH").unwrap_or("target/debug/llama2-rs".to_string())
}

fn get_model_path() -> PathBuf {
    let mut path = env::current_dir().expect("Failed to get current directory");
    path.push("stories15M_BlockQ8.bin");
    path
}

/// Load a wordlist into a HashSet for quick lookup.
/// https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-usa.txt
fn load_wordlist() -> HashSet<String> {
    let list_path = env::current_dir()
        .unwrap()
        .join("tests")
        .join("data")
        .join("google-10000-english-usa.txt");
    let data = fs::read_to_string(list_path).expect("Unable to read the wordlist");
    data.lines().map(|line| line.to_string()).collect()
}
// Allow 15% of the words to be outside the wordlist [list is most frequent 10000 words]
const ACCEPTABLE_OUTSIDE_WORDLIST_PERCENTAGE: f32 = 15.0;

#[test]
fn test_model_output_sanity() {
    // 1. Run your model and capture its output.
    let exe_path = get_exe_path();
    let model_path = get_model_path();
    assert!(
        model_path.exists(),
        "Model does not exist at {:?}",
        model_path
    );
    let output = Command::new(exe_path)
        .arg(model_path)
        .arg("-q")
        .args(["-n", "30"])
        .output()
        .expect("Failed to run the model binary");

    let output_str = String::from_utf8(output.stdout).expect("Not UTF8 output");
    // 2. Tokenize the output.
    let tokens: Vec<String> = output_str
        .split_whitespace()
        .filter_map(|word| {
            let cleaned = word
                .chars()
                .filter(|ch| ch.is_alphabetic())
                .collect::<String>()
                .to_lowercase();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect();

    // 3. Check tokens against a wordlist.
    let wordlist = load_wordlist();

    let mut outside_wordlist_count = 0;
    for token in &tokens {
        if !wordlist.contains(token) {
            println!("{}", token);
            outside_wordlist_count += 1;
        }
    }

    let outside_wordlist_percentage = (outside_wordlist_count as f32 / tokens.len() as f32) * 100.0;

    assert!(
        outside_wordlist_percentage <= ACCEPTABLE_OUTSIDE_WORDLIST_PERCENTAGE,
        "Too many tokens outside of wordlist: {:.2}% > {}%",
        outside_wordlist_percentage,
        ACCEPTABLE_OUTSIDE_WORDLIST_PERCENTAGE
    );
}
