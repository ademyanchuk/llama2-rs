use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[test]
fn test_non_existing_file() {
    // Path to your executable. Adjust if needed.
    let exe_path = "target/debug/llama2-rs";

    // Run the application with a non-existing file path as argument.
    let output = Command::new(exe_path)
        .arg("not_exist.bin")
        .output()
        .expect("Failed to execute the command");

    // Convert the output to a String for easier assertions.
    let output_msg = String::from_utf8_lossy(&output.stderr); // assuming the error message is in stderr

    // Check that the output contains the expected error message.
    assert!(output_msg
        .contains("invalid value 'not_exist.bin' for '<MODEL_PATH>': The path does not exist"));

    // Check the exit code.
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn test_invalid_model_file() {
    // Create a dummy temporary file
    let mut tmp_file = NamedTempFile::new().expect("Unable to create temporary file");
    writeln!(tmp_file, "This is not a valid model.").expect("Unable to write to temporary file");

    // Path to your release executable. Adjust if needed.
    let exe_path = "target/debug/llama2-rs";

    // Run the application with the temporary file path as argument.
    let output = Command::new(exe_path)
        .arg(tmp_file.path())
        .output()
        .expect("Failed to execute the command");

    // Convert the output to a String for easier assertions.
    let output_msg = String::from_utf8_lossy(&output.stderr); // assuming the error message is in stderr

    // Check that the output contains the expected error message.
    println!("{}", output_msg);
    assert!(output_msg.contains("Model args validation error"));

    // Temporary file is automatically deleted when tmp_file goes out of scope.
}

#[test]
fn test_invalid_qmodel_file() {
    // Create a dummy temporary file
    let mut tmp_file = NamedTempFile::new().expect("Unable to create temporary file");
    writeln!(tmp_file, "This is not a valid model.").expect("Unable to write to temporary file");

    // Path to your release executable. Adjust if needed.
    let exe_path = "target/debug/llama2-rs";

    // Run the application with the temporary file path as argument.
    let output = Command::new(exe_path)
        .arg(tmp_file.path())
        .arg("-q")
        .output()
        .expect("Failed to execute the command");

    // Convert the output to a String for easier assertions.
    let output_msg = String::from_utf8_lossy(&output.stderr); // assuming the error message is in stderr

    // Check that the output contains the expected error message.
    println!("{}", output_msg);
    assert!(output_msg.contains("Model args validation error"));

    // Temporary file is automatically deleted when tmp_file goes out of scope.
}
