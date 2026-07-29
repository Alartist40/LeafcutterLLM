//! Safetensor streaming backend.
//!
//! Spawns `scripts/leafcutter_safetensor_run.py` as a subprocess and
//! parses its newline-delimited JSON events:
//!
//!   {"type":"thinking_open"}      -- before first think_open token
//!   {"type":"thinking_close"}     -- after think_close token
//!   {"type":"token","text":"..."} -- streamed surface token
//!   {"type":"done","tokens":N,"duration_s":D}
//!   {"type":"error","message":"..."}
//!
//! Uses HuggingFace transformers under the hood.  This is the "reference"
//! backend — proven correct on Ornith via direct transformers testing
//! (top-1 "Paris" on a factual prompt), but slow on CPU.  AirLLM uses
//! the same approach with layer-sharded safetensors.
//!
//! Use this when:
//!   * the native GGUF engine produces wrong output for a hybrid model
//!     (Qwen3.5 / Ornith / Gemma3.5)
//!   * the user has a .safetensors checkpoint and wants guaranteed-correct
//!     chat TODAY, even if slow
//!
//! Once the native engine is fixed, this backend stays as a fallback.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[derive(Debug, Serialize)]
struct RunCommand<'a> {
    path: &'a str,
    prompt: &'a str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    stop: Vec<String>,
    think_open: i64,
    think_close: i64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RunEvent {
    #[serde(rename = "thinking_open")]
    ThinkingOpen,
    #[serde(rename = "thinking_close")]
    ThinkingClose,
    #[serde(rename = "token")]
    Token {
        text: String,
        #[serde(default)]
        #[allow(dead_code)]
        in_thinking: bool,
    },
    #[serde(rename = "done")]
    Done {
        tokens: usize,
        #[allow(dead_code)]
        duration_s: f64,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

/// Run the safetensors model and stream tokens into the callback.
///
/// `callback` receives `(text, in_thinking)` per streamed surface token
/// and returns `true` to continue or `false` to abort.
///
/// Returns Ok(total_tokens) on success or Err on subprocess failure.
pub fn stream<F>(
    model_dir: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    stop: &[String],
    mut callback: F,
) -> Result<usize, String>
where
    F: FnMut(&str, bool) -> bool,
{
    // Locate the python script.  It lives next to the leafcutter
    // binary's source tree — try a few likely paths.
    let script_candidates = [
        "scripts/leafcutter_safetensor_run.py",
        "./scripts/leafcutter_safetensor_run.py",
        "../scripts/leafcutter_safetensor_run.py",
    ];
    let script = script_candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .ok_or_else(|| {
            "leafcutter_safetensor_run.py not found; expected in scripts/".to_string()
        })?;

    let cmd = RunCommand {
        path: model_dir,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop: stop.to_vec(),
        think_open: 248068,
        think_close: 248069,
    };
    let cmd_json = serde_json::to_string(&cmd).map_err(|e| format!("json encode: {}", e))?;

    // Pick a Python interpreter that has transformers + torch installed.
    let python = pick_python().ok_or_else(|| {
        "no Python with transformers+torch found; install: \
         pip install transformers torch safetensors"
            .to_string()
    })?;

    let mut child = Command::new(&python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn {python} {script}: {e}"))?;

    eprintln!("[safetensor-backend] spawned {python} {script}");
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(cmd_json.as_bytes())
            .map_err(|e| format!("write stdin: {e}"))?;
    }
    // Close stdin so the Python script's read() returns EOF promptly.
    drop(child.stdin.take());

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "no stdout from python".to_string())?;
    let reader = BufReader::new(stdout);

    let mut total_tokens = 0usize;
    let mut in_thinking = false;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read line: {e}"))?;
        if line.is_empty() {
            continue;
        }
        let ev: RunEvent = serde_json::from_str(&line)
            .map_err(|e| format!("bad event line {:?}: {}", line, e))?;
        match ev {
            RunEvent::ThinkingOpen => {
                in_thinking = true;
            }
            RunEvent::ThinkingClose => {
                in_thinking = false;
            }
            RunEvent::Token { text, .. } => {
                if !callback(&text, in_thinking) {
                    break;
                }
            }
            RunEvent::Done { tokens, .. } => {
                total_tokens = tokens;
            }
            RunEvent::Error { message } => {
                return Err(format!("python error: {message}"));
            }
        }
    }

    let status = child
        .wait()
        .map_err(|e| format!("wait python: {e}"))?;
    if !status.success() {
        return Err(format!("python exited with {status:?}"));
    }
    Ok(total_tokens)
}

/// Detect a usable Python interpreter.
fn pick_python() -> Option<String> {
    for cand in ["python3", "python"] {
        if Command::new(cand)
            .arg("-c")
            .arg("import transformers, torch, safetensors; print('ok')")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .is_some()
        {
            return Some(cand.to_string());
        }
    }
    None
}
