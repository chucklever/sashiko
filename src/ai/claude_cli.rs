// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AI provider that shells out to the `claude` CLI instead of calling the API directly.
//! This uses the local Claude Code installation (subscription auth) rather than API credits.
//!
//! ## Safety
//!
//! The `claude --print` flag runs in text-completion mode: no tools, no file
//! access, no session persistence, no network calls. The CLI reads a prompt
//! from stdin and writes a response to stdout — it cannot modify the
//! filesystem or execute commands. This makes it inherently safe for use as
//! a completion backend without any additional sandboxing.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::debug;

use super::cli_common::{build_prompt, parse_inner_response};
use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, AiUsage, ClassifyAiError,
    ProviderCapabilities, cache_identity_with,
};
use crate::utils::utf8_prefix;

#[derive(Debug, thiserror::Error)]
pub enum ClaudeCliError {
    #[error("Failed to spawn claude CLI: {0}")]
    Spawn(String),
    #[error("claude CLI timed out after 10 minutes")]
    Timeout,
    #[error("claude CLI wait error: {0}")]
    Wait(String),
    #[error("claude CLI error: {0}")]
    Cli(String),
    #[error("Failed to parse claude CLI JSON output: {0}")]
    Parse(String),
}

impl ClassifyAiError for ClaudeCliError {
    fn ai_error_class(&self) -> AiErrorClass {
        match self {
            ClaudeCliError::Spawn(_) => AiErrorClass::Fatal,
            ClaudeCliError::Timeout => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Wait(_) => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Cli(_) => AiErrorClass::Fatal,
            ClaudeCliError::Parse(_) => AiErrorClass::Fatal,
        }
    }
}

pub struct ClaudeCliProvider {
    pub model: String,
    pub effort: Option<String>,
}

#[async_trait]
impl AiProvider for ClaudeCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);

        debug!("claude-cli prompt length: {} chars", prompt.len());

        let mut args = vec![
            "--print".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--no-session-persistence".to_string(),
        ];

        args.push("--model".to_string());
        args.push(self.model.clone());

        if let Some(effort) = &self.effort {
            args.push("--effort".to_string());
            args.push(effort.clone());
        }

        let mut child = Command::new("claude")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| ClaudeCliError::Spawn(e.to_string()))?;

        // Write prompt to stdin then close it
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        // 10-minute timeout per CLI call — a hung claude process won't block forever
        let output = timeout(Duration::from_secs(600), child.wait_with_output())
            .await
            .map_err(|_| ClaudeCliError::Timeout)?
            .map_err(|e| ClaudeCliError::Wait(e.to_string()))?;

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if !line.trim().is_empty() {
                    debug!("[claude-cli stderr] {}", line);
                }
            }
        }

        let raw = String::from_utf8_lossy(&output.stdout);

        if !output.status.success() {
            // Try to extract the actual error message from the JSON output.
            // The CLI emits a JSON object with is_error=true and the reason
            // in the "result" field even when it exits non-zero.
            if let Ok(outer) = serde_json::from_str::<Value>(&raw)
                && let Some(msg) = outer["result"].as_str()
            {
                return Err(ClaudeCliError::Cli(msg.to_string()).into());
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ClaudeCliError::Cli(format!(
                "exited with {}: {}",
                output.status,
                stderr.trim()
            ))
            .into());
        }
        let outer = parse_cli_output(&raw)?;

        if outer["is_error"].as_bool().unwrap_or(false) {
            return Err(ClaudeCliError::Cli(
                outer["result"]
                    .as_str()
                    .unwrap_or("unknown error")
                    .to_string(),
            )
            .into());
        }

        let result_text = outer["result"].as_str().unwrap_or("").trim().to_string();

        // Parse usage from the outer JSON
        let usage = parse_usage(&outer);

        // Parse the inner response — tool calls or content
        parse_inner_response("claude-cli", &result_text, usage)
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        let chars: usize = request
            .messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|c| c.len())
            .sum();
        chars / 4
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: context_window_for_model(&self.model),
        }
    }

    fn cache_identity(&self) -> String {
        cache_identity_with(&self.model, &[("effort", self.effort.as_deref())])
    }
}

fn parse_cli_output(raw: &str) -> Result<Value, ClaudeCliError> {
    serde_json::from_str(raw)
        .map_err(|e| ClaudeCliError::Parse(format!("{}\nRaw: {}", e, utf8_prefix(raw, 200))))
}

/// Pick the context window to advertise for a given model name. The value is
/// currently metadata only (no consumer gates on it), so the mapping is coarse.
/// Opus 4.7 ships with a 1M window by default via Claude Code, and any model
/// can be selected with the `[1m]` suffix to opt into the 1M variant.
/// Verified against Claude Code 2.1.132 for opus-4-7, sonnet-4-6, sonnet-4-6[1m],
/// haiku-4-5.
fn context_window_for_model(model: &str) -> usize {
    if model.contains("[1m]") || model.contains("opus-4-7") {
        1_000_000
    } else {
        200_000
    }
}

fn parse_usage(outer: &Value) -> Option<AiUsage> {
    let u = &outer["usage"];
    if u.is_null() {
        return None;
    }
    let input = u["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output = u["output_tokens"].as_u64().unwrap_or(0) as usize;
    let cache_read = u["cache_read_input_tokens"].as_u64().unwrap_or(0) as usize;
    let cache_write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0) as usize;
    // input_tokens arrives without the cached prefix, so the two cache
    // counts fold back in here.  cached_tokens is a breakdown of
    // prompt_tokens, and a consumer subtracts it to get uncached input.
    let total_input = input + cache_read + cache_write;
    Some(AiUsage {
        prompt_tokens: total_input,
        completion_tokens: output,
        total_tokens: total_input + output,
        cached_tokens: if cache_read > 0 {
            Some(cache_read)
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_preview_handles_multibyte_cutoff() {
        let raw = format!("{}🙂not-json", "a".repeat(199));

        let error = parse_cli_output(&raw).unwrap_err();

        assert!(matches!(&error, ClaudeCliError::Parse(_)));
        assert!(error.to_string().contains(&"a".repeat(199)));
    }

    #[test]
    fn test_parse_usage_folds_cache_counts_into_prompt_tokens() {
        let outer = serde_json::json!({
            "usage": {
                "input_tokens": 500,
                "output_tokens": 20,
                "cache_read_input_tokens": 19500,
                "cache_creation_input_tokens": 1000,
            }
        });

        let usage = parse_usage(&outer).unwrap();

        // Uncached input is prompt_tokens less the cached breakdown, so it
        // covers the fresh input and the prefix the model had to write.
        assert_eq!(usage.prompt_tokens, 21000);
        assert_eq!(usage.cached_tokens, Some(19500));
        assert_eq!(usage.total_tokens, 21020);
    }

    #[test]
    fn test_parse_usage_without_a_cache_read_reports_no_cached_tokens() {
        let outer = serde_json::json!({
            "usage": {
                "input_tokens": 500,
                "output_tokens": 20,
            }
        });

        let usage = parse_usage(&outer).unwrap();

        assert_eq!(usage.prompt_tokens, 500);
        assert_eq!(usage.cached_tokens, None);
    }
}
