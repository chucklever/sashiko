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

//! Prompt construction and response parsing shared by the CLI-backed providers
//! (claude-cli, codex-cli, copilot-cli, devin-cli, kiro-cli).
//!
//! What these providers share is the wire format: each one hands the
//! subprocess the prompt `build_prompt()` renders and feeds the model's reply
//! back through `parse_inner_response()`. The three that take the prompt on
//! stdin also share `write_prompt_and_wait()`. What differs is how the reply
//! comes out — claude-cli reads one JSON document from stdout, codex-cli and
//! copilot-cli read a JSONL event stream, devin-cli passes the prompt as a
//! temp-file path with stdin closed, and kiro-cli speaks ACP over a session.
//!
//! Functions that log take the caller's provider name, since the message
//! otherwise names whichever provider the code happens to live next to.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use serde_json::Value;
use tokio::process::{Child, Command};
use tracing::{debug, warn};

use crate::ai::{AiRequest, AiResponse, AiRole, AiUsage, ToolCall};

/// How long a CLI may produce nothing at all before it counts as hung.
///
/// This bounds silence, not runtime.  A review at high reasoning effort runs
/// the CLI's own agent loop, which reads files and reasons for as long as the
/// patch demands, so runtime says nothing about health: measured runs against
/// `codex exec` span from seconds to past eleven minutes, and the longest were
/// the ones doing the most work.  A run that is reading files reports each
/// step as it finishes and goes quiet for a minute at most, but one that
/// spends a whole turn reasoning emits nothing until the turn ends, and 350
/// seconds is the longest such silence measured across 79 runs.  The value
/// clears that with room to spare, because killing a live review costs the
/// whole review while a hung one costs only the wait.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(900);

/// Build the full text prompt from the AiRequest.
/// Embeds system prompt, conversation history, tool definitions, and instructions.
pub fn build_prompt(request: &AiRequest) -> String {
    let mut out = String::new();

    // System prompt
    if let Some(sys) = &request.system {
        out.push_str("<system>\n");
        out.push_str(sys);
        out.push_str("\n</system>\n\n");
    }

    // Conversation history
    for msg in &request.messages {
        match &msg.role {
            AiRole::System => {
                // Already handled above; skip embedded system messages
            }
            AiRole::User => {
                out.push_str("<user>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</user>\n\n");
            }
            AiRole::Assistant => {
                out.push_str("<assistant>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        out.push_str(&format!(
                            "<tool_call id=\"{}\" name=\"{}\">\n{}\n</tool_call>\n",
                            call.id, call.function_name, call.arguments
                        ));
                    }
                }
                out.push_str("</assistant>\n\n");
            }
            AiRole::Tool => {
                let id = msg.tool_call_id.as_deref().unwrap_or("?");
                out.push_str(&format!("<tool_result id=\"{}\">\n", id));
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</tool_result>\n\n");
            }
        }
    }

    // Tool definitions and response instructions
    if let Some(tools) = &request.tools
        && !tools.is_empty()
    {
        out.push_str("<available_tools>\n");
        for tool in tools {
            out.push_str(&format!(
                "- name: {}\n  description: {}\n  parameters: {}\n\n",
                tool.name, tool.description, tool.parameters
            ));
        }
        out.push_str("</available_tools>\n\n");
        out.push_str(
            "RESPONSE FORMAT: You MUST respond with a SINGLE valid JSON object only (no markdown, no explanation).\n\
             To call tools: {\"tool_calls\": [{\"id\": \"c1\", \"function_name\": \"TOOL_NAME\", \"arguments\": {ARGS}}, {\"id\": \"c2\", \"function_name\": \"OTHER_TOOL\", \"arguments\": {ARGS2}}]}\n\
             Put ALL tool calls in ONE tool_calls array. Do NOT output multiple JSON objects.\n\
             For your final answer: {\"content\": \"YOUR RESPONSE\"}\n\
             Do not mix both. Output exactly one JSON object.\n",
        );
    } else if let Some(instruction) = request
        .response_format
        .as_ref()
        .and_then(|f| f.format_json_schema_instruction())
    {
        out.push_str(&instruction);
        out.push('\n');
    }

    out
}

/// What a CLI had produced when the idle deadline cut it off.
///
/// A killed run is the one case where the output is worth keeping even though
/// the call failed: for a JSONL provider it names the last step the CLI
/// finished, which is the only account of what it was doing when it stopped.
#[derive(Debug)]
pub struct IdleOutput {
    pub after: Duration,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Names the half of a CLI call that failed, so the caller can label it.
#[derive(Debug)]
pub enum CliIoError {
    /// The prompt could not be written to the child's stdin.
    Write(std::io::Error),
    /// An output pipe could not be read.
    Read(std::io::Error),
    /// The child could not be waited on.
    Wait(std::io::Error),
    /// The child went quiet for longer than the idle deadline and was killed.
    Idle(IdleOutput),
}

impl std::fmt::Display for CliIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliIoError::Write(e) => write!(f, "stdin write failed: {e}"),
            CliIoError::Read(e) => write!(f, "output read failed: {e}"),
            CliIoError::Wait(e) => write!(f, "wait error: {e}"),
            CliIoError::Idle(idle) => write!(
                f,
                "produced no output for {}s and was killed after {} bytes",
                idle.after.as_secs(),
                idle.stdout.len() + idle.stderr.len()
            ),
        }
    }
}

/// Spawn a CLI in a process group of its own, so a timeout can take down the
/// whole tree.
///
/// `codex` and `copilot` install as node wrappers that run the real binary as
/// a grandchild.  Killing the child alone leaves that grandchild attached to
/// the same pipes and billing against the same subscription until it notices
/// they are gone, which measured at over a minute past the kill.  A group of
/// its own also keeps the CLI off this process's group, so a signal aimed at
/// the daemon's terminal does not reach a review mid-flight.
///
/// `write_prompt_and_wait()` signals the group this establishes; a child
/// spawned any other way must not be passed to it.
pub fn spawn_cli(cmd: &mut Command) -> std::io::Result<Child> {
    cmd.process_group(0).kill_on_drop(true).spawn()
}

/// Signal the whole group, having spawned its leader with `process_group(0)`
/// so that the group holds the CLI and its descendants and nothing else.
fn kill_group(pgid: i32) {
    // SAFETY: kill(2) touches no memory of this process.  A negative pid
    // addresses the group led by `pgid`, which `spawn_cli` created for the
    // child alone, so the signal cannot reach the daemon or its siblings.
    unsafe { libc::kill(-pgid, libc::SIGKILL) };
}

/// Read from a pipe that may already have reached EOF, in a form `select!` can
/// poll.  A half that is done never completes again, leaving the other half to
/// drive the loop.
async fn read_open<R>(pipe: &mut Option<R>, buf: &mut [u8]) -> std::io::Result<usize>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;

    match pipe {
        Some(pipe) => pipe.read(buf).await,
        None => std::future::pending().await,
    }
}

/// Feed `prompt` to a spawned CLI while draining its output, and return what
/// the child produced.
///
/// The write runs concurrently with the wait rather than before it.  A patch
/// review prompt runs to hundreds of KB, and a child that emits more than a
/// pipe buffer of output before consuming all of stdin blocks on its own full
/// stdout while the writer blocks on its full stdin.  `wait_with_output()`
/// drains both output pipes, so the child keeps making progress as the prompt
/// goes in.  Closing stdin afterwards is what puts these CLIs into
/// non-interactive mode, so the handle is taken here and dropped when the
/// write finishes.
///
/// A child that stops reading early -- it rejected the invocation, or it has
/// all the input it needs -- closes the pipe, and the write then fails with
/// EPIPE.  That is not a failure of the call: the child ran to completion
/// either way, and whatever it wrote is in the returned output, whether that
/// is the answer or the reason it quit.  Report the EPIPE and hand the output
/// back.
///
/// `idle_timeout` bounds silence rather than runtime, so a CLI that keeps
/// reporting progress runs as long as the work takes while one that has
/// wedged is cut off.  Reaching the deadline kills the child's process group
/// -- `spawn_cli` must have established it -- and returns what the CLI had
/// produced up to that point.  Killing rather than dropping is what releases
/// a write still blocked on the child's stdin.
pub async fn write_prompt_and_wait(
    mut child: Child,
    prompt: String,
    idle_timeout: Duration,
) -> std::result::Result<std::process::Output, CliIoError> {
    use tokio::io::AsyncWriteExt;

    let pgid = child.id().map(|id| id as i32);
    let mut stdin = child.stdin.take();
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let prompt_bytes = prompt.into_bytes();

    let write = async {
        if let Some(mut stdin) = stdin.take() {
            stdin.write_all(&prompt_bytes).await?;
            stdin.flush().await?;
        }
        Ok::<(), std::io::Error>(())
    };

    // Draining both pipes here is what lets the write run concurrently: a
    // child that fills its stdout before consuming all of stdin would
    // otherwise block on its own full pipe while the writer blocks on the
    // child's.
    let drain = async {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut out_buf = [0u8; 8192];
        let mut err_buf = [0u8; 8192];

        while stdout.is_some() || stderr.is_some() {
            let step = tokio::time::timeout(idle_timeout, async {
                tokio::select! {
                    n = read_open(&mut stdout, &mut out_buf) => (true, n),
                    n = read_open(&mut stderr, &mut err_buf) => (false, n),
                }
            })
            .await;

            let (is_stdout, read) = match step {
                Ok(step) => step,
                Err(_) => {
                    if let Some(pgid) = pgid {
                        kill_group(pgid);
                    }
                    // The group signal reaches the child too, but only where
                    // the child leads that group.  Signal it directly as
                    // well, so the wait below cannot block on a child the
                    // group kill did not cover.
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(CliIoError::Idle(IdleOutput {
                        after: idle_timeout,
                        stdout: out,
                        stderr: err,
                    }));
                }
            };

            match read.map_err(CliIoError::Read)? {
                0 if is_stdout => stdout = None,
                0 => stderr = None,
                n if is_stdout => out.extend_from_slice(&out_buf[..n]),
                n => err.extend_from_slice(&err_buf[..n]),
            }
        }

        let status = child.wait().await.map_err(CliIoError::Wait)?;
        Ok(std::process::Output {
            status,
            stdout: out,
            stderr: err,
        })
    };

    let (written, output) = tokio::join!(write, drain);
    let output = output?;

    match written {
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
            debug!("CLI closed stdin before the prompt finished: {}", e);
            Ok(output)
        }
        Err(e) => Err(CliIoError::Write(e)),
        Ok(()) => Ok(output),
    }
}

/// Parse a provider's response body into an AiResponse.
///
/// `provider` names the caller for the log messages below ("codex-cli",
/// "kiro-cli", and so on).
pub fn parse_inner_response(
    provider: &str,
    text: &str,
    usage: Option<AiUsage>,
) -> Result<AiResponse> {
    // Try extracting JSON (might be in a markdown code block)
    let json_str = extract_json(text);

    if let Ok(v) = serde_json::from_str::<Value>(&json_str) {
        return parse_single_json(&v, &json_str, usage);
    }

    // Try JSONL: multiple JSON objects on separate lines (model sometimes emits
    // separate tool_calls objects per line instead of one combined object)
    let mut merged_tool_calls: Vec<ToolCall> = Vec::new();
    let mut had_json = false;
    for line in json_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            had_json = true;
            if let Some(calls) = v["tool_calls"].as_array() {
                for c in calls {
                    if let Some(tc) = parse_tool_call(c) {
                        merged_tool_calls.push(tc);
                    }
                }
            }
        }
    }

    // Across the whole merge, not per line, so that a model restarting its
    // numbering in each line does not repeat an id
    assign_unique_ids(&mut merged_tool_calls);

    if !merged_tool_calls.is_empty() {
        debug!(
            "{}: merged {} tool calls from JSONL response",
            provider,
            merged_tool_calls.len()
        );
        return Ok(AiResponse {
            content: None,
            thought: None,
            thought_signature: None,
            tool_calls: Some(merged_tool_calls),
            usage,
            truncated: false,
        });
    }

    if had_json {
        // Had valid JSON lines but no tool calls — return original text as content
        // (json_str from extract_json may be mangled if text had multiple objects)
        return Ok(AiResponse {
            content: Some(text.to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            usage,
            truncated: false,
        });
    }

    // Not parseable as JSON — return raw text.  Consumers vary in how much
    // they salvage: a stage validator brace-scans the content and usually
    // recovers the object this extractor declined to guess at, so the warning
    // is not by itself a failure.  The snippet tells the two apart -- text that
    // ends in a JSON object was recovered downstream, text that holds none was
    // not -- without re-running the call to find out.
    warn!(
        chars = text.len(),
        snippet = %crate::utils::head_tail_snippet(text, 200, 200),
        "{provider} response not valid JSON, returning as raw content"
    );
    Ok(AiResponse {
        content: Some(text.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

/// Convert one entry of a `tool_calls` array into a `ToolCall`.
///
/// A missing "id" leaves the field empty for `assign_unique_ids` to fill in.
fn parse_tool_call(c: &Value) -> Option<ToolCall> {
    let id = c["id"].as_str().unwrap_or_default().to_string();
    let name = c["function_name"].as_str()?.to_string();
    let args = c["arguments"].clone();
    Some(ToolCall {
        id,
        function_name: name,
        arguments: args,
        thought_signature: None,
    })
}

/// Give every call in a batch a distinct id, keeping the ones the model
/// supplied.
///
/// SessionRunner records one AiMessage per tool result and carries the call's
/// id over as tool_call_id, which build_prompt renders as the `id` of a
/// `<tool_result>` block. Two calls sharing an id leave the model unable to
/// tell which output came from which call, so it can read a file's contents as
/// the git log it also asked for. Models drop the "id" the prompt asks for, and
/// one that splits its batch across several JSON objects tends to number each
/// object from c1 again, so neither the supplied ids nor their absence can be
/// relied on.
///
/// A call keeps its supplied id unless an earlier call in the batch already
/// claimed it. Every other call takes the lowest cN not claimed by any of them.
fn assign_unique_ids(calls: &mut [ToolCall]) {
    let mut taken: HashSet<String> = HashSet::new();
    let mut unnamed: Vec<usize> = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        if call.id.is_empty() || !taken.insert(call.id.clone()) {
            unnamed.push(i);
        }
    }

    let mut next = 1;
    for i in unnamed {
        calls[i].id = loop {
            let candidate = format!("c{next}");
            next += 1;
            if taken.insert(candidate.clone()) {
                break candidate;
            }
        };
    }
}

fn parse_single_json(v: &Value, json_str: &str, usage: Option<AiUsage>) -> Result<AiResponse> {
    // Tool calls?
    if let Some(calls) = v["tool_calls"].as_array() {
        let mut tool_calls: Vec<ToolCall> = calls.iter().filter_map(parse_tool_call).collect();
        assign_unique_ids(&mut tool_calls);

        if !tool_calls.is_empty() {
            return Ok(AiResponse {
                content: None,
                thought: None,
                thought_signature: None,
                tool_calls: Some(tool_calls),
                usage,
                truncated: false,
            });
        }
    }

    // Content field?
    if let Some(content) = v["content"].as_str() {
        return Ok(AiResponse {
            content: Some(content.to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            usage,
            truncated: false,
        });
    }

    // Any other JSON — return it as content string (e.g. {"concerns": [...]})
    Ok(AiResponse {
        content: Some(json_str.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

/// Extract JSON from text that may be wrapped in markdown fences.
/// Returns the content of the first fenced block that parses as JSON, else the
/// first json-tagged block, else the first block, else the original text trimmed.
/// Does NOT try to find outermost braces — that can silently produce invalid JSON
/// when the text contains multiple objects (e.g. JSONL), which the JSONL fallback
/// in parse_inner_response handles better.
fn extract_json(text: &str) -> String {
    // A reasoning summary or a quoted hunk often arrives fenced ahead of the
    // JSON, so no single fence is the answer. The json tag breaks the tie for
    // a JSONL body, which parses only line by line and so never wins outright.
    let normalized = text.replace("\r\n", "\n");
    let mut json_tagged: Option<&str> = None;
    let mut first_body: Option<&str> = None;
    let mut rest = normalized.as_str();
    while let Some(start) = rest.find("```") {
        let after_ticks = &rest[start + 3..];
        let Some(nl) = after_ticks.find('\n') else {
            break;
        };
        let tag = after_ticks[..nl].trim();
        let body = &after_ticks[nl + 1..];
        // A fence tag is one word; anything else is prose that happened to
        // follow a backtick run, so keep looking.
        if tag.contains(char::is_whitespace) {
            rest = after_ticks;
            continue;
        }
        let Some(end) = body.find("\n```") else { break };
        let inner = body[..end].trim();
        if serde_json::from_str::<Value>(inner).is_ok() {
            return inner.to_string();
        }
        if tag.get(..4).is_some_and(|t| t.eq_ignore_ascii_case("json")) {
            json_tagged.get_or_insert(inner);
        }
        first_body.get_or_insert(inner);
        rest = &body[end + 4..];
    }
    json_tagged
        .or(first_body)
        .unwrap_or(normalized.trim())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiRequest, AiResponseFormat, AiRole, AiTool};
    use serde_json::json;

    fn make_request(messages: Vec<AiMessage>) -> AiRequest {
        AiRequest {
            system: None,
            messages,
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
            workspace: None,
        }
    }

    fn simple_user_msg() -> Vec<AiMessage> {
        vec![AiMessage {
            role: AiRole::User,
            content: Some("hi".to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            tool_call_id: None,
        }]
    }

    #[test]
    fn test_build_prompt_json_format_without_tools() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json { schema: None });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("ONLY a valid JSON object"));
        assert!(!prompt.contains("tool_calls"));
    }

    #[test]
    fn test_build_prompt_json_format_with_schema() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json {
            schema: Some(
                json!({"type": "object", "properties": {"selected_prompts": {"type": "array"}}}),
            ),
        });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("selected_prompts"));
        assert!(prompt.contains("matching this schema"));
    }

    #[test]
    fn test_build_prompt_with_tools_includes_format() {
        let mut req = make_request(simple_user_msg());
        req.tools = Some(vec![AiTool {
            name: "git_log".to_string(),
            description: "Show git log".to_string(),
            parameters: json!({"type": "object"}),
        }]);

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("tool_calls"));
        assert!(prompt.contains("<available_tools>"));
    }

    #[test]
    fn test_build_prompt_text_format_no_instruction() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Text);

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }

    #[test]
    fn test_build_prompt_no_format_no_instruction() {
        let req = make_request(simple_user_msg());

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }

    #[test]
    fn test_parse_tool_calls_json() {
        let text = r#"{"tool_calls":[{"id":"c1","function_name":"read_file","arguments":{"path":"README.md"}}]}"#;
        let resp = parse_inner_response("test-cli", text, None).unwrap();
        assert!(resp.tool_calls.is_some());
        let calls = resp.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn test_parse_plain_content() {
        let text = r#"{"content":"No issues found in this patch."}"#;
        let resp = parse_inner_response("test-cli", text, None).unwrap();
        assert_eq!(
            resp.content.as_deref(),
            Some("No issues found in this patch.")
        );
        assert!(resp.tool_calls.is_none());
    }

    fn spawn_sh(script: &str) -> tokio::process::Child {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        spawn_cli(&mut cmd).unwrap()
    }

    #[tokio::test]
    async fn write_prompt_and_wait_keeps_the_output_of_a_child_that_ignores_stdin() {
        // The child exits 0 without draining stdin, so a prompt too large for
        // the pipe buffer ends in EPIPE even though the run succeeded.
        let child = spawn_sh("echo done; exit 0");

        let output = write_prompt_and_wait(child, "x".repeat(4 * 1024 * 1024), IDLE_TIMEOUT)
            .await
            .expect("EPIPE on a successful run must not discard the output");
        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "done");
    }

    #[tokio::test]
    async fn a_child_that_keeps_emitting_outlives_the_idle_window() {
        // Total runtime is six times the idle window; every gap within it is
        // shorter.  This is the shape of a review at high reasoning effort,
        // and the wall-clock cap this replaces killed exactly these runs.
        let child = spawn_sh("for i in 1 2 3 4 5 6 7 8 9 10 11 12; do echo tick; sleep 0.25; done");

        let output = write_prompt_and_wait(child, String::new(), Duration::from_millis(500))
            .await
            .expect("steady output must hold the deadline open");
        assert!(output.status.success());
        assert_eq!(output.stdout.split(|b| *b == b'\n').count() - 1, 12);
    }

    #[tokio::test]
    async fn a_silent_child_is_killed_at_the_idle_deadline() {
        let child = spawn_sh("sleep 60");

        let started = tokio::time::Instant::now();
        let err = write_prompt_and_wait(child, String::new(), Duration::from_millis(300))
            .await
            .expect_err("silence past the deadline must not wait out the child");
        assert!(matches!(err, CliIoError::Idle(_)), "got {err:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn the_idle_kill_reaches_a_grandchild() {
        // `codex` and `copilot` are node wrappers that run the real binary as
        // a grandchild.  Killing the wrapper alone leaves that grandchild
        // billing against the same subscription.
        // Both sleeps outlast the test by far, so nothing here ends on its
        // own: the grandchild is gone at the end only if the kill reached it.
        let child = spawn_sh("sleep 600 & echo $!; sleep 600");

        let err = tokio::time::timeout(
            Duration::from_secs(10),
            write_prompt_and_wait(child, String::new(), Duration::from_millis(300)),
        )
        .await
        .expect("the deadline must kill the child, not wait it out")
        .expect_err("silence past the deadline must fail");
        let CliIoError::Idle(idle) = err else {
            panic!("got {err:?}")
        };

        // The grandchild's pid is the one line the script emits before going
        // quiet.
        let pid = String::from_utf8_lossy(&idle.stdout)
            .trim()
            .parse::<i32>()
            .unwrap();
        for _ in 0..50 {
            if unsafe { libc::kill(pid, 0) } != 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // Do not leave it running if the assert is about to fail.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        panic!("grandchild {pid} survived the process-group kill");
    }

    #[test]
    fn test_parse_tool_calls_without_ids_get_distinct_ids() {
        let text = r#"{"tool_calls":[{"function_name":"git_log","arguments":{}},{"function_name":"read_file","arguments":{"path":"a"}}]}"#;
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_parse_jsonl_tool_calls_get_distinct_ids() {
        let text = "{\"tool_calls\":[{\"function_name\":\"git_log\",\"arguments\":{}}]}\n\
                    {\"tool_calls\":[{\"function_name\":\"read_file\",\"arguments\":{}}]}";
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_parse_tool_calls_mixed_ids_stay_distinct() {
        // The supplied id collides with the position-derived fallback the
        // second call would otherwise get.
        let text = r#"{"tool_calls":[{"id":"c2","function_name":"git_log","arguments":{}},{"function_name":"read_file","arguments":{"path":"a"}}]}"#;
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "c2");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_parse_jsonl_repeated_supplied_ids_stay_distinct() {
        // A model that splits its batch across lines tends to restart its
        // numbering in each one.
        let text = "{\"tool_calls\":[{\"id\":\"c1\",\"function_name\":\"git_log\",\"arguments\":{}}]}\n\
                    {\"tool_calls\":[{\"id\":\"c1\",\"function_name\":\"read_file\",\"arguments\":{}}]}";
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_parse_tool_calls_keep_supplied_ids() {
        let text = r#"{"tool_calls":[{"id":"call_a","function_name":"git_log","arguments":{}},{"id":"call_b","function_name":"read_file","arguments":{}}]}"#;
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls[0].id, "call_a");
        assert_eq!(calls[1].id, "call_b");
    }

    #[test]
    fn test_extract_json_unwraps_any_fence_tag() {
        for tag in ["json", "JSON", "jsonc", "json5", ""] {
            let text = format!("```{tag}\n{{\"content\":\"ok\"}}\n```");
            let resp = parse_inner_response("test-cli", &text, None).unwrap();
            assert_eq!(resp.content.as_deref(), Some("ok"), "tag {tag:?}");
        }
    }

    #[test]
    fn test_extract_json_skips_a_leading_non_json_fence() {
        // Reasoning summaries and quoted hunks arrive fenced ahead of the
        // answer, so the JSON block is not always the first fence.
        let text = "Here is the offending hunk:\n\
                    ```c\n\
                    int x = 1;\n\
                    ```\n\n\
                    ```json\n\
                    {\"tool_calls\":[{\"id\":\"c1\",\"function_name\":\"read_file\",\"arguments\":{}}]}\n\
                    ```";
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "c1");
    }

    #[test]
    fn test_extract_json_prefers_a_json_tagged_fence() {
        // A JSONL body does not parse as a single value, so the tag is what
        // picks it over the reasoning block ahead of it.
        let text = "```text\n\
                    thinking out loud\n\
                    ```\n\n\
                    ```json\n\
                    {\"tool_calls\":[{\"id\":\"c1\",\"function_name\":\"git_log\",\"arguments\":{}}]}\n\
                    {\"tool_calls\":[{\"id\":\"c2\",\"function_name\":\"read_file\",\"arguments\":{}}]}\n\
                    ```";
        let calls = parse_inner_response("test-cli", text, None)
            .unwrap()
            .tool_calls
            .unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_parse_raw_text_fallback() {
        let text = "This is not JSON at all.";
        let resp = parse_inner_response("test-cli", text, None).unwrap();
        assert_eq!(resp.content.as_deref(), Some(text));
    }
}
