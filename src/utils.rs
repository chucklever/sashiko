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

use regex::Regex;
use std::sync::OnceLock;

static KEY_REGEX: OnceLock<Regex> = OnceLock::new();
static URL_CRED_REGEX: OnceLock<Regex> = OnceLock::new();

/// Returns the longest prefix of `input` that is valid UTF-8 and no longer
/// than `max_bytes` bytes.
///
/// This is useful when text must fit a byte-limited preview. Direct string
/// slicing can panic when the byte limit falls inside a multi-byte character.
pub fn utf8_prefix(input: &str, max_bytes: usize) -> &str {
    let mut end = input.len().min(max_bytes);
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    &input[..end]
}

/// Redacts sensitive information from a string.
///
/// Specifically targets:
/// - API keys in query parameters (e.g., `key=AIza...`)
/// - Credentials in URLs (e.g., `https://user:pass@host`)
pub fn redact_secret(s: &str) -> String {
    let key_re =
        KEY_REGEX.get_or_init(|| Regex::new(r"(?i)(key|token|secret)=([a-zA-Z0-9_\-]+)").unwrap());

    let url_cred_re = URL_CRED_REGEX.get_or_init(|| Regex::new(r"://([^/:]+):([^/@]+)@").unwrap());

    let redacted_params = key_re.replace_all(s, "$1=[REDACTED]");
    let redacted_url = url_cred_re.replace_all(&redacted_params, "://[REDACTED]:[REDACTED]@");

    redacted_url.to_string()
}

/// Renders `s` for a single log line as its first `head` and last `tail`
/// characters with the elided count between them.  A parse that gives up
/// reports what it choked on, and both ends carry information: text that opens
/// with prose and closes with a JSON object is indistinguishable from text that
/// never emitted JSON when only the head is shown.  Newlines become `\n` so the
/// result stays one grep-able line, and the cuts land on character boundaries,
/// since slicing by byte offset panics inside a multi-byte character.
pub fn head_tail_snippet(s: &str, head: usize, tail: usize) -> String {
    let total = s.chars().count();
    let joined = if total <= head + tail {
        s.to_string()
    } else {
        let head_end = s.char_indices().nth(head).map_or(s.len(), |(i, _)| i);
        let tail_start = s
            .char_indices()
            .nth(total - tail)
            .map_or(s.len(), |(i, _)| i);
        format!(
            "{}...[{} chars elided]...{}",
            &s[..head_end],
            total - head - tail,
            &s[tail_start..]
        )
    };
    joined.replace('\n', "\\n").replace('\r', "\\r")
}

/// Cleans a JSON string by escaping unescaped control characters inside string literals.
///
/// This is particularly useful for parsing LLM-generated JSON, which sometimes
/// contains literal newlines or tabs inside string values instead of the
/// correct escape sequences (`\n`, `\t`, etc.).
pub fn clean_json_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_string = false;
    let mut escape = false;

    for c in input.chars() {
        if in_string {
            if escape {
                out.push('\\');
                out.push(c);
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                out.push(c);
                in_string = false;
            } else if c == '\n' {
                out.push_str("\\n");
            } else if c == '\r' {
                out.push_str("\\r");
            } else if c == '\t' {
                out.push_str("\\t");
            } else if c < '\x20' {
                use std::fmt::Write;
                write!(&mut out, "\\u{:04x}", c as u32).unwrap();
            } else {
                out.push(c);
            }
        } else {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
        }
    }

    // If the string ended while still in an escape sequence
    if escape {
        out.push('\\');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_prefix_preserves_character_boundaries() {
        assert_eq!(utf8_prefix("plain text", 5), "plain");
        assert_eq!(utf8_prefix("abc🙂def", 7), "abc🙂");
        assert_eq!(utf8_prefix("abc🙂def", 6), "abc");
        assert_eq!(utf8_prefix("🙂", 0), "");
        assert_eq!(utf8_prefix("short", 100), "short");
    }

    #[test]
    fn test_head_tail_snippet_keeps_both_ends() {
        let text = "prose about the patch\nmore prose\n{\"concerns\": []}";
        let snippet = head_tail_snippet(text, 10, 10);
        assert_eq!(snippet, "prose abou...[29 chars elided]...erns\": []}");
    }

    #[test]
    fn test_head_tail_snippet_passes_short_text_through() {
        assert_eq!(head_tail_snippet("a\nb", 10, 10), "a\\nb");
        assert_eq!(head_tail_snippet("", 10, 10), "");
    }

    #[test]
    fn test_head_tail_snippet_on_multibyte_boundary() {
        // Three-byte characters on both cuts: byte-offset slicing would panic
        let s = "☃☃☃☃☃☃";
        assert_eq!(head_tail_snippet(s, 2, 2), "☃☃...[2 chars elided]...☃☃");
    }

    #[test]
    fn test_redact_gemini_key() {
        let url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=AIzaSyD-12345";
        let redacted = redact_secret(url);
        assert_eq!(
            redacted,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=[REDACTED]"
        );
    }

    #[test]
    fn test_redact_git_credentials() {
        let url = "https://user:password123@github.com/torvalds/linux.git";
        let redacted = redact_secret(url);
        assert_eq!(
            redacted,
            "https://[REDACTED]:[REDACTED]@github.com/torvalds/linux.git"
        );
    }

    #[test]
    fn test_redact_mixed() {
        let s = "Error connecting to https://user:pass@host/api?key=secret_value";
        let redacted = redact_secret(s);
        assert_eq!(
            redacted,
            "Error connecting to https://[REDACTED]:[REDACTED]@host/api?key=[REDACTED]"
        );
    }

    #[test]
    fn test_no_secrets() {
        let s = "https://github.com/torvalds/linux.git";
        let redacted = redact_secret(s);
        assert_eq!(redacted, s);
    }

    #[test]
    fn test_clean_json_string() {
        let valid = r#"{"name": "test", "value": "a\nb"}"#;
        assert_eq!(clean_json_string(valid), valid);

        let invalid = "{\"name\": \"test\", \"value\": \"a\nb\"}";
        let fixed = r#"{"name": "test", "value": "a\nb"}"#;
        assert_eq!(clean_json_string(invalid), fixed);

        let invalid_tab = "{\"key\": \"val\tue\"}";
        let fixed_tab = r#"{"key": "val\tue"}"#;
        assert_eq!(clean_json_string(invalid_tab), fixed_tab);

        let structural = "{\n  \"key\": \"value\"\n}";
        assert_eq!(clean_json_string(structural), structural);
    }
}
