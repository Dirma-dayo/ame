use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use crate::state::PetState;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Parses a plain-text example file into few-shot (lead, assistant) pairs.
/// Expected format: blocks separated by a blank line, each with either a
/// "User:" line or a "System:" line, plus a "Petto:" line.
///
/// Assistant lines are wrapped into the exact `{"reply": ..., "mood": ...}`
/// envelope the system prompt requires (see `wrap_as_llm_json`), rather
/// than staying as bare text. Otherwise the few-shot examples silently
/// contradict the system prompt's JSON-only instruction — the model sees
/// "assistant turns look like plain text" right next to "you MUST respond
/// with ONLY a JSON object", and a small model tends to drift toward
/// whichever pattern is freshest in context (the examples), which is
/// exactly what was producing the un-parseable raw output.
pub fn load_example_dialogue(path: impl AsRef<Path>) -> Result<Vec<ChatMessage>> {
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Couldn't read example file at {:?}", path.as_ref()))?;

    let mut messages = Vec::new();

    for block in text.split("\n\n") {
        let mut lead_line = None; // either "User:" or "System:"
        let mut assistant_line = None;

        for line in block.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("User:") {
                lead_line = Some(ChatMessage::user(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("System:") {
                lead_line = Some(ChatMessage::system(rest.trim()));
            } else if let Some(rest) = line.strip_prefix("Petto:") {
                assistant_line = Some(rest.trim().to_string());
            }
        }

        if let (Some(lead), Some(a)) = (lead_line, assistant_line) {
            messages.push(lead);
            messages.push(ChatMessage::assistant(wrap_as_llm_json(&a)));
        }
    }

    Ok(messages)
}

/// The structured shape the model's replies are parsed into. `sticker`
/// is optional — Ame only includes it when she wants to send one (see
/// the system prompt in app.rs), and `#[serde(default)]` means it's
/// simply `None` for every reply that omits or nulls the field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmReply {
    pub reply: ReplyContent,
    #[serde(default)]
    pub mood: String,
    #[serde(default)]
    pub sticker: Option<String>,
}

/// Wraps a raw example line into the `{"reply": ..., "mood": "neutral"}`
/// envelope. If the line already looks like a JSON array (our
/// multi-bubble examples, e.g. `["hi.", "hi again."]`), it's parsed and
/// re-emitted as the `reply` array rather than being nested as a raw
/// string, so the model sees the real target shape either way.
fn wrap_as_llm_json(raw: &str) -> String {
    let trimmed = raw.trim();

    let reply_value: serde_json::Value = if trimmed.starts_with('[') {
        serde_json::from_str(trimmed)
            .unwrap_or_else(|_| serde_json::Value::String(trimmed.to_string()))
    } else {
        serde_json::Value::String(trimmed.to_string())
    };

    serde_json::json!({ "reply": reply_value, "mood": "neutral" }).to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

/// `reply` from the model's JSON can be either a single string (the
/// normal case) or an array of strings (an occasional "double/triple
/// texting" burst of short back-to-back messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReplyContent {
    Multi(Vec<String>),
    Single(String),
}

impl ReplyContent {
    /// Flattens into an ordered list of bubbles to display. A plain
    /// string collapses into one bubble; an array becomes a burst,
    /// with empty/whitespace-only entries dropped.
    pub fn into_bubbles(self) -> Vec<String> {
        match self {
            ReplyContent::Multi(items) => items
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            ReplyContent::Single(text) => vec![text],
        }
    }
}

/// Robustly extracts an `LlmReply` from whatever the model actually sent
/// back. Local models don't always follow the requested JSON shape
/// exactly — sometimes it's a bare `["msg1", "msg2"]` with no envelope,
/// sometimes it's plain text, and sometimes there's trailing garbage
/// after an otherwise-valid value (extra brackets/quotes tacked on).
///
/// Strategy: find the earliest `{` or `[` in the raw text, then use a
/// streaming JSON deserializer that reads exactly one complete value and
/// stops — ignoring anything that follows, instead of requiring the
/// whole remaining string to be valid JSON. Then interpret whatever
/// value came out (object / array / bare string) into an `LlmReply`.
pub fn parse_llm_reply(raw: &str) -> LlmReply {
    if let Some(start) = find_json_start(raw) {
        let slice = &raw[start..];
        let mut stream = serde_json::Deserializer::from_str(slice).into_iter::<serde_json::Value>();
        if let Some(Ok(value)) = stream.next() {
            return llm_reply_from_value(value, raw);
        }
    }

    LlmReply {
        reply: ReplyContent::Single(raw.trim().to_string()),
        mood: "neutral".to_string(),
        sticker: None,
    }
}

fn find_json_start(raw: &str) -> Option<usize> {
    let brace = raw.find('{');
    let bracket = raw.find('[');
    match (brace, bracket) {
        (Some(b), Some(k)) => Some(b.min(k)),
        (Some(b), None) => Some(b),
        (None, Some(k)) => Some(k),
        (None, None) => None,
    }
}

fn llm_reply_from_value(value: serde_json::Value, raw: &str) -> LlmReply {
    match value {
        serde_json::Value::Object(_) => serde_json::from_value(value).unwrap_or_else(|_| LlmReply {
            reply: ReplyContent::Single(raw.trim().to_string()),
            mood: "neutral".to_string(),
            sticker: None,
        }),
        serde_json::Value::Array(items) => {
            let strings: Vec<String> = items
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            LlmReply {
                reply: ReplyContent::Multi(strings),
                mood: "neutral".to_string(),
                sticker: None,
            }
        }
        serde_json::Value::String(s) => LlmReply {
            reply: ReplyContent::Single(s),
            mood: "neutral".to_string(),
            sticker: None,
        },
        other => LlmReply {
            reply: ReplyContent::Single(other.to_string()),
            mood: "neutral".to_string(),
            sticker: None,
        },
    }
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}
/// Default Ollama endpoint, used whenever the proxy field is empty.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8080";

pub struct OllamaClient {
    base_url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let body = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        };

        let response: ChatResponse = ureq::post(&format!("{}/api/chat", self.base_url))
            .send_json(&body)
            .context("Couldn't reach Ollama — is `ollama serve` running?")?
            .into_json()
            .context("Couldn't parse Ollama response")?;

        Ok(response.message.content)
    }
}

pub fn build_llm_messages(
    persona: &str,
    examples: &[ChatMessage],
    history: &[ChatMessage],
    state: &PetState,
    user_message: &str,
) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::system(persona)];
    messages.extend_from_slice(examples);
    messages.extend_from_slice(history);
    messages.push(ChatMessage::system(state.state_prompt()));
    messages.push(ChatMessage::user(user_message));
    messages
}

pub fn list_models(base_url: &str) -> Result<Vec<String>> {
    let response: TagsResponse =
        ureq::get(&format!("{}/api/tags", base_url))
            .call()
            .context("Couldn't reach Ollama")?
            .into_json()
            .context("Couldn't parse model list")?;

    Ok(response.models.into_iter().map(|m| m.name).collect())
}

pub struct LlmWorker {
    tx: Sender<Vec<ChatMessage>>,
    rx: Receiver<Result<String>>,
}

impl LlmWorker {
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<Vec<ChatMessage>>();
        let (res_tx, res_rx) = mpsc::channel::<Result<String>>();

        let model = model.into();
        let base_url = base_url.into();

        thread::spawn(move || {
            let client = OllamaClient::new(model, base_url);

            while let Ok(messages) = req_rx.recv() {
                let result = client.chat(&messages);
                if res_tx.send(result).is_err() {
                    break;
                }
            }
        });

        Self { tx: req_tx, rx: res_rx }
    }
    // ask()/poll() unchanged

    pub fn ask(&self, messages: Vec<ChatMessage>) {
        let _ = self.tx.send(messages);
    }

    pub fn poll(&self) -> Option<Result<String>> {
        self.rx.try_recv().ok()
    }
}