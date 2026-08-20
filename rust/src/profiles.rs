//! Per-model architecture profiles.
//!
//! Different model families were trained with different prompt formats,
//! system prompts, and sampling defaults.  Forcing one format on all of
//! them produces garbage — Ollama gets this right by shipping a
//! model-specific profile per GGUF (Modelfile).
//!
//! Each [`ModelProfile`] holds:
//!   - the chat-template family (controls how system/user/assistant wrap)
//!   - a default system prompt (used if the caller passes empty)
//!   - default sampling params (temperature, top_k, top_p)
//!   - stop tokens
//!   - whether the assistant turn should open with a reasoning tag
//!   - whether to do raw text continuation (no chat template)
//!
//! `resolve_profile` matches an architecture + (optional) model name to
//! a profile.  Unknown architectures fall back to a sensible default
//! derived from `general.architecture`.

use crate::model::gguf::GGUFValue;
use std::collections::HashMap;

/// Sampling defaults that work well for each family.
#[derive(Debug, Clone, Copy)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

/// One entry in the stop-token list.
#[derive(Debug, Clone)]
pub struct StopToken(pub usize, pub &'static str);

/// A profile describes how to talk to one model family.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Short stable name used by `--profile`.
    pub name: &'static str,
    /// Human description shown to the user.
    pub description: &'static str,
    /// Which GGUF architectures this profile applies to.
    pub architectures: &'static [&'static str],
    /// Default system prompt.  Used if the caller passes empty.
    pub default_system: &'static str,
    /// Sampling defaults.  The user can still override on the CLI.
    pub sampling: SamplingDefaults,
    /// Tokens that should stop generation.
    pub stop_tokens: &'static [StopToken],
    /// The assistant turn should open with a reasoning tag like
    /// `` for Ornith/Qwen3.5.  When true, we add
    /// `\n<think>\n` after `<|im_start|>assistant\n`.
    pub opens_with_thinking: bool,
    /// Use raw text continuation (no chat template wrapping).  Used for
    /// base-model-only models that have no instruction tuning.
    pub raw_continuation: bool,
}

/// The Ornith profile — a Qwen3.5 reasoning model.
///
/// Captured by tokenising Ollama's actual prompt (see `/tmp/ollama_debug.log`)
/// and `ollama show ornith:9b`.  Defaults match Ollama exactly so the model
/// sees the same context it was trained on.
pub const ORNITH_PROFILE: ModelProfile = ModelProfile {
    name: "ornith",
    description: "Ornith / Qwen3.5 reasoning model (Ollama-compatible)",
    architectures: &["qwen35", "qwen35moe", "qwen36", "ornith"],
    default_system:
        "You are Ornith, an open-source agentic coding assistant. \
         Think step by step in a reasoning block, then act. \
         Use the provided tools when they help. \
         Be concise, correct, and direct: write working code and \
         explain only what is non-obvious.",
    sampling: SamplingDefaults {
        temperature: 0.6,
        top_k: 20,
        top_p: 0.95,
        repeat_penalty: 1.1,
    },
    stop_tokens: &[
        StopToken(248046, "<|im_end|>"),       // Qwen ChatML end
        StopToken(248044, "<|endoftext|>"),    // GPT-2 EOS
    ],
    opens_with_thinking: true,
    raw_continuation: false,
};

/// Ministral — Mistral instruction format.
pub const MINISTRAL_PROFILE: ModelProfile = ModelProfile {
    name: "ministral",
    description: "Ministral / Mistral 3B instruction model",
    architectures: &["mistral", "mistral3"],
    default_system:
        "You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created \
         by Mistral AI, a French startup headquartered in Paris.\n\
         You power an AI assistant called Le Chat.\n\
         Your knowledge base was last updated on 2023-10-01.\n\
         The current date is {today}.\n\n\
         When you're not sure about some information or when the user's request \
         requires up-to-date or specific data, you must use the available tools \
         to fetch the information. Do not hesitate to use tools whenever they can \
         provide a more accurate or complete response. If no relevant tools are \
         available, then clearly state that you don't have the information and \
         avoid making up anything.\n\
         If the user's question is not clear, ambiguous, or does not provide \
         enough context for you to accurately answer the question, you do not \
         try to answer it right away and you rather ask the user to clarify \
         their request.\n\
         You follow these instructions in all languages, and always respond to \
         the user in the language they use or request.",
    sampling: SamplingDefaults {
        temperature: 0.15,    // Ollama default for Ministral-3:3b
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.05,
    },
    stop_tokens: &[
        StopToken(2, "</s>"),                   // Mistral generic
        StopToken(4, "[/INST]"),                // Mistral-3 turn end (vocab id 4)
    ],
    opens_with_thinking: false,
    raw_continuation: false,
};

/// Llama 3 instruction format.
pub const LLAMA3_PROFILE: ModelProfile = ModelProfile {
    name: "llama3",
    description: "Llama 3 / 3.1 / 3.2 / 3.3 instruction model",
    architectures: &["llama"],
    default_system:
        "You are a helpful, respectful and honest assistant. \
         Always answer as helpfully as possible.",
    sampling: SamplingDefaults {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.1,
    },
    stop_tokens: &[
        StopToken(128009, "<|eot_id|>"),        // Llama 3 turn end
        StopToken(128001, "<|end_of_text|>"),   // Llama BOS
    ],
    opens_with_thinking: false,
    raw_continuation: false,
};

/// Qwen 2/3 (non-reasoning) instruction model — base ChatML without
/// the `` tag.
pub const QWEN_CHAT_PROFILE: ModelProfile = ModelProfile {
    name: "qwen-chat",
    description: "Qwen 2 / 3 plain instruction model (no reasoning)",
    architectures: &["qwen2", "qwen3"],
    default_system: "You are a helpful assistant.",
    sampling: SamplingDefaults {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.1,
    },
    stop_tokens: &[
        StopToken(151645, "<|im_end|>"),       // Qwen2/3 ChatML end
        StopToken(151643, "<|endoftext|>"),    // Qwen EOS
    ],
    opens_with_thinking: false,
    raw_continuation: false,
};

/// Gemma — turn-based format.
pub const GEMMA_PROFILE: ModelProfile = ModelProfile {
    name: "gemma",
    description: "Gemma instruction model",
    architectures: &["gemma", "gemma2", "gemma3"],
    default_system: "",
    sampling: SamplingDefaults {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.0,
    },
    stop_tokens: &[
        StopToken(1, "<end_of_turn>"),
        StopToken(106, "<eos>"),
    ],
    opens_with_thinking: false,
    raw_continuation: false,
};

/// Fallback for anything we don't recognise.
pub const FALLBACK_PROFILE: ModelProfile = ModelProfile {
    name: "fallback",
    description: "Generic fallback (no chat template)",
    architectures: &[],
    default_system: "",
    sampling: SamplingDefaults {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
    },
    stop_tokens: &[],
    opens_with_thinking: false,
    raw_continuation: true,
};

/// All built-in profiles, in lookup order (first match wins).
pub const BUILTIN_PROFILES: &[&ModelProfile] = &[
    &ORNITH_PROFILE,
    &LLAMA3_PROFILE,
    &MINISTRAL_PROFILE,
    &QWEN_CHAT_PROFILE,
    &GEMMA_PROFILE,
];

/// Resolve a profile from GGUF metadata.  Match order:
///   1. explicit profile name in `--profile <name>`
///   2. `general.architecture` against each profile's `architectures`
///   3. fall back to `FALLBACK_PROFILE`
pub fn resolve_profile(
    metadata: &HashMap<String, GGUFValue>,
    explicit: Option<&str>,
) -> ModelProfile {
    if let Some(name) = explicit {
        for p in BUILTIN_PROFILES {
            if p.name == name {
                return (*p).clone();
            }
        }
        eprintln!(
            "[profile] unknown profile '{}', falling back to architecture-based match",
            name
        );
    }

    let arch = metadata
        .get("general.architecture")
        .and_then(|v| {
            if let GGUFValue::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        })
        .unwrap_or("");

    for p in BUILTIN_PROFILES {
        if p.architectures.contains(&arch) {
            return (*p).clone();
        }
    }

    FALLBACK_PROFILE.clone()
}

/// Render a single-turn prompt using a profile's template family.
///
/// For reasoning models (opens_with_thinking=true), the assistant turn
/// opens with `\n<think>\n` so the model can produce a thinking block.
/// This matches what Ollama actually sends to the model.
pub fn render_prompt(
    profile: &ModelProfile,
    system: &str,
    user: &str,
) -> String {
    if profile.raw_continuation {
        if system.is_empty() {
            return user.to_string();
        }
        return format!("{system}\n\n{user}");
    }

    let sys = if system.is_empty() {
        profile.default_system
    } else {
        system
    };

    // Detect arch-specific template from the profile's name
    // (architectures are 1:1 with template family for built-ins).
    match profile.name {
        "ornith" | "qwen-chat" | "qwen3-thinking" => {
            // ChatML: <|im_start|>{role}\n{content}<|im_end|>\n
            // Do NOT pre-inject the thinking opener — the model emits
            // `` itself at the start of its response (verified via
            // Ollama prompt context comparison, 2026-07-28).
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                sys, user
            )
        }
        "llama3" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            sys, user
        ),
        "ministral" => {
            // Mistral [INST] format.
            if sys.is_empty() {
                format!("[INST] {} [/INST]", user)
            } else {
                format!("[INST] {}\n{} [/INST]", sys, user)
            }
        }
        "gemma" => format!(
            "<start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n",
            sys, user
        ),
        _ => {
            if sys.is_empty() {
                user.to_string()
            } else {
                format!("{sys}\n\n{user}")
            }
        }
    }
}

/// Look up the first stop-token id for a profile, defaulting to nothing.
pub fn first_stop_token(profile: &ModelProfile) -> Option<usize> {
    profile.stop_tokens.first().map(|s| s.0)
}

/// Render a multi-turn chat prompt using a profile's template family.
///
/// `history` is a slice of (role, content) tuples in chronological order.
/// Supported roles: "system" (inserted once at the top, replacing the
/// profile default if also present), "user", "assistant", and "tool".
///
/// For reasoning models (opens_with_thinking=true), the FINAL assistant
/// turn opens with `\n<think>\n` so the model can produce a thinking
/// block.  Historical assistant turns are emitted verbatim, since the
/// model already produced those tokens in earlier turns.
pub fn render_chat_prompt(
    profile: &ModelProfile,
    system: &str,
    history: &[(String, String)],
) -> String {
    if profile.raw_continuation {
        // Raw continuation: just concatenate everything with blank
        // lines.  No chat template.
        let mut out = String::new();
        if !system.is_empty() {
            out.push_str(system);
            out.push_str("\n\n");
        }
        for (_role, content) in history {
            out.push_str(content);
            out.push_str("\n\n");
        }
        return out;
    }

    let sys = if system.is_empty() {
        profile.default_system
    } else {
        system
    };

    // Format each (role, content) into the chat template's per-turn
    // wrapper.  The final assistant turn is rendered OPEN (no closing
    // <|im_end|>) so the model can continue from it.
    let (turn_open, turn_close, role_user, role_assistant, role_system, role_tool) = match profile.name {
        "ornith" | "qwen-chat" | "qwen3-thinking" => (
            "<|im_start|>", "<|im_end|>\n",
            "user", "assistant", "system", "tool",
        ),
        "llama3" => (
            "<|start_header_id|>", "<|eot_id|>",
            "user", "assistant", "system", "tool",
        ),
        "gemma" => (
            "<start_of_turn>", "<end_of_turn>\n",
            "user", "model", "system", "tool",
        ),
        _ => ("", "\n", "user", "assistant", "system", "tool"),
    };

    // Special-case Ministral which uses [SYSTEM_PROMPT]/[INST]/[/INST] tags
    // (Mistral-3 instruction format — different from Mistral v1's INST-wrapped
    // system).  System is in its own [SYSTEM_PROMPT]…[/SYSTEM_PROMPT] block;
    // each user turn is a [INST]…[/INST] pair.
    if profile.name == "ministral" {
        let mut out = String::new();
        if !sys.is_empty() {
            out.push_str(&format!("[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT]", sys));
        }
        for (role, content) in history {
            match role.as_str() {
                "user" => {
                    out.push_str(&format!("[INST]{} [/INST]", content));
                }
                "assistant" => {
                    // Assistant turn follows the previous [/INST] until next [INST].
                    out.push_str(&format!(" {}", content));
                }
                "system" => {} // already in [SYSTEM_PROMPT]
                _ => {}
            }
        }
        return out;
    }

    let mut out = String::new();

    // System turn (inserted once at the top).
    if !sys.is_empty() {
        out.push_str(&format!("{}{}{}\n{}{}", turn_open, role_system, "\n", sys, turn_close));
    }

    // Walk the conversation history, emitting user/assistant/tool turns.
    let mut next_is_assistant = false;
    for (role, content) in history {
        match role.as_str() {
            "user" => {
                out.push_str(&format!("{}{}{}\n{}{}", turn_open, role_user, "\n", content, turn_close));
                next_is_assistant = true;
            }
            "assistant" => {
                // Historical assistant turn — close it like any other turn.
                out.push_str(&format!("{}{}{}\n{}{}", turn_open, role_assistant, "\n", content, turn_close));
                next_is_assistant = true;
            }
            "tool" => {
                out.push_str(&format!("{}{}{}\n{}{}", turn_open, role_tool, "\n", content, turn_close));
                next_is_assistant = true;
            }
            "system" => {
                // Extra system messages inside history are ignored here;
                // we only honor the first (or profile default).
            }
            _ => {}
        }
    }

    // Open the assistant turn for the model to continue.
    out.push_str(&format!("{}{}", turn_open, role_assistant));
    if profile.opens_with_thinking {
        out.push_str("\n<think>\n");
    } else {
        out.push_str("\n");
    }

    // Mark where the model should start writing.
    let _ = next_is_assistant;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ornith_template_chatml_open_assistant() {
        let p = &ORNITH_PROFILE;
        assert!(p.opens_with_thinking);
        let rendered = render_prompt(p, "", "hi");
        // The model emits its own `<think>` opener, so the prompt is an
        // open ChatML assistant turn (no injected thinking tag).
        assert!(rendered.starts_with("<|im_start|>system\n"));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
        // Default system must be used.
        assert!(rendered.contains("Ornith"));
    }

    #[test]
    fn test_ministral_template_uses_inst() {
        let p = &MINISTRAL_PROFILE;
        let rendered = render_prompt(p, "", "hi");
        // Default system is prepended inside the [INST] block.
        assert!(rendered.starts_with("[INST] You are Ministral-3-3B-Instruct-2512"));
        assert!(rendered.ends_with("hi [/INST]"));
        // A custom system replaces the default.
        let rendered2 = render_prompt(p, "custom sys", "hi");
        assert!(rendered2.starts_with("[INST] custom sys\nhi [/INST]"));
    }

    #[test]
    fn test_llama3_template() {
        let p = &LLAMA3_PROFILE;
        let rendered = render_prompt(p, "", "hi");
        assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(rendered.contains("helpful"));
    }

    #[test]
    fn test_fallback_is_raw() {
        let p = &FALLBACK_PROFILE;
        assert!(p.raw_continuation);
    }
}
