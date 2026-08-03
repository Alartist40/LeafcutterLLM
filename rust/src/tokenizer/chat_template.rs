//! Lightweight chat-template detector & renderer.
//!
//! GGUF models embed a Jinja2 `tokenizer.chat_template` string.
//! We don't parse full Jinja2; instead we detect known signatures
//! and render the equivalent prompt for a single user turn.

use crate::model::gguf::GGUFValue;
use std::collections::HashMap;

/// Detected chat-template family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFamily {
    Llama3,   // <|begin_of_text|> … <|start_header_id|> … <|end_header_id|>
    Mistral,  // [INST] … [/INST]
    ChatML,   // <|im_start|> … <|im_end|>
    Gemma,    // <start_of_turn>user\n … <end_of_turn>
    Ministral,// [SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]
    Unknown,
}

impl TemplateFamily {
    /// Detect family from raw Jinja2 template string.
    pub fn detect(template: &str) -> Self {
        let t = template.to_lowercase();
        // Ministral-2512 uniquely uses a [SYSTEM_PROMPT]…[/SYSTEM_PROMPT] tag.
        // The "Unsloth template fixes" GGUF embeds the same family (the literal
        // word "think" may be absent), so match on the SYSTEM_PROMPT tag first.
        if t.contains("[system_prompt]") || (t.contains("system_prompt") && t.contains("think")) {
            TemplateFamily::Ministral
        } else if t.contains("start_header_id") || t.contains("end_header_id") {
            TemplateFamily::Llama3
        } else if t.contains("[inst]") || t.contains("[/inst]") {
            TemplateFamily::Mistral
        } else if t.contains("im_start") || t.contains("im_end") {
            TemplateFamily::ChatML
        } else if t.contains("start_of_turn") && t.contains("end_of_turn") {
            TemplateFamily::Gemma
        } else {
            TemplateFamily::Unknown
        }
    }

    /// Render a single-turn user prompt for this family.
    pub fn render(self, system: &str, user: &str) -> String {
        match self {
            TemplateFamily::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n",
                system, user
            ),
            TemplateFamily::Mistral => {
                // Mistral / Mixtral format: [INST] system\nuser [/INST]
                if system.is_empty() {
                    format!("[INST] {} [/INST]", user)
                } else {
                    format!("[INST] {}\n{} [/INST]", system, user)
                }
            }
            TemplateFamily::ChatML => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system, user
            ),
            TemplateFamily::Gemma => format!(
                "<start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n",
                system, user
            ),
            TemplateFamily::Ministral => {
                // Ministral-2512 uses [SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST].
                // The model was trained with a specific default system message; when
                // none is supplied we use the Ministral-3-3B-Instruct-2512 default
                // (truncated to the essential identity + thinking-format instructions).
                let sys = if system.is_empty() {
                    "You are Ministral-3-3B-Instruct-2512, an AI assistant.\n\
                     # HOW YOU SHOULD THINK AND ANSWER\n\n\
                     First draft your thinking process (inner monologue). \
                     Format your response using Markdown, in the same language as the input.\n\n\
                     Your thinking process must follow the template below:\
                     [THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. \
                     Be as long as you want until you are confident to generate the response.[/THINK]\
                     Then, provide a self-contained response."
                        .to_string()
                } else {
                    system.to_string()
                };
                format!(
                    "[SYSTEM_PROMPT]{}[/SYSTEM_PROMPT][INST]{}[/INST]",
                    sys, user
                )
            }
            TemplateFamily::Unknown => {
                // Fallback: plain user message with optional system prefix
                if system.is_empty() {
                    user.to_string()
                } else {
                    format!("{system}\n\n{user}")
                }
            }
        }
    }
}

/// Read `tokenizer.chat_template` from GGUF metadata, detect its family,
/// and render a single-turn prompt.
pub fn apply_chat_template_from_gguf(
    metadata: &HashMap<String, GGUFValue>,
    system: &str,
    user: &str,
) -> String {
    if let Some(GGUFValue::String(template)) = metadata.get("tokenizer.chat_template") {
        let family = TemplateFamily::detect(template);
        family.render(system, user)
    } else {
        // No template in GGUF — detect architecture and use a sensible default.
        let arch = metadata
            .get("general.architecture")
            .and_then(|v| if let GGUFValue::String(s) = v { Some(s.as_str()) } else { None })
            .unwrap_or("");
        match arch {
            "qwen2" | "qwen3" | "qwen35" | "qwen35moe" | "qwen36" | "ornith" => {
                // ChatML — used by all Qwen-family models.
                format!(
                    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                )
            }
            "llama" | "mistral" | "mistral3" => {
                // Llama-3 / Mistral instruction format.
                format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            }
            _ => {
                if system.is_empty() {
                    user.to_string()
                } else {
                    format!("{system}\n\n{user}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama3() {
        let tpl = r#"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% endfor %}"#;
        assert_eq!(TemplateFamily::detect(tpl), TemplateFamily::Llama3);
    }

    #[test]
    fn test_detect_mistral() {
        let tpl = r#"{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}"#;
        assert_eq!(TemplateFamily::detect(tpl), TemplateFamily::Mistral);
    }

    #[test]
    fn test_detect_chatml() {
        let tpl = r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"#;
        assert_eq!(TemplateFamily::detect(tpl), TemplateFamily::ChatML);
    }

    #[test]
    fn test_detect_ministral() {
        let tpl = r#"{%- set default_system_message = 'Think' %}{%- if messages[0]['role'] == 'system' %}{{- '[SYSTEM_PROMPT]' -}}{% endif %}{%- if tools is defined %}[TOOL_CALLS]"#;
        assert_eq!(TemplateFamily::detect(tpl), TemplateFamily::Ministral);
    }

    #[test]
    fn test_render_llama3() {
        let out = TemplateFamily::Llama3.render("Be helpful.", "Hello");
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(out.contains("Be helpful."));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_render_mistral() {
        let out = TemplateFamily::Mistral.render("", "Hello");
        assert_eq!(out, "[INST] Hello [/INST]");
    }

    #[test]
    fn test_render_chatml() {
        let out = TemplateFamily::ChatML.render("Sys", "Hi");
        assert!(out.contains("<|im_start|>system\nSys"));
        assert!(out.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_render_ministral() {
        let out = TemplateFamily::Ministral.render("", "What is 2+2?");
        assert!(out.starts_with("[SYSTEM_PROMPT]"));
        assert!(out.contains("[THINK]"));
        assert!(out.ends_with("[INST]What is 2+2?[/INST]"));
    }
}
