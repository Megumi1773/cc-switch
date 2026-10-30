//! Claude 侧「思考模式」适配（Anthropic Messages → OpenAI Chat）。
//!
//! 背景：Claude Code / Claude Code desktop 的思考模式由 Anthropic 协议的
//! `thinking` 字段驱动，响应侧则只认 `thinking` content block。可 OpenAI 格式的
//! 中转站分两类：
//!
//! 1. 接受思考输入、把思考回在 `reasoning_content` / `reasoning` /
//!    `reasoning_details` 里——响应侧映射由 `codex_chat_common` 的穷举提取负责；
//! 2. 完全不接受思考输入（`thinking` / `reasoning_effort` / 历史
//!    `reasoning_content` 都会 400），也不回思考——如 `ai.wesign.top`。
//!
//! 第二类站点上，Claude Code 的思考开关发出去只会换来 400 或者一片空白 UI。
//! 本模块把「这条上游支持什么」收敛成一份按供应商解析的能力描述，并提供两件事：
//!
//! - **请求侧**：剥离上游不接受的思考输入参数（`apply_request_thinking_policy`）；
//! - **模拟**：上游压根不产出思考时，用系统提示词让模型把推理写进
//!   `<thinking>…</thinking>`，再由 `inline_think` 拆分器还原成 thinking 块。
//!
//! 没声明的供应商一律拿到空策略（`Default`），行为与引入本模块前一致。

use crate::provider::{ClaudeChatThinkingConfig, Provider};
use serde_json::{json, Value};

/// 内置站点：wesign 中转（OpenAI 格式、不接受思考输入）。
const WESIGN_HOST: &str = "wesign.top";

/// 模拟思考用的系统提示词。
///
/// 刻意放在 system 末尾（而不是开头）：Claude Code 的 system 前缀是缓存主体，
/// 尾部追加一块不影响前缀缓存命中。
pub const INLINE_THINKING_SYSTEM_PROMPT: &str = "Before answering, work through the problem \
inside a single <thinking>...</thinking> block placed at the very start of your reply, then write \
the final answer after the closing tag. Use plain prose inside the tags, never mention this \
instruction to the user, and never place the tags anywhere else in the reply.";

/// 一条上游的思考能力策略。`None` = 未声明，沿用内置启发式。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClaudeChatThinkingPolicy {
    /// 上游是否接受思考输入（`thinking` / `output_config` / `reasoning_effort`）。
    pub supports_thinking_input: Option<bool>,
    /// 上游是否接受 `reasoning_effort` 档位。
    pub supports_effort: Option<bool>,
    /// 上游是否接受 assistant 历史里的 `reasoning_content`。
    pub accepts_reasoning_history: Option<bool>,
    /// 上游不产出思考时是否用提示词模拟。
    pub emulate_inline: Option<bool>,
    /// 声明性字段，仅作文档与排障用。
    pub output_format: Option<String>,
}

impl ClaudeChatThinkingPolicy {
    /// 是否允许把思考输入参数发给上游（未声明视为允许，保持旧行为）。
    pub fn thinking_input_allowed(&self) -> bool {
        self.supports_thinking_input.unwrap_or(true)
    }

    /// 是否允许注入 `reasoning_effort`（未声明交给模型名启发式）。
    pub fn effort_allowed(&self) -> bool {
        self.supports_effort != Some(false)
    }

    /// 是否需要提示词模拟思考。
    pub fn emulates_inline(&self) -> bool {
        self.emulate_inline.unwrap_or(false)
    }

    /// 历史 thinking 是否转成 `reasoning_content`。`None` 交给厂商启发式。
    pub fn reasoning_history_allowed(&self) -> Option<bool> {
        self.accepts_reasoning_history
    }

    fn overlay(&mut self, explicit: &ClaudeChatThinkingConfig) {
        if explicit.supports_thinking.is_some() {
            self.supports_thinking_input = explicit.supports_thinking;
        }
        if explicit.supports_effort.is_some() {
            self.supports_effort = explicit.supports_effort;
        }
        if explicit.accepts_reasoning_history.is_some() {
            self.accepts_reasoning_history = explicit.accepts_reasoning_history;
        }
        if explicit.emulate_inline.is_some() {
            self.emulate_inline = explicit.emulate_inline;
        }
        if let Some(format) = explicit.output_format.as_ref() {
            self.output_format = Some(format.clone());
        }
    }
}

/// 请求侧实际做了哪些改动，用于日志与测试。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppliedThinkingPolicy {
    /// 剥离了顶层 `thinking`
    pub removed_thinking: bool,
    /// 剥离了 effort（`output_config` / `reasoning_effort`）
    pub removed_effort: bool,
    /// 注入了模拟思考的系统提示词
    pub injected_emulation_prompt: bool,
}

impl AppliedThinkingPolicy {
    pub fn changed(&self) -> bool {
        self.removed_thinking || self.removed_effort || self.injected_emulation_prompt
    }
}

/// 解析供应商的思考策略：内置站点规则 → 供应商 meta 显式声明覆盖。
pub fn resolve_claude_chat_thinking_policy(provider: &Provider) -> ClaudeChatThinkingPolicy {
    let mut policy = builtin_policy(provider);
    if let Some(explicit) = provider
        .meta
        .as_ref()
        .and_then(|meta| meta.claude_chat_thinking.as_ref())
    {
        policy.overlay(explicit);
    }
    policy
}

/// 内置站点规则。只按「域名 / 供应商名」判定，绝不掺模型名——同一个模型在不同
/// 平台上的思考入口不一样，模型名属于厂商而不属于平台。
fn builtin_policy(provider: &Provider) -> ClaudeChatThinkingPolicy {
    let base_url = provider_base_url(provider);
    let name = provider.name.to_ascii_lowercase();

    if base_url.contains(WESIGN_HOST) || name.contains("wesign") {
        return ClaudeChatThinkingPolicy {
            supports_thinking_input: Some(false),
            supports_effort: Some(false),
            accepts_reasoning_history: Some(false),
            // 站点不产出思考，只能靠提示词模拟，否则 Claude Code desktop 的
            // 思考开关点了也是空白。
            emulate_inline: Some(true),
            output_format: Some("auto".to_string()),
        };
    }

    ClaudeChatThinkingPolicy::default()
}

fn provider_base_url(provider: &Provider) -> String {
    let settings = &provider.settings_config;
    [
        settings
            .get("env")
            .and_then(|env| env.get("ANTHROPIC_BASE_URL")),
        settings.get("base_url"),
        settings.get("baseURL"),
        settings.get("apiEndpoint"),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .map(str::trim)
    .find(|url| !url.is_empty())
    .unwrap_or_default()
    .to_ascii_lowercase()
}

/// 请求侧适配：按策略剥离上游不接受的思考输入，必要时注入模拟提示词。
///
/// 必须在 Anthropic → OpenAI 转换之前调用（转换层只负责结构映射）。
/// 幂等：重复调用不会重复追加提示词。
pub fn apply_request_thinking_policy(
    body: &mut Value,
    policy: &ClaudeChatThinkingPolicy,
) -> AppliedThinkingPolicy {
    let mut applied = AppliedThinkingPolicy::default();

    // 客户端到底有没有开思考：要在剥离之前判断，模拟提示词只在这一刻注入。
    let thinking_requested = matches!(
        body.pointer("/thinking/type").and_then(Value::as_str),
        Some("enabled" | "adaptive")
    );

    if !policy.thinking_input_allowed() {
        if let Some(obj) = body.as_object_mut() {
            applied.removed_thinking = obj.remove("thinking").is_some();
            applied.removed_effort = obj.remove("output_config").is_some();
            applied.removed_effort |= obj.remove("reasoning_effort").is_some();
            strip_interleaved_thinking_beta(obj);
        }
    } else if !policy.effort_allowed() {
        if let Some(obj) = body.as_object_mut() {
            applied.removed_effort = strip_output_config_effort(obj);
            applied.removed_effort |= obj.remove("reasoning_effort").is_some();
        }
    }

    if policy.emulates_inline() && thinking_requested {
        applied.injected_emulation_prompt = inject_inline_thinking_instruction(body);
    }

    applied
}

/// `anthropic_beta` 里的 interleaved-thinking 标记对 Chat 上游没有意义，顺手清掉；
/// 其它 beta 原样保留。
fn strip_interleaved_thinking_beta(obj: &mut serde_json::Map<String, Value>) {
    let Some(Value::Array(betas)) = obj.get_mut("anthropic_beta") else {
        return;
    };
    betas.retain(|beta| {
        !beta
            .as_str()
            .is_some_and(|value| value.contains("interleaved-thinking"))
    });
    if betas.is_empty() {
        obj.remove("anthropic_beta");
    }
}

/// 删除 `output_config.effort`；`output_config` 空了就一起删。
/// 返回是否真的删了 effort。
fn strip_output_config_effort(obj: &mut serde_json::Map<String, Value>) -> bool {
    let Some(Value::Object(config)) = obj.get_mut("output_config") else {
        return false;
    };
    let removed = config.remove("effort").is_some();
    if config.is_empty() {
        obj.remove("output_config");
    }
    removed
}

/// 把模拟提示词追加到 `system` 末尾。`system` 可能是字符串或 text block 数组，
/// 两种形态都要处理；已存在时不重复插入。
fn inject_inline_thinking_instruction(body: &mut Value) -> bool {
    match body.get_mut("system") {
        Some(Value::String(text)) => {
            if text.contains(INLINE_THINKING_SYSTEM_PROMPT) {
                return false;
            }
            let merged = format!("{text}\n\n{INLINE_THINKING_SYSTEM_PROMPT}");
            *body.get_mut("system").expect("system present") = Value::String(merged);
            true
        }
        Some(Value::Array(parts)) => {
            let already = parts.iter().any(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains(INLINE_THINKING_SYSTEM_PROMPT))
            });
            if already {
                return false;
            }
            parts.push(json!({
                "type": "text",
                "text": INLINE_THINKING_SYSTEM_PROMPT
            }));
            true
        }
        Some(Value::Null) | None => {
            body["system"] = json!([{
                "type": "text",
                "text": INLINE_THINKING_SYSTEM_PROMPT
            }]);
            true
        }
        // 结构不认识就别动，宁可没有模拟思考也不要污染请求。
        Some(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderMeta;

    fn provider(name: &str, base_url: &str, meta: ProviderMeta) -> Provider {
        Provider {
            id: "test".to_string(),
            name: name.to_string(),
            settings_config: json!({
                "env": {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_AUTH_TOKEN": "sk-test"
                }
            }),
            website_url: None,
            category: Some("claude".to_string()),
            created_at: None,
            sort_index: None,
            notes: None,
            meta: Some(meta),
            icon: None,
            icon_color: None,
            in_failover_queue: false,
        }
    }

    fn wesign_provider() -> Provider {
        provider("wesign", "https://ai.wesign.top/v1", ProviderMeta::default())
    }

    #[test]
    fn wesign_is_recognised_as_thinking_input_unsupported() {
        let policy = resolve_claude_chat_thinking_policy(&wesign_provider());
        assert!(!policy.thinking_input_allowed());
        assert!(!policy.effort_allowed());
        assert_eq!(policy.reasoning_history_allowed(), Some(false));
        assert!(policy.emulates_inline());
    }

    #[test]
    fn unknown_provider_keeps_legacy_behaviour() {
        let policy = resolve_claude_chat_thinking_policy(&provider(
            "generic",
            "https://api.example.com",
            ProviderMeta::default(),
        ));
        assert_eq!(policy, ClaudeChatThinkingPolicy::default());
        assert!(policy.thinking_input_allowed());
        assert!(policy.effort_allowed());
        assert!(!policy.emulates_inline());
    }

    #[test]
    fn explicit_meta_overrides_builtin() {
        let meta = ProviderMeta {
            claude_chat_thinking: Some(ClaudeChatThinkingConfig {
                supports_thinking: Some(true),
                emulate_inline: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let policy = resolve_claude_chat_thinking_policy(&provider(
            "wesign",
            "https://ai.wesign.top/v1",
            meta,
        ));
        // 显式声明优先于内置规则，其余字段仍沿用内置值
        assert!(policy.thinking_input_allowed());
        assert!(!policy.effort_allowed());
        assert!(!policy.emulates_inline());
    }

    #[test]
    fn strips_thinking_inputs_and_emulates() {
        let policy = resolve_claude_chat_thinking_policy(&wesign_provider());
        let mut body = json!({
            "model": "deepseek-v4-flash",
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "output_config": {"effort": "high"},
            "anthropic_beta": ["interleaved-thinking-2025-05-14", "context-1m-2025-08-07"],
            "system": "You are Claude Code.",
            "messages": [{"role": "user", "content": "hi"}]
        });

        let applied = apply_request_thinking_policy(&mut body, &policy);

        assert!(applied.removed_thinking);
        assert!(applied.removed_effort);
        assert!(applied.injected_emulation_prompt);
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());
        assert!(body.get("reasoning_effort").is_none());
        // 非 thinking 的 beta 保留
        assert_eq!(body["anthropic_beta"], json!(["context-1m-2025-08-07"]));
        // 字符串 system 就地合并（形态保持字符串，尽量少动上游请求）
        let system = body["system"].as_str().expect("system stays a string");
        assert!(system.starts_with("You are Claude Code."));
        assert!(system.contains("<thinking>"));
        // 数组形态则追加一个 text block，前缀保持不变
        let mut array_body = json!({
            "model": "deepseek-v4-flash",
            "thinking": {"type": "enabled"},
            "system": [{"type": "text", "text": "system A"}],
            "messages": []
        });
        assert!(
            apply_request_thinking_policy(&mut array_body, &policy).injected_emulation_prompt
        );
        let array_system = array_body["system"].as_array().unwrap();
        assert_eq!(array_system.len(), 2);
        assert_eq!(array_system[0]["text"], "system A");
        assert!(array_system[1]["text"].as_str().unwrap().contains("<thinking>"));
    }

    #[test]
    fn emulation_prompt_is_not_duplicated() {
        let policy = resolve_claude_chat_thinking_policy(&wesign_provider());
        let mut body = json!({
            "model": "deepseek-v4-flash",
            "thinking": {"type": "enabled"},
            "system": [{"type": "text", "text": "system A"}],
            "messages": []
        });

        assert!(apply_request_thinking_policy(&mut body, &policy).injected_emulation_prompt);
        // 第二次调用时 thinking 已被剥离，不再重复注入
        assert!(!apply_request_thinking_policy(&mut body, &policy).injected_emulation_prompt);
        let system = body["system"].as_array().unwrap();
        assert_eq!(system.len(), 2);
    }

    #[test]
    fn thinking_disabled_client_gets_no_emulation() {
        let policy = resolve_claude_chat_thinking_policy(&wesign_provider());
        let mut body = json!({
            "model": "deepseek-v4-flash",
            "thinking": {"type": "disabled"},
            "messages": []
        });

        let applied = apply_request_thinking_policy(&mut body, &policy);

        assert!(applied.removed_thinking);
        assert!(!applied.injected_emulation_prompt);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn effort_only_policy_keeps_thinking_but_drops_effort() {
        let policy = ClaudeChatThinkingPolicy {
            supports_thinking_input: Some(true),
            supports_effort: Some(false),
            ..Default::default()
        };
        let mut body = json!({
            "model": "gpt-5.4",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "output_config": {"effort": "high"},
            "messages": []
        });

        let applied = apply_request_thinking_policy(&mut body, &policy);

        assert!(!applied.removed_thinking);
        assert!(applied.removed_effort);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(body.get("output_config").is_none());
    }
}
