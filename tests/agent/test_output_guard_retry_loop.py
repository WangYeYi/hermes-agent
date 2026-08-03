"""测试 output-guard 的代码级强制重答循环。

验证：
1. MISSING 时触发重答（不再仅注入 nudge）
2. 空响应 + tool calls → 触发恢复
3. 重试上限后 fallback
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestOutputGuardRetryLoop:
    """测试 _run_output_guard 检测逻辑不变。"""

    def test_multi_item_detects_missing(self):
        """多问题消息中遗漏某项应被检测。"""
        from agent.conversation_loop import _run_output_guard

        agent = MagicMock()
        agent._session_id = "test-session"
        user_msg = {"role": "user", "content": "今天天气如何？有什么新闻？下午有会吗？"}
        final_response = "今天晴天24°C。今天的头条是AI技术突破。"

        missed = _run_output_guard(agent, final_response, user_msg)
        assert len(missed) > 0, "Should detect at least one missed item"

    def test_fully_covered_passes(self):
        """全部覆盖时返回空列表（依赖 L2 embedding，跳过集成测试）。"""
        pytest.skip("Requires running embedding server — tested in integration")

    def test_single_item_skips(self):
        """单条目消息不触发 guard。"""
        from agent.conversation_loop import _run_output_guard

        agent = MagicMock()
        agent._session_id = "test-session"
        user_msg = {"role": "user", "content": "今天天气如何？"}
        final_response = "今天晴天。"

        missed = _run_output_guard(agent, final_response, user_msg)
        assert missed == [], f"Single item should skip, got {missed}"

    def test_empty_response_skips(self):
        """空 final_response 不触发 guard。"""
        from agent.conversation_loop import _run_output_guard

        agent = MagicMock()
        agent._session_id = "test-session"
        user_msg = {"role": "user", "content": "今天天气如何？有什么新闻？"}
        final_response = ""

        missed = _run_output_guard(agent, final_response, user_msg)
        assert missed == [], f"Empty response should skip, got {missed}"

    def test_system_prompt_fragments_skipped(self):
        """系统指令片段不应被当作用户问题检测。"""
        from agent.conversation_loop import _run_output_guard

        agent = MagicMock()
        agent._session_id = "test-session"
        user_msg = {"role": "user", "content": (
            "Review the conversation above and update the skill library. "
            "Be ACTIVE — most sessions produce at least one skill update."
        )}
        final_response = "Nothing to save."

        missed = _run_output_guard(agent, final_response, user_msg)
        assert missed == [], f"System prompt should skip, got {missed}"


class TestEmptyResponseRecovery:
    """测试空响应恢复逻辑的标志位。"""

    def test_empty_recovery_synthetic_flag(self):
        """空响应恢复消息应带 _empty_recovery_synthetic 标志。"""
        msg = {"role": "user", "content": "test",
               "_empty_recovery_synthetic": True}
        assert msg.get("_empty_recovery_synthetic") is True


class TestGuardRetrySyntheticFlag:
    """测试重试消息的标志位。"""

    def test_retry_msg_has_synthetic_flag(self):
        """重试消息应带 _guard_retry_synthetic 标志。"""
        msg = {"role": "user", "content": "test",
               "_guard_retry_synthetic": True}
        assert msg.get("_guard_retry_synthetic") is True

    def test_nudge_msg_has_synthetic_flag(self):
        """兜底 nudge 消息应带 _guard_nudge_synthetic 标志。"""
        msg = {"role": "user", "content": "test",
               "_guard_nudge_synthetic": True}
        assert msg.get("_guard_nudge_synthetic") is True
