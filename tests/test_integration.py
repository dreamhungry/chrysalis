"""端到端集成测试

验证各模块的正确性和协同工作能力。
无需LLM服务即可运行（使用Mock后端）。
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Windows终端UTF-8兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.decision.base import DecisionBackend
from src.learning.feedback_updater import FeedbackUpdater
from src.learning.pattern_extractor import PatternExtractor
from src.learning.reflector import Reflector
from src.learning.rl_updater import RLUpdater
from src.memory.backends.markdown_backend import MarkdownFileBackend
from src.memory.interaction_store import InteractionStore
from src.memory.memory_manager import MemoryManager
from src.personality.personality_store import PersonalityStore
from src.personality.trait_vector import TraitVector
from src.agent.agent_core import AgentCore


# ==================== Mock决策后端 ====================

class MockDecisionBackend(DecisionBackend):
    """模拟决策后端，用于测试"""

    def __init__(self):
        self._call_count = 0

    def generate_response(self, user_input, personality_vector, conversation_history, context=None):
        self._call_count += 1
        # 根据人格向量生成不同风格的响应
        friendliness = personality_vector[0] if len(personality_vector) > 0 else 0
        if friendliness > 0.3:
            prefix = "你好呀！"
        elif friendliness < -0.3:
            prefix = "嗯。"
        else:
            prefix = ""
        return f"{prefix}这是对「{user_input}」的回复 (#{self._call_count})"

    def update_model(self, training_data):
        return False

    def evaluate_action(self, action, state, personality_vector):
        return 0.5

    def get_backend_info(self):
        return {"type": "mock", "calls": self._call_count}


# ==================== 测试函数 ====================

def test_trait_vector():
    """测试人格向量模块"""
    print("=" * 50)
    print("测试 1: TraitVector 人格向量")
    print("=" * 50)

    # 创建和初始化
    tv = TraitVector(dimensions=10)
    assert tv.dimensions == 10
    assert np.allclose(tv.vector, np.zeros(10))
    print("  ✓ 初始化为零向量")

    # 设置特征值
    tv.set_trait("friendliness", 0.8)
    tv.set_trait("humor", 0.6)
    assert abs(tv.get_trait("friendliness") - 0.8) < 1e-6
    print("  ✓ 设置和获取特征值")

    # 增量更新
    delta = np.array([0.1] * 10)
    tv.update(delta, learning_rate=0.5)
    assert tv.get_trait("friendliness") <= 1.0  # 不超过边界
    print("  ✓ 增量更新和归一化")

    # 序列化/反序列化
    json_str = tv.to_json()
    tv2 = TraitVector.from_json(json_str)
    assert np.allclose(tv.vector, tv2.vector)
    print("  ✓ JSON序列化/反序列化")

    # 自然语言描述
    desc = tv.to_description()
    assert isinstance(desc, str) and len(desc) > 0
    print(f"  ✓ 人格描述: {desc}")

    # 相似度计算
    tv3 = TraitVector(dimensions=10, initial_values=tv.vector * 0.5)
    dist = tv.distance(tv3)
    sim = tv.cosine_similarity(tv3)
    print(f"  ✓ 距离={dist:.3f}, 余弦相似度={sim:.3f}")

    print("  [PASS] TraitVector 测试通过\n")


def test_personality_store():
    """测试人格持久化"""
    print("=" * 50)
    print("测试 2: PersonalityStore 持久化")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = PersonalityStore(os.path.join(tmpdir, "personality.json"))

        # 保存
        tv = TraitVector(dimensions=10)
        tv.set_trait("friendliness", 0.7)
        tv.set_trait("creativity", -0.3)
        store.save(tv)
        print("  ✓ 保存人格向量")

        # 加载
        tv_loaded = store.load()
        assert tv_loaded is not None
        assert abs(tv_loaded.get_trait("friendliness") - 0.7) < 1e-6
        print("  ✓ 加载人格向量")

        # 历史快照
        store.save_snapshot(tv, event="test_event")
        store.save_snapshot(tv, event="test_event_2")
        history = store.get_evolution_history()
        assert len(history) == 2
        print(f"  ✓ 历史快照: {len(history)}条")

    print("  [PASS] PersonalityStore 测试通过\n")


def test_markdown_memory():
    """测试Markdown记忆存储"""
    print("=" * 50)
    print("测试 3: MarkdownFileBackend 记忆存储")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "history.md")
        backend = MarkdownFileBackend(md_path)

        # 添加交互
        id1 = backend.add_interaction("你好", "你好！有什么可以帮你的？")
        id2 = backend.add_interaction(
            "天气怎么样？", "今天天气不错。", feedback=0.8
        )
        id3 = backend.add_interaction(
            "谢谢", "不客气！", metadata={"topic": "greeting"}
        )
        assert backend.count() == 3
        print(f"  ✓ 添加3条交互 (IDs: {id1[:8]}..., {id2[:8]}..., {id3[:8]}...)")

        # 查询最近记录
        recent = backend.get_recent(2)
        assert len(recent) == 2
        assert recent[-1]["user_input"] == "谢谢"
        print("  ✓ 查询最近记录")

        # 按ID查询
        record = backend.get_by_id(id2)
        assert record is not None
        assert record["feedback"] == 0.8
        print("  ✓ 按ID查询")

        # 获取反馈记录
        feedbacks = backend.get_all_feedbacks()
        assert len(feedbacks) == 1
        print("  ✓ 获取反馈记录")

        # 更新反馈
        success = backend.update_feedback(id1, 0.9)
        assert success
        assert backend.get_by_id(id1)["feedback"] == 0.9
        print("  ✓ 更新反馈")

        # 验证Markdown文件内容
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "# Chrysalis 交互历史记录" in content
        assert "你好" in content
        assert "天气怎么样" in content
        assert "⭐" in content
        print("  ✓ Markdown文件格式正确")

        # 测试重新加载（从文件解析）
        backend2 = MarkdownFileBackend(md_path)
        assert backend2.count() >= 3  # 至少3条（可能包含反馈更新记录的解析）
        print(f"  ✓ 重新加载: {backend2.count()}条记录")

    print("  [PASS] MarkdownFileBackend 测试通过\n")


def test_memory_manager():
    """测试记忆管理器工厂"""
    print("=" * 50)
    print("测试 4: MemoryManager 工厂")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "history.md")
        store = MemoryManager.create_store("markdown", storage_path=md_path)

        store.add_interaction("测试", "收到")
        assert store.count() == 1
        print("  ✓ 通过工厂创建Markdown存储")

    print("  [PASS] MemoryManager 测试通过\n")


def test_learning_modules():
    """测试学习模块"""
    print("=" * 50)
    print("测试 5: Learning 学习模块")
    print("=" * 50)

    tv = TraitVector(dimensions=10)
    tv.set_trait("friendliness", 0.5)
    initial_vector = tv.vector.copy()

    # 反馈更新器
    updater = FeedbackUpdater(base_learning_rate=0.1)
    updater.update(tv, feedback=0.9, context={"personality_vector": tv.vector.tolist()})
    assert not np.allclose(tv.vector, initial_vector)
    print(f"  ✓ FeedbackUpdater: friendliness {initial_vector[0]:.3f} -> {tv.vector[0]:.3f}")

    # 模式提取器
    interactions = [
        {"user_input": "你好？", "agent_response": "你好！" * 30, "feedback": 0.8},
        {"user_input": "你是谁？", "agent_response": "我是AI助手。", "feedback": 0.7},
        {"user_input": "天气如何？", "agent_response": "今天天气不错。", "feedback": 0.9},
    ]
    extractor = PatternExtractor(analysis_window=10)
    patterns = extractor.extract_patterns(interactions)
    assert "avg_feedback" in patterns
    print(f"  ✓ PatternExtractor: avg_feedback={patterns['avg_feedback']:.2f}")

    before = tv.vector.copy()
    extractor.update_from_patterns(tv, interactions)
    print(f"  ✓ 模式更新: norm变化 {np.linalg.norm(before):.3f} -> {np.linalg.norm(tv.vector):.3f}")

    # RL更新器
    rl = RLUpdater()
    rl.record_step(tv.vector, 0.8)
    rl.record_step(tv.vector, 0.9)
    rl.record_step(tv.vector, 0.7)
    assert rl.buffer_size == 3
    avg_return = rl.update(tv)
    assert avg_return is not None
    assert rl.buffer_size == 0
    print(f"  ✓ RLUpdater: avg_return={avg_return:.3f}")

    # 反思器（统计模式）
    reflector = Reflector(decision_backend=None)
    result = reflector.reflect(tv, interactions)
    assert result is not None
    assert "analysis" in result
    print(f"  ✓ Reflector (统计): {result['method']}")

    print("  [PASS] Learning 测试通过\n")


def test_agent_core():
    """测试Agent Core端到端"""
    print("=" * 50)
    print("测试 6: AgentCore 端到端")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建各模块
        tv = TraitVector(dimensions=10)
        tv.set_trait("friendliness", 0.6)

        personality_store = PersonalityStore(
            os.path.join(tmpdir, "personality.json")
        )

        md_path = os.path.join(tmpdir, "history.md")
        interaction_store = MemoryManager.create_store(
            "markdown", storage_path=md_path
        )

        mock_backend = MockDecisionBackend()

        agent = AgentCore(
            decision_backend=mock_backend,
            trait_vector=tv,
            interaction_store=interaction_store,
            personality_store=personality_store,
        )

        # 多轮对话
        responses = []
        for i in range(5):
            response = agent.chat(f"测试消息 {i + 1}")
            responses.append(response)
            print(f"  对话 {i + 1}: {response[:50]}...")

        assert len(responses) == 5
        assert agent.memory.count() == 5
        print(f"  ✓ 5轮对话完成, 记录数: {agent.memory.count()}")

        # 提供反馈
        initial_personality = tv.vector.copy()
        agent.provide_feedback(0.9)
        agent.provide_feedback(0.2)
        agent.provide_feedback(0.8)
        print(f"  ✓ 3次反馈已记录")

        # 验证人格发生变化
        assert not np.allclose(tv.vector, initial_personality)
        print(f"  ✓ 人格向量已更新: norm {np.linalg.norm(initial_personality):.3f} -> {np.linalg.norm(tv.vector):.3f}")

        # 查看人格信息
        info = agent.get_personality_info()
        assert info["interaction_count"] == 5
        assert info["update_count"] > 0
        print(f"  ✓ 人格信息: {info['description']}")

        # 验证持久化
        loaded = personality_store.load()
        assert loaded is not None
        print(f"  ✓ 人格持久化验证通过")

        # 验证演化历史
        history = personality_store.get_evolution_history()
        assert len(history) > 0
        print(f"  ✓ 演化历史: {len(history)}条快照")

        # 验证Markdown文件
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "测试消息 1" in md_content
        assert "测试消息 5" in md_content
        print(f"  ✓ Markdown交互历史验证通过 ({len(md_content)} bytes)")

    print("  [PASS] AgentCore 测试通过\n")


def test_decision_backend_switching():
    """测试决策后端动态切换"""
    print("=" * 50)
    print("测试 7: 决策后端切换")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        tv = TraitVector(dimensions=10)
        personality_store = PersonalityStore(
            os.path.join(tmpdir, "personality.json")
        )
        interaction_store = MemoryManager.create_store(
            "markdown", storage_path=os.path.join(tmpdir, "history.md")
        )

        backend1 = MockDecisionBackend()
        agent = AgentCore(
            decision_backend=backend1,
            trait_vector=tv,
            interaction_store=interaction_store,
            personality_store=personality_store,
        )

        # 使用后端1
        r1 = agent.chat("你好")
        assert "mock" in agent.decision.get_backend_info()["type"]
        print(f"  ✓ 后端1响应: {r1[:40]}...")

        # 切换后端
        backend2 = MockDecisionBackend()
        agent.switch_backend(backend2)
        r2 = agent.chat("再见")
        assert backend2._call_count == 1
        print(f"  ✓ 后端2响应: {r2[:40]}...")

    print("  [PASS] 后端切换测试通过\n")


# ==================== 主入口 ====================

def main():
    print("\n" + "=" * 60)
    print("  Chrysalis 集成测试")
    print("=" * 60 + "\n")

    tests = [
        test_trait_vector,
        test_personality_store,
        test_markdown_memory,
        test_memory_manager,
        test_learning_modules,
        test_agent_core,
        test_decision_backend_switching,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败 (共 {len(tests)} 项)")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
