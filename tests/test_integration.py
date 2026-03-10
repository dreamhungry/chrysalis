"""End-to-End Integration Tests

Validates correctness and collaboration of all modules.
Can run without LLM service (using Mock backend).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Windows terminal UTF-8 compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root directory is in Python path
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


# ==================== Mock Decision Backend ====================

class MockDecisionBackend(DecisionBackend):
    """Mock decision backend for testing"""

    def __init__(self):
        self._call_count = 0

    def generate_response(self, user_input, personality_vector, conversation_history, context=None):
        self._call_count += 1
        # Generate different style responses based on personality vector
        friendliness = personality_vector[0] if len(personality_vector) > 0 else 0
        if friendliness > 0.3:
            prefix = "Hello there!"
        elif friendliness < -0.3:
            prefix = "Hmm."
        else:
            prefix = ""
        return f"{prefix} This is a response to '{user_input}' (#{self._call_count})"

    def update_model(self, training_data):
        return False

    def evaluate_action(self, action, state, personality_vector):
        return 0.5

    def get_backend_info(self):
        return {"type": "mock", "calls": self._call_count}


# ==================== Test Functions ====================

def test_trait_vector():
    """Test personality vector module"""
    print("=" * 50)
    print("Test 1: TraitVector Personality Vector")
    print("=" * 50)

    # Create and initialize
    tv = TraitVector(dimensions=10)
    assert tv.dimensions == 10
    assert np.allclose(tv.vector, np.zeros(10))
    print("  [OK] Initialized to zero vector")

    # Set trait values
    tv.set_trait("friendliness", 0.8)
    tv.set_trait("humor", 0.6)
    assert abs(tv.get_trait("friendliness") - 0.8) < 1e-6
    print("  [OK] Set and get trait values")

    # Incremental update
    delta = np.array([0.1] * 10)
    tv.update(delta, learning_rate=0.5)
    assert tv.get_trait("friendliness") <= 1.0  # Not exceeding boundary
    print("  [OK] Incremental update and normalization")

    # Serialization/deserialization
    json_str = tv.to_json()
    tv2 = TraitVector.from_json(json_str)
    assert np.allclose(tv.vector, tv2.vector)
    print("  [OK] JSON serialization/deserialization")

    # Natural language description
    desc = tv.to_description()
    assert isinstance(desc, str) and len(desc) > 0
    print(f"  [OK] Personality description: {desc}")

    # Similarity calculation
    tv3 = TraitVector(dimensions=10, initial_values=tv.vector * 0.5)
    dist = tv.distance(tv3)
    sim = tv.cosine_similarity(tv3)
    print(f"  [OK] Distance={dist:.3f}, Cosine similarity={sim:.3f}")

    print("  [PASS] TraitVector test passed\n")


def test_personality_store():
    """Test personality persistence"""
    print("=" * 50)
    print("Test 2: PersonalityStore Persistence")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = PersonalityStore(os.path.join(tmpdir, "personality.json"))

        # Save
        tv = TraitVector(dimensions=10)
        tv.set_trait("friendliness", 0.7)
        tv.set_trait("creativity", -0.3)
        store.save(tv)
        print("  [OK] Saved personality vector")

        # Load
        tv_loaded = store.load()
        assert tv_loaded is not None
        assert abs(tv_loaded.get_trait("friendliness") - 0.7) < 1e-6
        print("  [OK] Loaded personality vector")

        # History snapshots
        store.save_snapshot(tv, event="test_event")
        store.save_snapshot(tv, event="test_event_2")
        history = store.get_evolution_history()
        assert len(history) == 2
        print(f"  [OK] History snapshots: {len(history)} records")

    print("  [PASS] PersonalityStore test passed\n")


def test_markdown_memory():
    """Test Markdown memory storage"""
    print("=" * 50)
    print("Test 3: MarkdownFileBackend Memory Storage")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "history.md")
        backend = MarkdownFileBackend(md_path)

        # Add interactions
        id1 = backend.add_interaction("Hello", "Hello! How can I help you?")
        id2 = backend.add_interaction(
            "How's the weather?", "The weather is nice today.", feedback=0.8
        )
        id3 = backend.add_interaction(
            "Thanks", "You're welcome!", metadata={"topic": "greeting"}
        )
        assert backend.count() == 3
        print(f"  [OK] Added 3 interactions (IDs: {id1[:8]}..., {id2[:8]}..., {id3[:8]}...)")

        # Query recent records
        recent = backend.get_recent(2)
        assert len(recent) == 2
        assert recent[-1]["user_input"] == "Thanks"
        print("  [OK] Queried recent records")

        # Query by ID
        record = backend.get_by_id(id2)
        assert record is not None
        assert record["feedback"] == 0.8
        print("  [OK] Queried by ID")

        # Get feedback records
        feedbacks = backend.get_all_feedbacks()
        assert len(feedbacks) == 1
        print("  [OK] Retrieved feedback records")

        # Update feedback
        success = backend.update_feedback(id1, 0.9)
        assert success
        assert backend.get_by_id(id1)["feedback"] == 0.9
        print("  [OK] Updated feedback")

        # Verify Markdown file content
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "# Chrysalis Interaction History" in content
        assert "Hello" in content
        assert "How's the weather?" in content
        print("  [OK] Markdown file format correct")

        # Test reload (parse from file)
        backend2 = MarkdownFileBackend(md_path)
        assert backend2.count() >= 3  # At least 3 (may include feedback update records)
        print(f"  [OK] Reloaded: {backend2.count()} records")

    print("  [PASS] MarkdownFileBackend test passed\n")


def test_memory_manager():
    """Test memory manager factory"""
    print("=" * 50)
    print("Test 4: MemoryManager Factory")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "history.md")
        store = MemoryManager.create_store("markdown", storage_path=md_path)

        store.add_interaction("Test", "Received")
        assert store.count() == 1
        print("  [OK] Created Markdown storage via factory")

    print("  [PASS] MemoryManager test passed\n")


def test_learning_modules():
    """Test learning modules"""
    print("=" * 50)
    print("Test 5: Learning Modules")
    print("=" * 50)

    tv = TraitVector(dimensions=10)
    tv.set_trait("friendliness", 0.5)
    initial_vector = tv.vector.copy()

    # Feedback updater
    updater = FeedbackUpdater(base_learning_rate=0.1)
    updater.update(tv, feedback=0.9, context={"personality_vector": tv.vector.tolist()})
    assert not np.allclose(tv.vector, initial_vector)
    print(f"  [OK] FeedbackUpdater: friendliness {initial_vector[0]:.3f} -> {tv.vector[0]:.3f}")

    # Pattern extractor
    interactions = [
        {"user_input": "Hello?", "agent_response": "Hello!" * 30, "feedback": 0.8},
        {"user_input": "Who are you?", "agent_response": "I'm an AI assistant.", "feedback": 0.7},
        {"user_input": "How's the weather?", "agent_response": "The weather is nice today.", "feedback": 0.9},
    ]
    extractor = PatternExtractor(analysis_window=10)
    patterns = extractor.extract_patterns(interactions)
    assert "avg_feedback" in patterns
    print(f"  [OK] PatternExtractor: avg_feedback={patterns['avg_feedback']:.2f}")

    before = tv.vector.copy()
    extractor.update_from_patterns(tv, interactions)
    print(f"  [OK] Pattern update: norm changed {np.linalg.norm(before):.3f} -> {np.linalg.norm(tv.vector):.3f}")

    # RL updater
    rl = RLUpdater()
    rl.record_step(tv.vector, 0.8)
    rl.record_step(tv.vector, 0.9)
    rl.record_step(tv.vector, 0.7)
    assert rl.buffer_size == 3
    avg_return = rl.update(tv)
    assert avg_return is not None
    assert rl.buffer_size == 0
    print(f"  [OK] RLUpdater: avg_return={avg_return:.3f}")

    # Reflector (statistical mode)
    reflector = Reflector(decision_backend=None)
    result = reflector.reflect(tv, interactions)
    assert result is not None
    assert "analysis" in result
    print(f"  [OK] Reflector (statistical): {result['method']}")

    print("  [PASS] Learning test passed\n")


def test_agent_core():
    """Test Agent Core end-to-end"""
    print("=" * 50)
    print("Test 6: AgentCore End-to-End")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create modules
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

        # Multi-turn conversation
        responses = []
        for i in range(5):
            response = agent.chat(f"Test message {i + 1}")
            responses.append(response)
            print(f"  Conversation {i + 1}: {response[:50]}...")

        assert len(responses) == 5
        assert agent.memory.count() == 5
        print(f"  [OK] 5 turns completed, records: {agent.memory.count()}")

        # Provide feedback
        initial_personality = tv.vector.copy()
        agent.provide_feedback(0.9)
        agent.provide_feedback(0.2)
        agent.provide_feedback(0.8)
        print(f"  [OK] 3 feedbacks recorded")

        # Verify personality changed
        assert not np.allclose(tv.vector, initial_personality)
        print(f"  [OK] Personality updated: norm {np.linalg.norm(initial_personality):.3f} -> {np.linalg.norm(tv.vector):.3f}")

        # View personality info
        info = agent.get_personality_info()
        assert info["interaction_count"] == 5
        assert info["update_count"] > 0
        print(f"  [OK] Personality info: {info['description']}")

        # Verify persistence
        loaded = personality_store.load()
        assert loaded is not None
        print(f"  [OK] Personality persistence verified")

        # Verify evolution history
        history = personality_store.get_evolution_history()
        assert len(history) > 0
        print(f"  [OK] Evolution history: {len(history)} snapshots")

        # Verify Markdown file
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        assert "Test message 1" in md_content
        assert "Test message 5" in md_content
        print(f"  [OK] Markdown interaction history verified ({len(md_content)} bytes)")

    print("  [PASS] AgentCore test passed\n")


def test_decision_backend_switching():
    """Test decision backend dynamic switching"""
    print("=" * 50)
    print("Test 7: Decision Backend Switching")
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

        # Use backend 1
        r1 = agent.chat("Hello")
        assert "mock" in agent.decision.get_backend_info()["type"]
        print(f"  [OK] Backend 1 response: {r1[:40]}...")

        # Switch backend
        backend2 = MockDecisionBackend()
        agent.switch_backend(backend2)
        r2 = agent.chat("Goodbye")
        assert backend2._call_count == 1
        print(f"  [OK] Backend 2 response: {r2[:40]}...")

    print("  [PASS] Backend switching test passed\n")


# ==================== Main Entry ====================

def main():
    print("\n" + "=" * 60)
    print("  Chrysalis Integration Tests")
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
    print(f"  Results: {passed} passed, {failed} failed (total {len(tests)} tests)")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
