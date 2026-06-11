"""记忆层 V2 集成测试"""
import sys
import os
from unittest.mock import patch

# Add backend/app/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from dotenv import load_dotenv
load_dotenv()


def test_registry_loading():
    """测试 1: MemoryTypeRegistry 能正确加载 YAML。"""
    from service.memory_type_registry import get_memory_type_registry
    registry = get_memory_type_registry()
    names = registry.list_names()
    assert "research_finding" in names, f"Expected research_finding in {names}"
    assert "industry_entity" in names, f"Expected industry_entity in {names}"

    finding = registry.get("research_finding")
    assert finding is not None
    assert len(finding.fields) == 5
    assert finding.fields[0].name == "topic"
    assert finding.fields[0].merge_op.value == "immutable"
    print("PASS test_registry_loading")


def test_merge_operations():
    """测试 2: 四种合并策略。"""
    from service.memory_merge import apply_merge
    from schemas.memory_config import MergeOp

    # Replace
    assert apply_merge("old", "new", MergeOp.REPLACE) == "new"
    # Immutable (has existing value)
    assert apply_merge("old", "new", MergeOp.IMMUTABLE) == "old"
    # Immutable (no existing value - first write)
    assert apply_merge(None, "new", MergeOp.IMMUTABLE) == "new"
    # Sum
    assert apply_merge("line1", "line2", MergeOp.SUM) == "line1\nline2"
    # Patch
    result = apply_merge("市占率约35%", "SEARCH: 35%\nREPLACE: 42%", MergeOp.PATCH)
    assert "42%" in result
    # None current
    assert apply_merge(None, "new", MergeOp.PATCH) == "new"
    print("PASS test_merge_operations")


def test_build_memory_context_empty():
    """测试 3: 无记忆时 build_memory_context 返回空 dict。"""
    # MemoryService.__init__ 需要这些环境变量来创建 OpenAI client
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-dummy-key")
    os.environ.setdefault("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    os.environ.setdefault("DASHSCOPE_MODEL", "qwen-plus")

    from service.memory_service import get_memory_service
    service = get_memory_service()

    # Mock retrieve_memories to return empty list (no Milvus needed)
    with patch.object(service, "retrieve_memories", return_value=[]):
        result = service.build_memory_context(
            user_id="nonexistent-test-user-id",
            current_query="新能源汽车",
        )
    assert isinstance(result, dict)
    assert "context_text" in result
    assert "memory_ids" in result
    assert result["context_text"] == ""
    assert result["memory_ids"] == []
    print("PASS test_build_memory_context_empty")


if __name__ == "__main__":
    test_registry_loading()
    test_merge_operations()
    test_build_memory_context_empty()
    print("\nAll tests passed")
