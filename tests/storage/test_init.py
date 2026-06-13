import pytest
import agentflow.storage


def test_storage_lazy_exports():
    assert agentflow.storage.make_agent_memory_tool is not None
    assert agentflow.storage.make_user_memory_tool is not None
    assert agentflow.storage.memory_tool is not None

    with pytest.raises(AttributeError):
        _ = agentflow.storage.invalid_attribute_name_xxx
