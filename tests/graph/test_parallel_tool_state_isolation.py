"""Parallel tools must not clobber each other's state writes (audit H1).

Previously every parallel tool mutated ONE shared AgentState, and the merge blindly
`setattr` every field from each tool's view of it. Two tools writing *different*
fields still lost data: whichever merged last wrote back its own stale copy of the
field the other had just changed.
"""

from typing import Annotated

import pytest
from pydantic import Field

from agentflow.core.graph.utils.invoke_node_handler import InvokeNodeHandler
from agentflow.core.state import AgentState


def _reduce_sum(left: int, right: int) -> int:
    """Toy reducer: combine two branches' numbers instead of picking a winner."""
    return left + right


class MultiFieldState(AgentState):
    alpha: str = "init"
    beta: str = "init"
    counter: Annotated[int, _reduce_sum] = 0
    items: list[str] = Field(default_factory=list)


class TestFieldReducerDiscovery:
    def test_finds_reducer_annotated_on_field(self):
        state = MultiFieldState()
        assert InvokeNodeHandler._get_field_reducer(state, "counter") is _reduce_sum

    def test_returns_none_for_plain_field(self):
        state = MultiFieldState()
        assert InvokeNodeHandler._get_field_reducer(state, "alpha") is None


class TestBranchMerge:
    """_merge_tool_state folds a branch back in, using the pre-tool baseline."""

    def setup_method(self):
        self.handler = InvokeNodeHandler("tools", lambda: None)

    def test_disjoint_writes_from_two_branches_both_survive(self):
        """The core H1 regression: A writes alpha, B writes beta -> keep both."""
        baseline = MultiFieldState(alpha="init", beta="init")
        target = baseline.model_copy(deep=True)

        branch_a = baseline.model_copy(deep=True)
        branch_a.alpha = "written_by_a"

        branch_b = baseline.model_copy(deep=True)
        branch_b.beta = "written_by_b"

        self.handler._merge_tool_state(target, branch_a, baseline=baseline)
        self.handler._merge_tool_state(target, branch_b, baseline=baseline)

        # Under the old blind-setattr merge, branch B carried alpha="init" and
        # would have stomped A's write back to "init".
        assert target.alpha == "written_by_a"
        assert target.beta == "written_by_b"

    def test_untouched_field_never_overwrites_a_sibling(self):
        baseline = MultiFieldState(alpha="init")
        target = baseline.model_copy(deep=True)
        target.alpha = "already_set_by_sibling"

        untouched_branch = baseline.model_copy(deep=True)  # did not touch alpha
        self.handler._merge_tool_state(target, untouched_branch, baseline=baseline)

        assert target.alpha == "already_set_by_sibling"

    def test_reducer_combines_both_branches(self):
        baseline = MultiFieldState(counter=0)
        target = baseline.model_copy(deep=True)

        a = baseline.model_copy(deep=True)
        a.counter = 3
        b = baseline.model_copy(deep=True)
        b.counter = 4

        self.handler._merge_tool_state(target, a, baseline=baseline)
        self.handler._merge_tool_state(target, b, baseline=baseline)

        # With a reducer, concurrent writes are combined, not raced.
        assert target.counter == 7

    def test_conflicting_write_without_reducer_warns(self, caplog):
        baseline = MultiFieldState(alpha="init")
        target = baseline.model_copy(deep=True)

        a = baseline.model_copy(deep=True)
        a.alpha = "from_a"
        b = baseline.model_copy(deep=True)
        b.alpha = "from_b"

        self.handler._merge_tool_state(target, a, baseline=baseline)
        with caplog.at_level("WARNING"):
            self.handler._merge_tool_state(target, b, baseline=baseline)

        # Last write still wins, but the developer is told it is non-deterministic
        # and that a reducer is how to fix it.
        assert target.alpha == "from_b"
        assert any("no reducer" in r.message for r in caplog.records)

    def test_execution_meta_and_context_are_never_merged(self):
        baseline = MultiFieldState()
        target = baseline.model_copy(deep=True)
        original_node = target.execution_meta.current_node

        branch = baseline.model_copy(deep=True)
        branch.execution_meta.current_node = "hijacked"

        self.handler._merge_tool_state(target, branch, baseline=baseline)
        assert target.execution_meta.current_node == original_node


class TestSingleToolFastPath:
    @pytest.mark.asyncio
    async def test_no_baseline_means_plain_assignment(self):
        """With one tool there is no race, so merging stays a simple write."""
        handler = InvokeNodeHandler("tools", lambda: None)
        target = MultiFieldState(alpha="init")
        branch = MultiFieldState(alpha="changed")

        handler._merge_tool_state(target, branch, baseline=None)
        assert target.alpha == "changed"
