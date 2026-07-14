"""M1 (routing), M2 (config defaults), M3 (retry classification)."""

import pytest

from agentflow.core.exceptions import GraphError
from agentflow.core.graph.agent_internal.execution import AgentExecutionMixin as AgentExecution
from agentflow.core.graph.edge import Edge
from agentflow.core.graph.utils.utils import get_next_node
from agentflow.core.state import AgentState
from agentflow.utils.constants import END, DEFAULT_ANONYMOUS_USER_ID


class TestConditionalEdgeFailsLoudly:
    """M1: a broken condition must not silently misroute the run."""

    def test_raising_condition_raises_instead_of_falling_through(self):
        def broken(state):
            raise ValueError("condition blew up")

        edges = [
            Edge(from_node="a", to_node="b", condition=broken),
            Edge(from_node="a", to_node="fallback"),  # static edge
        ]

        # Previously the exception was swallowed and the graph quietly took the
        # static edge to 'fallback' -- a path nobody chose. A confident wrong
        # answer is worse than a failed run.
        with pytest.raises(GraphError) as exc:
            get_next_node("a", AgentState(), edges)

        assert "GRAPH_ROUTING_001" in str(exc.value)

    def test_working_condition_still_routes(self):
        edges = [Edge(from_node="a", to_node="b", condition=lambda s: True)]
        assert get_next_node("a", AgentState(), edges) == "b"

    def test_false_condition_falls_through_to_static_edge(self):
        edges = [
            Edge(from_node="a", to_node="b", condition=lambda s: False),
            Edge(from_node="a", to_node="fallback"),
        ]
        assert get_next_node("a", AgentState(), edges) == "fallback"

    def test_no_outgoing_edges_ends(self):
        assert get_next_node("a", AgentState(), []) == END


class TestRetryClassification:
    """M3: only genuine status codes may be classified as retryable."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Error code: 500 - internal server error", 500),
            ("HTTP 429 Too Many Requests", 429),
            ("status_code=503 upstream unavailable", 503),
            ("Overloaded [529]", 529),
            # Real SDK shapes: code leading the message, or followed by its
            # standard reason phrase. Both are genuine statuses and must retry.
            ("503 Service Unavailable. This model is experiencing high demand.", 503),
            ("Rate limited: 429 Too Many Requests", 429),
        ],
    )
    def test_real_status_codes_are_detected(self, message, expected):
        assert AgentExecution._extract_status_code(Exception(message)) == expected

    @pytest.mark.parametrize(
        "message",
        [
            # The M3 bug: each of these contains "500"/"502"/"429" as data, not as
            # a status, and used to be retried as if the server had failed.
            "Invalid request: max_tokens must be <= 500",
            "model returned 502 tokens in the completion",
            "context window exceeded: 429000 tokens",
            "embedding dimension mismatch, expected 500",
        ],
    )
    def test_numbers_in_prose_are_not_status_codes(self, message):
        assert AgentExecution._extract_status_code(Exception(message)) is None

    def test_structured_status_code_attribute_wins(self):
        exc = Exception("something went wrong")
        exc.status_code = 429
        assert AgentExecution._extract_status_code(exc) == 429

    def test_httpx_style_response_status_code(self):
        class Resp:
            status_code = 503

        exc = Exception("upstream")
        exc.response = Resp()
        assert AgentExecution._extract_status_code(exc) == 503

    def test_implausible_code_is_rejected(self):
        assert AgentExecution._extract_status_code(Exception("code 999")) is None


class TestConfigDefaults:
    """M2: the mock user id is gone."""

    def test_anonymous_default_is_not_a_fake_looking_account(self):
        assert DEFAULT_ANONYMOUS_USER_ID == "anonymous"
        assert "test" not in DEFAULT_ANONYMOUS_USER_ID
