# """
# test1 — Trajectory & behaviour evaluation tests (compact).

# Verifies tool calling, response output, no-tool path, trajectory
# collector fields, and collector-backed evaluation in 6 focused tests.

# Run:
#     pytest examples/evaluation/test1/ -v -s
# """

# import json

# import pytest

# from agentflow.qa.evaluation import AgentEvaluator, CriterionConfig, EvalConfig, MatchType
# from agentflow.core.state import Message

# from .samples import CAPITAL_QUESTION, LONDON, NYC


# # ---------------------------------------------------------------------------
# # 1. Tool call + trajectory — single weather query
# # ---------------------------------------------------------------------------


# class TestToolAndTrajectory:
#     @pytest.mark.asyncio
#     async def test_tool_called_and_trajectory_correct(self, compiled_graph, collector):
#         """Weather query: get_weather called + trajectory matches expected sequence."""
#         config = EvalConfig(
#             criteria={
#                 "tool_name_match_score": CriterionConfig(threshold=1.0),
#                 "tool_trajectory_avg_score": CriterionConfig.trajectory(
#                     threshold=1.0,
#                     match_type=MatchType.EXACT,
#                 ),
#             },
#             reporter={"enabled": True},
#         )
#         evaluator = AgentEvaluator(compiled_graph, collector, config=config)
#         result = await evaluator.evaluate_case(NYC)
#         assert result.passed, _fail(result)


# # ---------------------------------------------------------------------------
# # 2. Response quality + tool result
# # ---------------------------------------------------------------------------


# class TestResponseOutput:
#     @pytest.mark.asyncio
#     async def test_response_matches_and_tool_result_attached(self, compiled_graph, collector):
#         """LLM-judge response match ≥ 0.7; every tool call has a result."""
#         config = EvalConfig(
#             criteria={"response_match_score": CriterionConfig.llm_judge(threshold=0.7)},
#             reporter={"enabled": True},
#         )
#         evaluator = AgentEvaluator(compiled_graph, collector, config=config)
#         result = await evaluator.evaluate_case(NYC)
#         assert result.passed, _fail(result)
#         assert result.actual_tool_calls, "No tool calls captured"
#         assert all(tc.result is not None for tc in result.actual_tool_calls)


# # ---------------------------------------------------------------------------
# # 3. No-tool path
# # ---------------------------------------------------------------------------


# class TestNoToolCalls:
#     @pytest.mark.asyncio
#     async def test_no_tool_and_keyword_present(self, compiled_graph, collector):
#         """General knowledge: no tool called, response contains 'Paris'."""
#         config = EvalConfig(
#             criteria={
#                 "tool_name_match_score": CriterionConfig(threshold=1.0),
#                 "contains_keywords": CriterionConfig.contains_keywords(
#                     keywords=["Paris"],
#                     threshold=1.0,
#                 ),
#             },
#             reporter={"enabled": True},
#         )
#         evaluator = AgentEvaluator(compiled_graph, collector, config=config)
#         result = await evaluator.evaluate_case(CAPITAL_QUESTION)
#         assert result.passed, _fail(result)


# # ---------------------------------------------------------------------------
# # 4. Trajectory collector — execution path verification
# # ---------------------------------------------------------------------------


# class TestTrajectoryCollector:
#     @pytest.mark.asyncio
#     async def test_collector_captures_execution(self, trajectory_app):
#         """Node visits, tool calls, timing, and to_dict() serialisation."""
#         app, coll = trajectory_app
#         coll.reset()
#         inp = {"messages": [Message.text_message(LONDON.conversation[0].user_content.get_text())]}
#         await app.ainvoke(inp, config={"thread_id": "test-collector", "recursion_limit": 10})

#         # Node visits
#         assert "MAIN" in coll.node_visits
#         assert coll.node_visits.count("MAIN") >= 2
#         # Tool calls
#         assert coll.tool_calls
#         assert any(tc.name == "get_weather" for tc in coll.tool_calls)
#         assert all(tc.result is not None for tc in coll.tool_calls)
#         # Timing
#         assert coll.duration > 0
#         # to_dict serialisation
#         d = coll.to_dict()
#         json.dumps(d, default=str)  # must not raise
#         assert "trajectory" in d and "tool_calls" in d and "node_visits" in d

#     @pytest.mark.asyncio
#     async def test_no_tool_path_clean(self, trajectory_app):
#         """General query: MAIN once, no tools, final_response non-empty."""
#         app, coll = trajectory_app
#         coll.reset()
#         inp = {
#             "messages": [
#                 Message.text_message(CAPITAL_QUESTION.conversation[0].user_content.get_text())
#             ]
#         }
#         await app.ainvoke(inp, config={"thread_id": "test-no-tool-path", "recursion_limit": 10})

#         assert "MAIN" in coll.node_visits
#         assert len(coll.tool_calls) == 0
#         assert coll.final_response and coll.final_response.strip()


# # ---------------------------------------------------------------------------
# # 5. Collector-backed evaluator — end-to-end
# # ---------------------------------------------------------------------------


# class TestCollectorEvaluator:
#     @pytest.mark.asyncio
#     async def test_evaluator_end_to_end(self, compiled_graph, collector):
#         """AgentEvaluator with collector: tool match passes, response populated."""
#         evaluator = AgentEvaluator(compiled_graph, collector=collector)
#         result = await evaluator.evaluate_case(NYC)
#         assert result.passed
#         assert result.actual_response
#         assert collector.final_response


# # ---------------------------------------------------------------------------
# # 6. Node response tracking
# # ---------------------------------------------------------------------------


# class TestNodeResponses:
#     @pytest.mark.asyncio
#     async def test_node_responses_and_to_dict(self, trajectory_app):
#         """node_responses captured; to_dict includes node_responses & final_response."""
#         app, coll = trajectory_app
#         coll.reset()
#         inp = {"messages": [Message.text_message("What is the weather in Amsterdam?")]}
#         await app.ainvoke(inp, config={"thread_id": "traj-nr", "recursion_limit": 10})

#         assert len(coll.node_responses) > 0
#         assert any(nr.has_tool_calls for nr in coll.node_responses)
#         assert coll.final_response

#         d = coll.to_dict()
#         json.dumps(d, default=str)
#         assert "node_responses" in d and "final_response" in d
#         assert len(d["node_responses"]) > 0


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------


# def _fail(result) -> str:
#     lines = [f"Case '{result.eval_id}' failed:"]
#     for cr in result.criterion_results:
#         if not cr.passed:
#             lines.append(f"  [{cr.criterion}] score={cr.score:.2f}")
#     return "\n".join(lines)
