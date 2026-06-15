Agentflow Core Python SDK — Production Readiness Review
Scope: the 10xscale-agentflow package (v0.7.5.1) in agentflow/. Overall this is a mature, well-structured framework with strong CI/CD release automation, a comprehensive test suite (138 files), a clean exception hierarchy, and genuinely good OTEL support. The gaps below are what stand between it and "production-grade SDK that other teams depend on."

Blockers (fix before claiming production-stable)
1. No py.typed marker — the package ships untyped to consumers.
Confirmed missing: no agentflow/agentflow/py.typed and no package-data rule in pyproject.toml. Despite extensive internal type hints, PEP 561 means downstream mypy/pyright see Any for every Agentflow symbol. For a "Production/Stable"-classified library this is the single highest-leverage fix: add the empty marker plus a [tool.setuptools.package-data] entry.

2. Core dependencies are unpinned.
pyproject.toml:51,64-66 lists pydantic, PyYAML, python-dotenv, and pydantic-ai with no version bounds at all. A Pydantic v2→v3 or breaking pydantic-ai release will silently break installs in the field. At minimum set lower bounds (pydantic>=2,<3). The uv.lock protects this repo's own builds but does nothing for pip install 10xscale-agentflow users.

3. README claims native Anthropic/Claude support that does not exist.
README headline (lines ~17, 24, 190) advertises native Anthropic and ANTHROPIC_API_KEY, but client_factory.py detect_provider only resolves google or openai. A user running detect_provider("claude-3-opus") gets openai and a construction failure. Either remove the claim or document it as "via OpenAI-compatible endpoint only." (Note: the README import-path drift that CLAUDE.md warns about appears to have been fixed — examples now use correct agentflow.core.* paths. CLAUDE.md's "broken examples" note is itself stale.)

High priority
4. No default timeout on LLM calls.
client_factory.py:34 accepts a timeout kwarg but enforces no default, and there is no top-level timeout wrapping invoke/ainvoke. A hung provider connection blocks indefinitely. Add a sane default client timeout and a per-request ceiling.

5. mypy is configured but never runs.
pyproject.toml has [tool.mypy], but it is absent from both .pre-commit-config.yaml and .github/workflows/ci.yml (confirmed: zero mypy references in either). It is dead config. Either wire it into CI or stop advertising type safety. This pairs directly with #1.

6. CI tests a single Python version on a single OS.
ci.yml runs only Python 3.13 on ubuntu-latest, yet the package claims >=3.12 and classifies 3.12/3.13. 3.12 is untested. Add a 3.12/3.13 matrix; consider macOS.

Medium priority
7. Silent exception swallowing in media + callback paths. Broad except Exception with debug-only logging in media_resolver.py (e.g. lines 100, 193, 234) and throughout callbacks.py hides real failures in production. Narrow these or at least log at warning with context.

8. No __aenter__/__aexit__ on CompiledGraph. Cleanup relies on callers remembering aclose(). Publisher backends (Kafka/RabbitMQ/Redis) may leak connections if shutdown raises. Add the async-context-manager protocol to the top-level graph.

9. Missing governance/policy files. No SECURITY.md (no vuln disclosure path) and no CONTRIBUTING.md. Both are table-stakes for an OSS framework inviting external dependence.

10. No migration guide or deprecation policy. The 0.5→0.7 flat→nested import restructure (agentflow.state → agentflow.core.state) has no MIGRATION.md, and there's no structured deprecation mechanism. changelogs.md is unstructured (no dates, no Keep-a-Changelog format). For a stable API, document the deprecation contract.

11. ConsolePublisher uses print(). console_publisher.py:61 writes to stdout. Acceptable for a console publisher by design, but worth confirming it never runs by default in a server context.

Lower priority / nice-to-have
No circuit breaker to complement fallback_models — repeated failures to a dead provider retry every call.
No secret masking in logs; API keys/auth headers could surface in debug output.
No Dependabot/CodeQL beyond Bandit.
Coverage gate at 70% is low for a production framework; 80%+ is a better bar.
normal_tests/ (12 files) excluded from discovery with no documented rationale.
What's already solid (don't regress)
Release automation: tag-gated, OIDC trusted PyPI publishing with version verification (release.yml).
Exception hierarchy with error_code/context/to_dict() and an explicit TransientStorageError for retryability.
OTEL tracing with gen_ai semantic conventions and three observability levels.
Proper library logging (getLogger(__name__) + NullHandler, no stdout pollution in the core path).
Parallel tool execution genuinely works (invoke_node_handler.py:182).
Comprehensive __all__ exports across all subpackages.
Suggested order of attack
py.typed + dependency pinning + fix the Anthropic README claim (an afternoon; ships in next patch).
LLM timeouts + wire mypy into CI + 3.12 matrix.
SECURITY.md, CONTRIBUTING.md, MIGRATION.md.
Tighten exception handling and add the graph async-context-manager protocol.
Items 1-3 are mostly mechanical and would move the package materially toward production-grade. Want me to implement the quick wins in #1 (add py.typed, set dependency bounds, correct the README)?




4. lets add keep a default, using env or globally we can change it
5. lets add
6. fine for now, will focus on later
7. lets fix it
8. lets add
9. lets add
10. skip for now
11. add it

No circuit breaker to complement fallback_models — repeated failures to a dead provider retry every call.
No secret masking in logs; API keys/auth headers could surface in debug output.
No Dependabot/CodeQL beyond Bandit.
Coverage gate at 70% is low for a production framework; 80%+ is a better bar.