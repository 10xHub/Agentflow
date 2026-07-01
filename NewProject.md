Decisions locked: self-contained project inside the repo (movable later), built on your local agentflow; data from live web search (Gemini grounding) + page fetch only; Gemini for everything; sequential supervisor master/worker, one company at a time; discover ~15-20 and deep-profile top 5; markdown-only deliverables; local persistence for search cache + dedup registry + resumable runs + run history.

On the "2-3 approaches" point: our Q&A already ruled out the alternatives (paid DB, single autonomous agent, parallel fan-out, SupervisorTeamAgent-decides-everything). The one genuinely open choice is how to make runs resumable locally without Postgres, which I'll flag inside the design.

Here's the design, in sections. Tell me what to change in any of them.

1. Project shape & location

Agentflow/
└── ma-scout/                      # new self-contained project (rename freely)
    ├── pyproject.toml             # depends on local agentflow via path/editable
    ├── ma_scout/
    │   ├── cli.py                 # CLI entrypoint (argparse/click)
    │   ├── config.py              # models, limits, paths, API key loading
    │   ├── criteria.py            # parse NL prompt -> CriteriaProfile
    │   ├── pipeline.py            # master orchestrator (the sequential loop)
    │   ├── discovery.py           # discovery agent + candidate scoring/ranking
    │   ├── profiler.py            # per-company supervisor team (RESEARCHER/ANALYST/WRITER)
    │   ├── schemas.py             # Pydantic: CriteriaProfile, Candidate, CompanyProfile
    │   ├── render.py              # CompanyProfile -> markdown (10 sections) + shortlist index
    │   └── store.py               # local SQLite: cache, dedup, run history, run progress
    ├── runs/                      # output: runs/<run-id>/<company>.md + shortlist.md
    └── tests/
Command: ma-scout "find IT consultancies in Catalonia ~€4.5M revenue, ~31% margin, buyers: IT consultancies/MSPs/buy-and-build" --discover 18 --profile 5

Approve folder name/location? (default Agentflow/ma-scout/)

2. Data flow (master/worker, sequential)

criteria prompt
   │
   ▼
[criteria.py]  LLM parses NL -> CriteriaProfile (sector, revenue band, margin, geo, buyer types)
   │
   ▼
[discovery.py]  Discovery agent (Gemini Flash + google_web_search + fetch_url)
   │            -> CandidateList (name, website, location, why_match, source_urls)
   │            -> dedup vs registry, LLM fit-score 0-100 vs criteria, rank, take top N(=5)
   ▼
[pipeline.py]  MASTER LOOP  (sequential, one company at a time, resumable)
   │   for each shortlisted company (skip ones already done this run):
   │      └── [profiler.py] per-company SupervisorTeamAgent invocation:
   │             SUPERVISOR (Gemini) coordinates:
   │               • RESEARCHER  -> google_web_search + fetch_url, gathers evidence
   │               • ANALYST     -> derives thesis/exit-signals/feasibility/risks + confidence
   │               • WRITER      -> emits structured CompanyProfile (output_schema)
   │             max_rounds caps work per company
   │      └── render to runs/<run-id>/<company>.md ; record done in SQLite
   ▼
[render.py]  write runs/<run-id>/shortlist.md (summary table + links)
Within-company runs use InMemoryCheckpointer (ephemeral, fine — each company is one short invocation). Resumability lives at the master level: SQLite records which companies are completed for a run, so re-invoking with --resume <run-id> skips finished companies. This avoids needing a Postgres checkpointer while still being fully local and robust.

Approve the flow + the "resume at company granularity" approach?

3. The 10-section profile (data contract)
A Pydantic CompanyProfile is the internal contract (guarantees all 10 sections via structured output + repair), then rendered to markdown. Fields map 1:1 to your sections: basic_snapshot, investment_thesis, exit_signals, market_context, strategic_value, deal_feasibility, integration_snapshot, key_risks, cost_of_inaction, leadership_ownership. Plus cross-cutting: fit_score, confidence (per data-quality, since financials are web-inferred), and sources (URLs backing claims).

The full structured object is stored in SQLite run history (so the data exists even though only .md is written to disk, per your choice).

Approve the schema (esp. adding confidence + sources given web-only data)?

4. Local persistence (single SQLite file)
One ma_scout.db (SQLite, no server) holds four things you asked for:

search_cache — keyed by query/url, with TTL, to skip repeat search/fetch and cut cost
company_registry — dedup across runs (by normalized name + domain); future runs skip/update
runs + run_companies — run history (criteria, timestamps, status) and per-company progress (enables --resume); stores the structured profile JSON
Approve SQLite as the single local store?

5. Testing & guardrails
Unit tests with mocked LLM/tools (agentflow TestAgent/mocks) — no live calls in CI.
Markdown rendering + SQLite layer tested directly (deterministic).
Cost/safety guardrails: hard max_rounds per company, discovery/profile caps, search cache, and a --dry-run that does discovery+scoring only.
Honest-data posture: every profile carries confidence + sources; the report explicitly marks financials as estimated.
Approve testing approach?

If sections 1-5 look right (or with your edits), I'll write this up as the spec doc, you review it, then we move to the implementation plan. What would you like to change?