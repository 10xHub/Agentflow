# Security Policy

## Supported versions

`10xscale-agentflow` is pre-1.0 and ships from a single release line. Security
fixes are applied to the latest published release only. Pin a known-good version
in production and upgrade promptly when a security release is announced.

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately through either channel:

- **GitHub Security Advisories** (preferred): open a private report at
  https://github.com/10xHub/agentflow/security/advisories/new
- **Email:** contact@10xscale.ai (you may also CC shudiptotrafder@gmail.com)

Include as much of the following as you can:

- A description of the issue and the impact you believe it has.
- The affected version(s) and, if known, the affected module/import path
  (e.g. `agentflow.core.llm.client_factory`).
- A minimal reproduction or proof of concept.
- Any suggested remediation.

### What to expect

- **Acknowledgement** within 3 business days.
- An initial assessment and severity triage within 7 business days.
- Coordinated disclosure: we will agree on a disclosure timeline with you and
  credit you in the advisory unless you prefer to remain anonymous.

## Scope

This policy covers the `10xscale-agentflow` core Python package in this
repository. Issues in the API server (`10xscale-agentflow-cli`), the TypeScript
client, or third-party dependencies should be reported against their respective
projects, though we are happy to help route a report.

### Things that are expected behaviour, not vulnerabilities

- **Tools execute arbitrary code by design.** Tools you register with a
  `ToolNode` run with the privileges of the host process. Only register trusted
  tools and treat tool inputs derived from model output as untrusted.
- **Provider API keys are read from the environment** (`OPENAI_API_KEY`,
  `GEMINI_API_KEY`, etc.). Protecting that environment is the deployer's
  responsibility.
- **Prompt injection** against an LLM is a property of the model/application
  design. Reports demonstrating a concrete privilege escalation or data
  exfiltration path *through the framework* are in scope; generic "the model can
  be jailbroken" reports are not.

## Good practice for deployers

- Keep `IS_DEBUG=false` and `MODE=production` in production.
- Never set `ORIGINS=*` in production.
- Use a secrets manager rather than committing `.env` files.
- Constrain which tools and MCP servers an agent can reach.
