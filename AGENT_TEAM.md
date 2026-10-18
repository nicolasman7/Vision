# Agent Team Playbook

Drop this file into the root of any project. Then tell Claude: **"Read AGENT_TEAM.md and run the project with an agent team."** To make it permanent, add the line `@AGENT_TEAM.md` to the project's `CLAUDE.md`.

It tells the main Claude session how to act as **lead**:
- hand work to small, labeled subagents that each have one narrow job
- show all of them live in a little game-style dashboard
- keep a log of everything they did

---

## Rule zero: the dashboard comes first

**Before any other subagent starts, build the dashboard and open it for the user.** The user wants to watch every agent from the very first second. No planning agent, no worker, nothing runs until the dashboard is up.

**Theme it on this project, not on football or any other default.** Read the README, CLAUDE.md and the code first, then pick a simple world that fits:

| Project | World | Stations (examples) |
|---|---|---|
| Recipe app | Kitchen | Pantry (data), Stove (ML/backend), Tasting table (tests), Menu board (docs), Plating (UI) |
| Compiler | Factory floor | Parser line, Optimizer press, QA bay, Manual room, Control panel |
| Web shop | Store | Stockroom, Checkout, Back office, Shop window, Help desk |
| Robotics | Workshop | Parts bin, Lab bench, Test track, Blueprint wall, Control room |
| Sports analytics | Field or court | Locker room, Film room, Practice field, Playbook library, Press box |

Keep it simple and charming. Make one world and get it working, not an elaborate game. If `tools/agent-hq/index.html` already exists, reuse it and only add stations if needed.

**Who builds it:** a single UI subagent (or the lead itself). It reports its own status while building, so it's the first character to appear on the board.

---

## 1. Status protocol (copy exactly)

```
tools/agent-hq/
  report.py              # agents call this to update status + log
  index.html             # the dashboard (one file)
  status/<id>.json       # live status, one file per agent   (git-ignored)
  logs/activity-YYYY-MM-DD.jsonl   # every update ever made   (committed)
  logs/rounds/round-N.md           # lead's summary of each round (committed)
```

Add to `.gitignore`:
```
tools/agent-hq/status/
```

`tools/agent-hq/report.py`:
```python
#!/usr/bin/env python3
"""Update one agent's status for the Agent HQ game, and append it to the activity log.

Usage: python3 tools/agent-hq/report.py <id> <name> <role> <state> "<task>"
state: idle | working | done | blocked
"""
import json
import os
import sys
import time
from pathlib import Path

STATES = {"idle", "working", "done", "blocked"}

agent_id, name, role, state, task = sys.argv[1:6]
if state not in STATES:
    sys.exit(f"state must be one of {sorted(STATES)}")

here = Path(__file__).parent
record = {"id": agent_id, "name": name, "role": role, "state": state, "task": task, "updated": time.time()}

# Live status: one file per agent, read by the game. Git-ignored.
status_dir = here / "status"
status_dir.mkdir(exist_ok=True)
tmp = status_dir / f".{agent_id}.tmp"
tmp.write_text(json.dumps(record))
os.replace(tmp, status_dir / f"{agent_id}.json")  # atomic: the game never reads a half-written file

# History: one JSON line per update, one file per day. Committed, so the timeline survives.
log_dir = here / "logs"
log_dir.mkdir(exist_ok=True)
with open(log_dir / f"activity-{time.strftime('%Y-%m-%d')}.jsonl", "a") as f:
    f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}) + "\n")
```

Every agent reports like this:
```
python3 tools/agent-hq/report.py wrangler "Data Wrangler" "Data" working "Joining sales with inventory"
```
- **When:** at the start, at every milestone (at least 4–6 times per job), and at the end.
- **Task text:** under 60 characters. Say what the agent is doing right now, not what it did.
- **End state:** `done "<headline result>"` when its work is finished, then the lead benches it (see §3). Use `blocked "<reason>"` when it's stuck.
- **Roles** are short words that map to stations, e.g. Lead, Data, ML, Docs, UI, Vision. Pick a set that fits the project.

---

## 2. Dashboard spec (what the UI agent builds)

**Tech**
- One file, `tools/agent-hq/index.html`: vanilla HTML/CSS/JS plus a `<canvas>`.
- **No CDNs, external scripts or web fonts.** It must work offline.
- Roughly 400–700 lines.
- Serve it with `python3 -m http.server 8765 -d tools/agent-hq` and open http://localhost:8765.
- Start the server in the background and leave it running for the whole session.

**Finding agents**
- Fetch `status/` (http.server returns a directory listing).
- Collect the `href`s ending in `.json`, then fetch each one with `cache: "no-store"`.
- Poll every 2 seconds.
- **Never hard-code agents.** New ids and roles must just appear.
- Friendly messages for an empty or missing `status/` ("No agents yet") and for an unreachable server (show the command that starts it).

**The world**
- A top-down pixel-art scene in the project's theme.
- One **station** per role, with a themed name and a small animated prop.
- One **generic station** for unknown roles.
- A **bench** for idle agents.
- A **done zone** for finished agents.
- An **entrance** where new agents walk in from.
- Spread agents who share a spot into slots, so up to about 4 can share without overlapping.

**Characters**
- Little pixel people drawn with rectangles; no image files.
- A distinct color per agent, derived from a hash of its id.
- A **name label above the head**.
- Smooth walking with leg animation. Never teleport.

**States**
| State | Behavior |
|---|---|
| working | Walks to its role's station, does a working animation (tool swing, sparkles), and shows a speech bubble with the task |
| idle | Sits on the bench, with a "…" bubble |
| done | Walks to the done zone, celebrates (jump, ✓, confetti) |
| blocked | Stands still with a red "!" and a red bubble showing the reason |
| stale | Working, but `updated` is over 10 minutes old: "zzz" and sitting down |

When an agent's task or state changes, its bubble pops or flashes so the user notices.

**Roster panel**
- An HTML panel on the right, not drawn on the canvas, so the text stays crisp. Below 820px wide it moves under the canvas.
- Two tab buttons, **"Active (N)"** and **"Bench (N)"**, with live counts.
  - Active = anything not idle (working, blocked, stale, done).
  - Bench = idle.
  - Default to Active. Remember the choice in `localStorage`, wrapped in try/catch so the page still works when storage is blocked.
  - Empty states: "Nobody's working right now" and "Bench is empty".
- Each row: color dot, name, role, state badge, task, and "updated Xs ago".
- Clicking a row or a character opens a details card: name, role, station, state, full task, time since update, id.

**Sharpness:** scale the canvas to the window, handle `devicePixelRatio`, use whole-number scaling, and set `image-rendering: pixelated`.

**Checks before calling it done**
- `node --check` on the extracted script.
- `curl` the page and `status/` to confirm the listing matches the parser.
- Cycle a fake agent `testbot` through working → blocked → done → idle (plus a stale timestamp and an unknown role), then **delete `status/testbot.json`**.
- A headless browser screenshot if one is available.
- **Never stop or restart the live server on 8765.** Test on 8766 and stop only that one.
- Tell the user to refresh once after any change to `index.html`. Statuses update live; page code doesn't.

---

## 3. How the team works

### Delegate by default (the lead is not a worker)
Any real work goes to a labeled subagent, so it runs in parallel and shows up on the dashboard. That includes writing code, running experiments, research, downloads, docs and dashboard changes.
- **Independent jobs launch in the same turn.**
- **Reuse benched agents first.**
- **The lead itself does only lead-only work:**
  - planning and briefing
  - reviewing handoffs (running the tests, spot-checks)
  - installing dependencies
  - git
  - project memory (`CLAUDE.md`) and round logs
- Quick answers to the user's questions need no agents.
- If the lead notices it is writing feature code or doing research itself, it should stop and hand the job to an agent.

**Make it automatic.** A `UserPromptSubmit` hook in the project's `.claude/settings.json` re-injects this reminder on every user message. Copy it as-is:
```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "printf '%s' '{\"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": \"AGENT TEAM REMINDER (AGENT_TEAM.md): you are the lead. Delegate real work to labeled subagents in parallel (reuse benched agents first) and keep them visible on Agent HQ via tools/agent-hq/report.py. Do yourself only lead-only work: planning, reviewing hand-offs, installing dependencies, git, and CLAUDE.md/round logs. Quick answers to questions need no agents.\"}}'",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```
If `.claude/settings.json` already exists, merge this into its `hooks` rather than replacing the file. A new hook takes effect from the next session; in an already-running session, open `/hooks` once to load it.

### Roles
- **Lead (the main session, "Coach").** It plans, hands out jobs, reviews every handoff, installs dependencies, and owns `CLAUDE.md` and the round logs. It shows up on the board as `lead`.
- **Strategist**, a read-only planning agent (the `Plan` agent type), is the first agent after the dashboard in every round.
  - It reads the project and runs quick checks (header requests, small samples).
  - It designs the team for the round.
  - It **builds nothing**.
- **Specialists**, each with exactly one narrow job, a fun but clear name, and a label. For example: Data Wrangler, Modeler, Field Surveyor, Film Scout, Bucket Scout.
- **Teacher**, a docs agent that always runs last. It writes up the round for the user in the project's progress or learning doc.

### A round, step by step
1. **The dashboard is up and open** (rule zero).
2. **Strategist plans.** Give it:
   - the current state
   - the issues to tackle
   - the hard constraints (below)
   - the deliverable, which is:
     - key ideas: what it checked, found and rejected
     - a **roster** of 3–6 agents, each with id, Name, role, a one-line niche, a full self-contained brief, dependencies, and which ones can run in parallel
     - a **conflict plan**: who owns each file
     - a list of things **blocked on the user**
3. **The lead reviews the plan and does phase 0 itself:**
   - installs every dependency the round needs (only the lead runs `uv add`, `npm install` and the like, because parallel installs clash on lockfiles)
   - makes a backup or commit if asked
   - adds dashboard stations if needed
4. **The lead launches phases.**
   - Agents within a phase run in parallel, in the background.
   - Set each agent's status before launch, so it appears immediately. Use `idle "Waiting for X"` for agents in a later phase.
5. **The lead reviews every handoff before the next phase.**
   - Run the tests.
   - Spot-check the outputs: numbers that look too good, forbidden columns or files, leftover test agents.
   - Confirm `git status` shows no data, secrets or large binaries.
   - Only then start the agents that depend on it.
6. **Teacher writes it up.** Give it condensed versions of every agent's report. Every number it uses must trace to a file or a report.
7. **The lead closes the round:**
   - updates the project memory (`CLAUDE.md` current state and session log)
   - writes `tools/agent-hq/logs/rounds/round-N.md`
   - benches everyone, itself included
   - gives the user a short summary plus the list of things waiting on them

### The bench
- **Never delete an agent when it finishes.** Set it to `idle "On the bench: ready for <niche> jobs"`.
- **Reuse before you create.** If a benched agent's niche fits the next job, resume it (keeping its context) instead of spawning a new one.
- The user can always see the reserves under the **Bench** tab.

### Hard constraints for every agent
- **No tasks that need the user**, including:
  - creating accounts or logging in
  - accepting terms or licenses
  - supplying private or copyrighted media
  - entering secrets
- **Decisions belong to the user.** Agents may research options and write a doc with evidence and a clearly labeled **Recommendation**, but they never lock a choice into the code. Keep these docs in a folder like `docs/research/`.
- Never read `.env` or other secret files. Never `git commit` or `git push` unless the user asked in this conversation.
- **File ownership:**
  - Each agent edits only the files listed as its own.
  - Shared modules are **add-only**: never rename or remove names another agent imports.
  - Generated data stays out of git.
- **Never stop the dashboard server.**
- House style: small readable code, no frameworks or abstractions nobody asked for, comments where a choice isn't obvious, and one runnable test for each piece of non-trivial logic.

### Brief template (paste for each agent)
```
You are **<Name>** (id `<id>`, role `<Role>`), a subagent specializing in <niche>.
Project: <path>. Your lead is "Coach". Read CLAUDE.md first.

Status (the user watches you live; report at least 5 times):
`python3 tools/agent-hq/report.py <id> "<Name>" "<Role>" <working|done|blocked> "<task <60 chars>"`
When finished: `idle "On the bench: ready for <niche> jobs"`.

Rules: <hard constraints above, trimmed to what applies>. Don't stop the server on 8765.

Goal: <one sentence>
Facts already checked (don't re-discover): <bullets from Strategist>
Do: <numbered steps with exact inputs/outputs>
Done when: <tests pass, numbers reproduce, files exist>
You own: <files>. Don't touch anything else.
Final report to Coach: <what to include, short>.
```

---

## 4. Logs

- **Activity log (automatic).** `report.py` appends every status update to `tools/agent-hq/logs/activity-YYYY-MM-DD.jsonl`, so you get a timestamped timeline of who did what. Commit it.
- **Round logs (the lead writes them).** After each round, write `tools/agent-hq/logs/rounds/round-N.md` with:
  - the date
  - the goal
  - one section per agent: job, result, files touched, what the lead's review found
  - what's waiting on the user
  - keep it to a page
- **Commits.** When the user wants commits, make one per agent or per phase, with a message saying which agent did what. That way the git history mirrors the team.

---

## 5. Quick start checklist

1. Read the project (README, CLAUDE.md, code layout).
2. Create `tools/agent-hq/report.py` (§1), add `status/` to `.gitignore`, and add the delegation reminder hook (§3) to `.claude/settings.json`.
3. Put the lead on the board: `python3 tools/agent-hq/report.py lead "Coach" "Lead" working "Setting up HQ"`.
4. Build the themed dashboard (§2), start the server in the background, open it for the user, and tell them the URL.
5. Launch Strategist (§3), review its plan, then run the phases.
6. Close the round: write the round log, bench everyone, summarize, and list what needs the user.
