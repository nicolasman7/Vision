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
