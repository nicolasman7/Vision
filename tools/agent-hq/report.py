#!/usr/bin/env python3
"""Update one agent's status for the Agent HQ game.

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

status_dir = Path(__file__).parent / "status"
status_dir.mkdir(exist_ok=True)
tmp = status_dir / f".{agent_id}.tmp"
tmp.write_text(json.dumps({"id": agent_id, "name": name, "role": role,
                           "state": state, "task": task, "updated": time.time()}))
os.replace(tmp, status_dir / f"{agent_id}.json")  # atomic: the game never reads a half-written file
