"""Write the Swift parity fixtures: PlayInputs scenarios + what Python predicts for them.
Run: uv run python -m scripts.make_app_fixtures

Output: app/VisionTests/Fixtures/parity.json, read by app/VisionTests/PlayPredictorTests.swift.
Inputs are keyed by the Swift PlayInputs field names (enums as their raw values); anything left out is nil.
"""

import json

from pipeline.prediction import predictor

OUT = "app/VisionTests/Fixtures/parity.json"

# Swift PlayInputs field -> predictor input name
SWIFT_TO_PY = {
    "down": "down", "ydstogo": "ydstogo", "yardline100": "yardline_100",
    "qtr": "qtr", "quarterSecondsRemaining": "quarter_seconds_remaining",
    "scoreDifferential": "score_differential",
    "posteamTimeouts": "posteam_timeouts_remaining", "defteamTimeouts": "defteam_timeouts_remaining",
    "noHuddle": "no_huddle", "formationFamily": "formation_family", "isEmpty": "is_empty",
    "backfieldCount": "ftn_n_offense_backfield",
    "offOL": "off_ol", "offRB": "off_rb", "offTE": "off_te", "offWR": "off_wr",
    "defDL": "def_dl", "defLB": "def_lb", "defDB": "def_db", "box": "ftn_box", "hash": "ftn_starting_hash",
}

SHOTGUN_EMPTY = dict(formationFamily="shotgun", isEmpty=True, backfieldCount=0, offOL=5, offRB=0, offTE=1, offWR=4)
HEAVY = dict(formationFamily="under_center", isEmpty=False, backfieldCount=2, offOL=5, offRB=1, offTE=3, offWR=1,
             defDL=4, defLB=4, defDB=3, box=8)

SCENARIOS = {
    "nothing known": {},
    "3rd & 9 shotgun empty": dict(down=3, ydstogo=9, yardline100=55, **SHOTGUN_EMPTY),
    "1st & 1 under center, 3 TE": dict(down=1, ydstogo=1, yardline100=1, **HEAVY),
    "red zone 2nd & 6": dict(down=2, ydstogo=6, yardline100=12, qtr=2, quarterSecondsRemaining=300,
                             formationFamily="shotgun", offRB=1, offTE=1, offWR=3),
    "goal to go 1st & 8": dict(down=1, ydstogo=8, yardline100=8),
    "only formation: pistol": dict(formationFamily="pistol"),
    "only formation: under center": dict(formationFamily="under_center"),
    "only down/distance 2nd & 10": dict(down=2, ydstogo=10),
    "only down/distance 3rd & 2": dict(down=3, ydstogo=2),
    "hash left, shotgun": dict(hash="L", formationFamily="shotgun"),
    "hash middle only": dict(hash="M"),
    "hash right, heavy": dict(hash="R", **HEAVY),
    "2-minute drill trailing": dict(down=1, ydstogo=10, yardline100=70, qtr=4, quarterSecondsRemaining=110,
                                    scoreDifferential=-6, posteamTimeouts=1, defteamTimeouts=3, noHuddle=True,
                                    formationFamily="shotgun"),
    "4th quarter kneel-down lead": dict(down=1, ydstogo=10, yardline100=40, qtr=4, quarterSecondsRemaining=90,
                                        scoreDifferential=10, posteamTimeouts=3, defteamTimeouts=0),
    "Q1 opening drive": dict(down=1, ydstogo=10, yardline100=75, qtr=1, quarterSecondsRemaining=900,
                             scoreDifferential=0, posteamTimeouts=3, defteamTimeouts=3),
    "Q3 only clock (no qtr)": dict(quarterSecondsRemaining=600),
    "only qtr": dict(qtr=3),
    "overtime": dict(down=2, ydstogo=7, yardline100=35, qtr=5, quarterSecondsRemaining=400),
    "personnel only 11": dict(offOL=5, offRB=1, offTE=1, offWR=3, defDL=4, defLB=2, defDB=5),
    "box 6 + no huddle": dict(box=6, noHuddle=True),
    "everything known": dict(down=3, ydstogo=4, yardline100=33, qtr=2, quarterSecondsRemaining=45,
                             scoreDifferential=3, posteamTimeouts=2, defteamTimeouts=1, noHuddle=False,
                             formationFamily="shotgun", isEmpty=False, backfieldCount=1,
                             offOL=5, offRB=1, offTE=1, offWR=3, defDL=4, defLB=2, defDB=5, box=6, hash="L"),
}


def main():
    bundle = predictor.load()
    fixtures = []
    for name, inputs in SCENARIOS.items():
        out = predictor.predict(bundle, **{SWIFT_TO_PY[k]: v for k, v in inputs.items()})
        fixtures.append({"name": name, "inputs": inputs, **out})
    with open(OUT, "w") as f:
        json.dump(fixtures, f, indent=1)
    print(f"wrote {len(fixtures)} scenarios to {OUT}")


if __name__ == "__main__":
    main()
