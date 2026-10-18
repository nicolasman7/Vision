"""NFL field geometry in yards, Big Data Bowl (BDB) convention.

x runs 0-120 along the field (0 = back of the left end zone), y runs 0-53.33 across
(0 = one sideline). Lines are treated as zero-width: a landmark is the centre of the
painted line, not its edge.
"""

YARD_M = 0.9144  # exact, by definition of the international yard

# NFL Rulebook, Rule 1 Sec. 1: field is 360 ft x 160 ft including 10-yd end zones.
LENGTH = 120.0
WIDTH = 160 / 3  # 53.33 yd; BDB docs give y in 0-53.3
END_ZONE = 10.0
GOAL_LINES = (END_ZONE, LENGTH - END_ZONE)  # x = 10 and x = 110

# NFL Rulebook, Rule 1 Sec. 1: inbound lines (hash marks) are 70 ft 9 in from each
# sideline = 23.583 yd. Cross-check: 160 ft - 2 * 70.75 ft = 18.5 ft between hashes,
# which equals the NFL goalpost width, as it should.
HASH_FROM_SIDELINE = 70.75 / 3
# Unsure: BDB does not document which sideline is y = 0 (home vs visitor). It only
# matters for flipping the picture, not for the geometry here.
SIDELINES = (0.0, WIDTH)
HASHES = (HASH_FROM_SIDELINE, WIDTH - HASH_FROM_SIDELINE)

# Rule 1 Sec. 1: yard lines every 5 yd between the goal lines (goal lines included).
YARD_LINES = tuple(float(x) for x in range(10, 111, 5))

ROWS = {"sideline_lo": SIDELINES[0], "hash_lo": HASHES[0], "hash_hi": HASHES[1], "sideline_hi": SIDELINES[1]}


def landmarks():
    """Named (x, y) points where each 5-yd line crosses a sideline or hash line.

    Name format: "x35_hash_lo" = the x=35 line (the left 25-yard line) on the low hash.
    """
    return {f"x{x:g}_{row}": (x, y) for x in YARD_LINES for row, y in ROWS.items()}
