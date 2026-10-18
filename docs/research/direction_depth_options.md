# Direction and depth: which buckets should the predictor use?

_Bucket Scout, 2026-10-17. Numbers come from `scripts/analyze_buckets.py` and are saved in `models/metrics_buckets.json`. This page gives evidence and a recommendation; **you** pick the targets._

## The setup (same for every option)

- **Data:** the 10-season table (`plays_2016_2025.parquet`, 342,499 plays).
- **Test:** 2022 weeks 15–22. The model trains on every earlier season plus 2022 weeks 1–14, so it never sees the future. A second test runs the same way on 2025 weeks 15–22.
- **Model:** one `HistGradientBoostingClassifier` with the pre-snap inputs from `model.py`, plus `formation_family`, `is_empty`, and the FTN pre-snap columns (hash, QB location, backfield count, box, motion) where they exist (2022+).
- **Baseline:** guess using the class frequencies from training. For example, if 39% of training runs went left, the baseline always says "left, 39%". For options that mix runs and passes, the baseline also knows run vs pass, because the model does too.
- **Skill:** how much the model beats the baseline. It's reported two ways:
  - **Accuracy points:** the model's accuracy minus the baseline's.
  - **Log-loss reduction:** what percentage of the baseline's log loss the model removes. This is the fairer way to compare a 2-class option with a 7-class one.

**Read log loss, not accuracy.** In 2025 weeks 15–22, runs swung to the right (53% of the left/right runs). The baseline's "always left" lost ground, and the model gained accuracy simply by guessing "right" more often. That's luck in the test weeks, not skill. Log loss scores the probabilities, so a shift like that barely moves it. Every log-loss gap also has a standard error in the JSON, about 0.0015–0.0045. Treat any gap under about 2 SE as noise.

## Decision 1: direction

Balance is the test-weeks class mix for 2022. Skill is shown as 2022 / 2025.

| Option | Classes (balance) | Baseline acc | Model acc | Acc skill | Log-loss reduction | Pros / cons |
|---|---|---|---|---|---|---|
| **D1** L/M/R, all plays (today) | R 39 / L 38 / M 23 | 40.2% | 41.1% | +0.9 / +3.2 | 1.0% / 1.0% | + one label for every play. − Mixes runs (some signal) with passes (none). |
| _D1p_ passes only (diagnostic) | R 41 / L 38 / M 21 | 41.2% | 40.6% | −0.6 / +1.2 | 0.3% / 0.7% | Where the ball goes depends on the QB's read **after** the snap. Basically unpredictable from today's inputs. |
| **D2** runs only, L/M/R | L 39 / R 36 / M 25 | 38.9% | 41.2% | +2.2 / +5.3 | **1.9% / 1.7%** | + Best signal of any direction option. + Runs are designed to a side before the snap. − Says nothing about passes. |
| **D3** runs only, L/R | L 52 / R 48 | 52.0% | 53.5% | +1.5 / +6.3 | 0.1% / 1.0% | − Drops the 25% of runs that go up the middle, so the app could never say "middle". − In 2022 it's a coin flip. |
| **D4** runs by gap (7 classes) | middle 25, six gaps 11–14% each | 25.2% | 25.1% | −0.1 / −0.3 | 1.3% / 1.2% | + Most detailed football meaning. − No accuracy skill at all. Too fine for today's inputs, and harder to read from video. |
| **D5** field-relative (hash side / middle / wide side) | hash 39 / wide 38 / M 23 | 38.9% | 40.1% | +1.1 / +3.0 | 0.4% / 1.2% | + Natural for football ("boundary vs field"). − 2022+ only, and middle-hash snaps (10%) get no label. − Frame unknown (see below). |
| **D6** D1 + team tendency | same as D1 | 40.2% | 40.8% | +0.6 / +3.7 | 1.4% / 1.3% | + Small but real gain over D1 (+0.4 points of reduction). − The app would need to know which team has the ball. |

**About D5.** FTN doesn't document whether `starting_hash` "L" is the offense's left or the left side seen from the press box. Reading A (used here) assumes the same frame as nflverse's left/right. Under reading B, the "hash side" and "wide side" labels simply swap. Every metric is identical, so both readings are covered. Field-relative labels are a relabeling of left/right using the hash, and the model already sees the hash. So D5 can't create new signal; it only changes what the baseline knows. Note also that D5's 2022 test trains on only 21,537 plays (FTN starts in 2022).

**About D6, leakage.** A team's tendency for week *w* is computed only from its games in weeks before *w*, within the same season. `tests/test_buckets.py` checks this: it rewrites a week's outcomes and asserts that week's own features don't change. The test weeks have 13+ earlier weeks of history.

**Bottom line for direction:** no option has meaningful skill with today's inputs. The best, D2, removes about 2% of the baseline's log loss. Even in the best case, today's inputs barely help predict direction.

## Decision 2: depth (passes only)

| Option | Classes (2022 test balance) | Baseline acc | Model acc | Acc skill | Log-loss reduction | Pros / cons |
|---|---|---|---|---|---|---|
| **P1** nflverse short/deep (≈15 yd) | short 82 / deep 18 | 82.4% | 82.5% | +0.1 / +0.1 | 5.4% / 6.7% | + Simple. − As a hard call it never beats "always short". − nflverse's label comes from the play text, not air_yards: it disagrees with `air_yards ≥ 15` on 2.2% of throws ("short" passes up to 44 yd). |
| **P2** ≤0 / 1–9 / 10–19 / 20+ | short 47 / behind 21 / interm. 20 / deep 12 | 46.6% | 47.0% | +0.3 / +0.9 | 4.7% / 5.7% | + Most detail. − The 20+ class is 12% and the model almost never calls it. |
| **P3** ≤0 / 1–9 / 10+ | short 47 / downfield 32 / behind 21 | 46.6% | 48.8% | **+2.2 / +3.6** | 5.0% / 5.5% | + Best accuracy skill, reduction on a par with P1 and P2, and no tiny classes. + "Behind the line" (screens, swings) is a distinct play type. − Merges intermediate and deep. |
| _P2 + "no throw" class_ | adds no_throw 11 | 41.5% | 42.4% | +0.9 / +1.0 | 4.0% / 4.6% | − The extra class adds noise; skill drops. |
| _No throw? (yes/no)_ | throw 89 / no 11 | 89.0% | 89.0% | 0 / 0 | 0.5% / 1.0% | Sacks and scrambles are nearly unpredictable before the snap. |

**Why P3.** The air-yards distribution has three natural chunks of similar size: ≤0 (21%), 1–9 (46%) and 10+ (33%). P2's last two classes are thin (20% and 12%). Class balance is fairly stable across all 10 seasons, within about 5 points per class. The biggest drift is "behind the line", which crept up from 18% (2016) to about 21–23% (2021–2025).

**Sacks and scrambles** (no air_yards, 11–12% of dropbacks). Three choices:
1. **Drop them and predict "intended depth if the ball is thrown."** Recommended. Whether a dropback ends in a sack is almost unpredictable pre-snap (≤1% log-loss reduction), and adding them as a class makes the depth model worse.
2. Give them their own class: tested above, and it's worse.
3. Predict them separately: no skill, so the app can show the plain base rate instead ("about 1 in 8 dropbacks ends without a throw"). That rate rose from 8.9% (2016) to 12.0% (2025).

## What Big Data Bowl x/y positions would likely add (can't be measured today)

Today's inputs say "SHOTGUN, 1 RB, 1 TE", but never **which side** anything is on. Direction is mostly a "which side" question, which is why every direction option is stuck near the baseline. Pre-snap (x, y) positions would give:
- **Formation strength:** which side the tight end is on, and how many receivers line up on each side ("trips right"). Offenses often run toward the strength.
- **Running back offset:** in shotgun, a back lined up to one side of the QB usually takes a handoff going the other way.
- **Motion direction**, not just "was there motion".
- **Defensive shade:** where the box defenders and safeties are, and which side the defense rolled toward.
- **Depth hints:** receiver splits and stacks/bunches (tight splits suggest crossers; a back already split out wide suggests a screen).
- **D5 without ambiguity:** BDB includes the play direction and absolute y, so "toward the boundary / toward the field" can be computed exactly, and "toward the strong side / weak side" becomes possible too.

None of this can be tested until BDB is downloaded (it needs your Kaggle account). It's also exactly what the vision stages will output.

## Recommendation (accept or overrule)

1. **Direction: use D2, runs only, left/middle/right, from the offense's view.** It's the only option with consistent (small) skill, it keeps "middle", and it's the label that formation strength should improve most once x/y positions exist. **Don't make pass direction a headline output for now** (D1p has close to zero skill). Re-run this comparison, including a strong-side/weak-side version of D5, after BDB features exist. Leave D4 (gaps) and D6 (team tendency) for later: D4 is too fine, and D6 needs the app to know the team.
2. **Depth: use P3, behind the line (≤0) / short (1–9) / downfield (10+), computed from `air_yards`, passes only.** Train on throws only, and predict "intended depth if thrown." Show the sack/scramble rate as a fixed note, not a model output.

## Surprises

- Today's direction model isn't broken. Direction (especially for passes) just isn't in today's inputs. Ten seasons of data plus the FTN columns barely moved it (D1 40.1% → 41.1%).
- nflverse's `pass_length` isn't strictly 15 air yards ("deep" passes as short as 7 yd). If short/deep is wanted, compute it from `air_yards`.
- Middle runs fell from about 28–30% (2016–2018) to about 25–27%. Sacks plus scrambles rose from 8.9% to 12.0%.
- The 2025 test weeks inflate accuracy skill for every run-direction option, because the late-season left/right mix flipped. Use log loss.
