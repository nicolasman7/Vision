# How we tried to make the model better

_Teacher, 2026-10-19. Every number comes from a metrics file in `models/`, named in each section. The app-version history is in [../CHANGELOG.md](../CHANGELOG.md)._

**The short version:** the run/pass model reached about 76% on unseen seasons early on, and most of what we tried since hasn't moved that number. The big wins were in **robustness** (handling missing inputs) and **honest confidence**, not raw accuracy. The ceiling comes from what the inputs say, not from how much data we have or how clever the training is.

| # | Attempt | Result | Kept? |
|---|---|---|---|
| 1 | More seasons (E1) | +0.1 points | Kept (it costs nothing) |
| 2 | FTN hand-charted columns (E3) | +0.1 points, inside the noise | Kept in the bundle, but it adds about nothing |
| 3a | Isotonic calibration | Log loss worse | Dropped |
| 3b | Platt calibration for run/pass | No change in backtest, worse on 2025 | Dropped (ships uncalibrated) |
| 3c | Temperature scaling for direction / depth | Small gain for direction | Kept |
| 4 | Input dropout for missing inputs | Big robustness win, about 0 cost | Kept |
| 5 | Direction / depth bucket studies | Picked D2 and P3. Both heads beat their base rate by about 3–4 points | Kept |
| 6 | On-device online learning | Log loss −0.003 over 69k plays, none in short sessions | Kept (app v3), with a careful learning rate |

## 1. More seasons (E1, `metrics_multiseason.json`)
Same test (2022 weeks 15–22), different training sets:

| Trained on | Training plays | Accuracy | Log loss |
|---|---|---|---|
| 2022 only | 25,762 | 75.23% | 0.491 |
| 2016–2022 | 228,489 | 75.27% | 0.489 |
| 2019–2022 | 128,543 | 75.35% | 0.487 |

Five times the data bought about 0.1 points. When more examples don't help, the model has already learned what the inputs can tell it.

## 2. FTN features (E3, `metrics_multiseason.json`)
FTN's hand-charted pre-snap details (hash, QB spot, backfield count, box) were tested by training on 2022–2024 and testing on 2025. Accuracy went 76.16% → 76.25%, log loss 0.483 → 0.482. Adding motion gave 76.09% and 0.482. These are the same information the formation already carries, said a different way.

## 3. Calibration (`metrics_2022.json`, `metrics_v3.json`)
- **Isotonic, 2022 model:** trained on weeks 1–10 and calibrated on 11–14, it made log loss on weeks 15–22 worse: 0.497 → 0.505. Dropped.
- **Platt, v3 run/pass:** a slope and offset on the log-odds, fit on 2024. In a backtest on 2024 it did essentially nothing (log loss 0.4665 → 0.4663). On 2025 it made things slightly worse (Modeler's report). Run/pass ships **uncalibrated**. It doesn't need the correction: plays it calls at about 75% are right 74.3% of the time.
- **Temperature, v3 heads:** direction got T = 1.24, which softened its overconfident guesses and lowered log loss from 1.076 to 1.073. Depth got T = 1.00, so it was already calibrated.

The lesson: calibrate only when the reliability table shows a real gap. Otherwise you're fitting noise.

## 4. Dropout for missing inputs (`metrics_v3.json`)
During training, inputs were randomly blanked (formation on 15% of rows, the scoreboard on 40%, the whole situation on 25%, all vision inputs on 10%). This lets one model handle any mix of missing inputs:
- **Cost on clean data:** 76.27% → 76.20%, which is nothing.
- **Matches the specialists:** each input pattern scores within about 0.25 points of a model trained only for that pattern. For example, with no situation it gets 74.9% vs 75.0%.
- **The win:** blank the formation on 30% of 2025 plays. The model trained without dropout falls to 72.8%. The dropout model holds at **75.4%** (log loss 0.524 vs 0.493). A wrong formation still hurts both, down to about 71.8%, which is why vision should say "unknown" instead of guessing.

## 5. Direction and depth buckets (`metrics_buckets.json`, `metrics_v3.json`)
Seven direction schemes (one of them diagnostic only) and five depth schemes were compared using log-loss reduction against a base rate:
- **Direction:** the best was D2 (runs only, left/middle/right), which removes only 1.9% (2022) and 1.7% (2025) of the base rate's log loss. Pass direction has almost no signal before the snap.
- **Depth:** P3 (≤0 / 1–9 / 10+ air yards, throws only) removes 5.0% and 5.5%, and is the only option with a clear accuracy gain.
- **The v3 heads on 2025:** direction 40.2% vs a 36.4% base rate, depth 48.6% vs 45.7%.

## 6. Online learning, round 6 (`metrics_online_learning.json`, `scripts/simulate_online_learning.py`)
**The idea:** let the app learn from plays the user reveals. The tree model stays frozen. On top of it sits a tiny logistic regression with 16 inputs: the tree's log-odds, plus down, distance, field position, formation, empty backfield, box, score, and a 4th-quarter flag. It starts as "copy the tree" (weight 1 on the tree's log-odds, 0 on everything else) and takes one SGD step per revealed play.

**Modeler's simulation:**
- **Plays:** the 69,201 plays from 2024–2025, which the trees never trained on.
- **Scoring:** prequential (see PROGRESS.md), so each play is graded before the learner sees its answer.
- **The grid:** 4 learning rates × 4 L2 strengths.
- **Session lengths:** short sessions (the first 50 or 200 plays) were checked over 100 random orders.

| Plays seen | Base log loss | Adjusted (lr 0.003) | Base accuracy | Adjusted accuracy |
|---|---|---|---|---|
| First 50 (avg of 100 orders) | 0.4726 | 0.4726 | 77.36% | 77.32% |
| First 200 (avg of 100 orders) | 0.4722 | 0.4723 | 77.27% | 77.28% |
| All 69,201 (random order) | 0.4707 | 0.4679 | 77.16% | 77.31% |
| All 69,201 (in date order) | 0.4707 | 0.4668 | 77.16% | 77.29% |

**The learning rate that hurt.** With lr = 0.1, a 50-play session got worse on average (log loss +0.016, accuracy 77.4% → 75.7%). In its worst order, log loss rose by +0.052. A large step lets a handful of surprising plays drag the model around. The selection rule was: "log loss may not get worse by more than 0.002 over the first 50 plays; among the rates that pass, take the biggest long-run gain." That picked **lr = 0.003, L2 = 0**.

**Why it barely helps:**
- **The base model has already seen about 273,000 plays.** One fan revealing 50 more can't teach it much it doesn't know.
- **Most misses aren't learnable patterns.** A team that passes on 3rd and 1 isn't making a mistake the model can fix. It's the play-caller staying unpredictable on purpose. No reweighting of the same inputs can see that coming.
- **The small long-run gain is real but tiny** (−0.003 to −0.004 log loss), and slightly bigger in date order. That fits the idea that it mostly tracks slow league-wide drift (for example, teams passing a bit more in some situations) that the frozen trees can't follow.

It's kept because it can't do much harm at this learning rate, and because watching a model learn honestly is educational. It's not a path to a much better model.

## What would actually help next
1. **Retrain the trees with 2024 included.** Right now 2024 (34,788 plays) is held out for calibration that we ended up not using. The newest season is the most relevant one.
2. **x/y positions**, from video (our vision stages) or from Big Data Bowl tracking data. Every attempt above is stuck because the inputs never say *which side* anything is on: where the tight end is, how the receivers split, where the back is offset. That matters most for direction.
3. **Safety alignment:** how many safeties are deep (one-high vs two-high) and where they stand. It's part of the x/y information. It's a defensive tell that today's inputs don't have at all.
