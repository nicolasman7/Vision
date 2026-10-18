# Where the game situation comes from (down, distance, clock, score)

_Research by Film Scout, 2026-10-17. Options with evidence only. The user decides._

**The problem:** whether a team runs or passes depends heavily on the situation. For example, 3rd-and-9 is usually a pass and 2nd-and-1 often a run. Any play predictor trained on Big Data Bowl or nflverse data will lean on features like `down`, `ydstogo` (yards to go), the clock, and the score. But a camera pointed at the players **can't see any of that**. The numbers live in the TV scorebug (the score box overlaid on broadcasts), in the stadium, or in the viewer's head.

## (a) Read the scorebug on-device (OCR)

**How:** Apple's Vision framework includes a text reader, `VNRecognizeTextRequest` (iOS 13+). A newer Swift version, `RecognizeTextRequest`, needs iOS 18+ ([docs](https://developer.apple.com/documentation/vision/vnrecognizetextrequest), [new API](https://developer.apple.com/documentation/vision/recognizetextrequest)). It runs fully on-device, which fits our constraints, and needs no model training. The settings that matter:
- `recognitionLevel`: `.fast` "returns results more quickly at the expense of accuracy"; `.accurate` "takes more time to produce a more comprehensive result" ([docs](https://developer.apple.com/documentation/vision/vnrequesttextrecognitionlevel)).
- `regionOfInterest`: read only part of the frame, such as the scorebug strip (a normalized rectangle with its origin at the bottom-left).
- `minimumTextHeight`: **the default is 1/32 of the image height**, and smaller text is ignored ([docs](https://developer.apple.com/documentation/vision/vnrecognizetextrequest/minimumtextheight)). Scorebug digits on a full frame are probably smaller than that, so you would either lower this value or crop to the scorebug first. Cropping also makes it faster.
- `customWords`: hints for words the reader should expect, e.g. `["1ST", "2ND", "3RD", "4TH", "&", "GOAL", "FLAG"]`.

**Turning text into numbers:** OCR returns strings like `3RD & 7` or `12:41`. Short patterns (regular expressions) such as `([1-4])(ST|ND|RD|TH) & (\d+|GOAL)` and `(\d{1,2}):(\d{2})` pull out the values. Down and distance only change between plays, so you only need to read them once, just before the snap, and can take a majority vote over a few frames to fix misreads.

**Why it's harder than it sounds:**
- **Every network's scorebug looks different, and they redesign them.** FOX, CBS, NBC, ESPN/ABC, Amazon, NFL Network, and every college network each use a different layout, position, and font. CBS introduced a new scorebug at Super Bowl LV ([Awful Announcing](https://awfulannouncing.com/nfl/cbs-new-scorebug-super-bowl-lv.html)). FOX rebuilt its scorebug for Super Bowl LVII, putting the clock and down-and-distance on top of the team logos ([SI](https://www.si.com/nfl/2023/02/13/fox-updates-score-bug-graphics-for-super-bowl-lvii)). FOX once moved its box down 40 px for late games, which hid the down and distance on RedZone feeds ([Awful Announcing](https://awfulannouncing.com/fox/fox-scorebug-move-nfl-redzone.html)). ESPN's MNF down-and-distance font drew complaints ([Awful Announcing](https://awfulannouncing.com/espn/espn-monday-night-football-scorebug-down-distance-font.html)). Generic OCR can read the text, but **knowing which number is which** (score vs. clock vs. timeouts) needs per-layout rules or a small learned layout parser.
- **The scorebug isn't always there.** It disappears during replays, commercials, and some full-screen graphics.
- **Phone pointed at a TV:** skew, glare, and moiré stripes. The scorebug is small in the phone's frame, so OCR errors go up.
- **All-22, end-zone, and stands video have no scorebug at all.** This is a hard limit: OCR gives nothing for three of our target sources. In the stands, the stadium scoreboard is far away and rarely in frame.

## (b) User input

The user taps in the situation. The minimum UI is one small panel that stays on screen and changes only when needed:

```
Down:     [1] [2] [3] [4]                  ← one tap
To go:    [-] 7 [+]   [Goal]               ← stepper, defaults to 10 after a 1st down
Ball on:  [OWN | OPP]  [-] 35 [+]          ← optional
Quarter:  [1][2][3][4][OT]   Clock: 12:41  ← optional
Score:    Us [-] 14 [+]   Them [-] 10 [+]  ← optional
```

- **Bare minimum:** down + distance. That is **two taps per play** at most, and often one, because distance can default to 10 after a first down.
- **Smart defaults cut the taps further:** after each play the app can guess the next down (down + 1, or reset to 1st & 10). The user only corrects it when the guess is wrong.
- **Works for every source:** All-22, end zone, stands, phone-at-TV. It is 100% reliable when the user is paying attention.
- **Cost:** it needs attention every play. That's fine for film study, but annoying live. Optional fields left blank can go to the model as "unknown", which the model would need to be trained to handle.

## (c) Hybrid

1. **Try OCR first** when a scorebug is found (a broadcast, or a phone at a TV). Show the read values **pre-filled** in the same panel from (b), with the confidence marked (e.g. greyed out if uncertain). The user confirms by doing nothing, or fixes it with one tap.
2. **Fall back to manual entry** when there is no scorebug (All-22, end zone, stands) or when OCR confidence is low.
3. **Carry state forward:** use the last known value plus the "next down" guess, so even manual mode is about one tap per play.
4. **Optional, for past NFL games only:** a nflverse play-by-play file has down, distance, clock, and score for every play. It downloads with no login ([play_by_play_2024.parquet](https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet), verified HTTP 200). If the user picks the game, the app could look plays up from a bundled copy, which keeps everything on-device. This doesn't work for live games, college, or high school.

**Cost:** the most code of the three (OCR, per-network layout rules, plus the UI). But the UI from (b) is needed anyway, so OCR becomes an add-on, not a requirement.

## Model impact (filled in from Modeler's experiment E4)

_Filled in by Teacher from `models/metrics_multiseason.json` (Modeler's experiments E4 and E5)._

**The question:** if the app knew nothing about the situation, how much worse would the run/pass guess be?

**How E4 tested it:** train on 2016–2024 (308,086 plays) and test on all of 2025 (34,413 plays). Start with a model that sees only the **alignment**: what a camera could see before the snap (formation family, empty backfield, personnel, box count, players on the field, shotgun, no-huddle, plus the FTN hand-charted hash, QB spot and backfield count). Then add the situation back one piece at a time. Each row includes everything above it.

| Model sees | Accuracy | Log loss (lower is better) |
|---|---|---|
| Nothing (always guess "pass") | 60.4% | 0.672 |
| Situation only, no alignment (for comparison) | 71.1% | 0.529 |
| **Alignment only, no situation** | **75.0%** | **0.527** |
| + down and distance | 75.7% | 0.503 |
| + yard line and goal-to-go | 75.8% | 0.498 |
| + score, quarter, clocks | 76.3% | 0.481 |
| **+ timeouts (full model)** | **76.2%** | **0.480** |

**In plain English:**
- **Without any situation, the model loses only about 1.2 points of accuracy** (76.2% → 75.0%). On its own, the alignment is a better guide than the scoreboard (75.0% vs 71.1%).
- **The situation mostly makes the confidence sharper, not the guesses better.** Log loss goes from 0.527 to 0.480. The yes/no calls hardly change, but the model becomes surer when it should be. Example: a shotgun formation already leans "pass". Knowing it's 3rd and 12 pushes the probability much closer to certain, while 2nd and 1 pulls it back toward a coin flip.
- **Down and distance are worth the most** (0.527 → 0.503). Score and clock come next (0.498 → 0.481). **Timeouts add nothing.**

**The catch: the alignment numbers above use perfect, human-charted formations.** The app will read formations from video, and it will get some wrong. Experiment E5 tested a different model (trained 2019–2022, tested on 2022 weeks 15–22) with the formation input blanked out:

| E5 model (2022 test) | Accuracy | Log loss |
|---|---|---|
| Formation known | 75.3% | 0.487 |
| **Formation unknown** (vision can't read it) | **70.5%** | **0.531** |

When the formation can't be read, that model drops about 5 points. It's then leaning on the situation and everything else it can see. So the situation is cheap insurance for the plays where vision struggles: All-22 piles, a phone at a bad angle, a cut-off frame.

This is evidence for the decision, not the decision itself. Film Scout's recommendation below is unchanged.

## Recommendation (you can overrule this)

**Build (b) user input first, designed so (c) can be added later.** Reasons:
- It is the only option that works for **every** input source. Three of the target sources have no scorebug, so OCR can never be the only path.
- It is small: one panel and two taps per play, and it needs no network-specific rules.
- Add scorebug OCR (`VNRecognizeTextRequest` on a cropped region) as a **pre-fill** for broadcast footage once the rest of the pipeline works, starting with whichever one or two networks you actually watch.
- Revisit this after E4. If the model barely uses the situation, even the input panel could be optional.

## Could not verify
- Real OCR accuracy on scorebugs, and speed on iPhone. I found no published study; it would need a quick on-device test with your own frames.
- Exact scorebug text sizes per network (the 1/32 `minimumTextHeight` concern is an estimate).
