# Vision: Progress Log

This file explains what has been built so far, why, and how it fits together. It is updated at the end of every work session. The chart only shows work that is done or in progress. There are no future plans in it.

```mermaid
gantt
    title Vision: work done so far
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Setup
    Project structure and docs :done, setup, 2026-10-17, 1d
```

---

## 2026-10-17: Project setup

### What was done
- Created the folder structure, a short README in every top-level folder, a root README, a `.gitignore`, `CLAUDE.md`, and this file.
- Ran `git init`, which turned the folder into a git repository. Nothing has been committed yet.
- No code was written on purpose. This step only decides **where things go** and **why**.

### Why it was needed
This project has two very different halves: a Python ML pipeline that runs on your Mac, and a Swift app that runs on your iPhone. They are connected by trained model files. Without a clear layout, data files, model weights, notebooks, and app code get mixed together fast, and it becomes hard to tell what is real code, what is an experiment, and what can be regenerated.

### How it fits the bigger pipeline
The folders follow the path the data takes:

```
data/  →  pipeline/ (train)  →  models/  →  pipeline/export  →  coreml/  →  app/
raw       Python code           .pt         coremltools         .mlpackage   iPhone
```

Inside `pipeline/`, each folder is one stage of the vision → prediction chain:

| Order | Folder | Question it answers |
|---|---|---|
| 1 | `view_classification` | What kind of shot is this? (overhead, broadcast, sideline, phone at a TV…) |
| 2 | `detection` | Where are the players and the ball in this frame? |
| 3 | `tracking` | Which box in this frame is the same player as in the last frame? |
| 4 | `field_mapping` | Where is each player standing on the real field, in yards? |
| 5 | `formation` | What formation is the offense in? |
| 6 | `prediction` | What play is coming, and how sure are we? |
| (7) | `export` | Convert trained models into Core ML for the iPhone |

### Concepts worth understanding

**1. A pipeline is like a signal chain.**
You've seen this in ECE: an antenna feeds a filter, which feeds a demodulator, which feeds a decoder. Each block has one job, a defined input, and a defined output. You can test each block alone. Vision works the same way. The detector doesn't need to know anything about football plays. It just outputs boxes. The predictor doesn't need to know anything about pixels. It just takes player positions. Keeping the stages in separate folders keeps those boundaries clean.

**2. The most important decision: everything becomes field coordinates.**
A broadcast camera and an overhead camera produce very different-looking images. But if the left tackle is standing at the 35-yard line, 2 yards inside the left hash, that fact is the same from every camera.
So stages 1–4 have one job: turn *any* video into a list like this:

```
frame 120:  player 7 → (x = 35.0 yd, y = 21.3 yd)
            player 12 → (x = 33.5 yd, y = 26.6 yd)
            ball      → (x = 35.0 yd, y = 26.6 yd)
```

Stages 5–6 only ever see that list. That has two big benefits:
- The prediction model can be trained on the **NFL Big Data Bowl** dataset, which already gives player positions in exactly this format for thousands of real plays. You don't need to hand-label video to train the predictor.
- Adding a new camera angle only affects the vision half. The predictor doesn't change.

In ECE terms, it's like agreeing on a standard interface, such as a bus protocol, so either side can change without breaking the other.

**3. Homography (a preview, since it shapes the layout).**
A homography is a 3×3 matrix that maps points on one flat plane to another flat plane, seen from a different viewpoint. The field is flat, so if you can find at least 4 points whose real-world positions you know (for example where a yard line crosses a hash mark), you can compute the matrix and then convert *any* pixel on the field into yards. That's why `field_mapping` cares so much about yard lines and hash marks. They are the known reference points.

**4. Why some folders are "git-ignored."**
`.gitignore` tells git which files never to track. We ignore:
- **Datasets and videos**: huge files, and NFL footage is copyrighted. It must never end up in a repo.
- **Model weights and Core ML exports**: large binary files you can regenerate by re-running training or export.
- **Junk**: Python caches, Xcode build output, macOS `.DS_Store` files.
- **`.env`**: where secrets like API keys would go.

Example: you download a 2 GB game file into `data/raw/`. `git status` won't even show it, so you can't commit it by accident.

**5. Why the empty folders contain `.gitkeep`.**
Git tracks files, not folders. An empty folder disappears when someone clones the repo. A `.gitkeep` is an empty placeholder file. Its only job is to make git keep the folder.

**6. Why Core ML?**
Your Mac trains models with PyTorch, but iPhones run models fastest through Core ML, which can use the Neural Engine (Apple's dedicated ML hardware). `coremltools` converts between them. Running on-device means no server, no internet needed, and your video never leaves the phone.
