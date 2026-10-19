# Round 6: app v3 (self-learning)

**Date:** 2026-10-19. The user asked for the model to learn from its mistakes on unseen plays.

**Coach's design:**
- Keep the tree model frozen; Core ML can't update trees on-device.
- Add a 16-feature online logistic-regression correction layer.
- Use prequential scoring: grade each play before learning from it.
- Add "why" factors, computed by blanking each input group.

**Agents:**
- **Modeler (ML):** a simulation on 69,201 unseen plays.
  - The first 50 or 200 plays: Δ log loss 0.000.
  - All plays: −0.0028 (random order) and −0.0039 (chronological).
  - A higher learning rate hurts short sessions.
  - It shipped lr 0.003, l2 0, plus a 30-step parity fixture.
- **Learner (ML, new):** OnlineLearner and MissExplainer, plus persistence and tests (parity within 1e-9).
- **Historian (Data):** the guess → reveal → learn flow, the Learning card and the Why bars.
- **Teacher (Docs):** CHANGELOG.md, research/model_improvement.md, PROGRESS entries.
- **App Architect:** MARKETING_VERSION 3.0.

**Honest outcome:** learning is safe but barely moves accuracy. The real next lever is retraining with 2024 and getting x/y positions.
