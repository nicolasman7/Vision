# Round 4: iPhone app shell

**Date:** 2026-10-19. Planned by Coach directly (no Strategist, to save tokens): the scope was clear and the interface between agents was defined up front.

- **Coach:**
  - Set up the dashboard and installed xcodegen; the user accepted the Xcode license.
  - Downloaded the iOS Simulator runtime and git-ignored the generated project.
  - Defined the PlayInputs/PlayPrediction/PlayPredictor interface so the agents could work in parallel.
  - Review: the full build, XCTest on the iPhone 18 Pro Simulator (2/2 pass), and launching the app (no-input prediction 62% pass, same as Python).
- **App Architect (new, UI):** `app/project.yml`, `app/Vision/App/`, `app/README.md` (VS Code SweetPad / Xcode / real-iPhone steps). BUILD SUCCEEDED.
- **Model Bridge (new, ML):** `app/Vision/Prediction/`, `app/VisionTests/`, `scripts/make_app_fixtures.py`. 21 fixtures; macOS max difference 1.1e-16.
- **Game Dev (UI, from the bench):** `app/Vision/UI/` (PredictView, InputPanelView, ResultCardView).
- **Teacher (Docs):** the PROGRESS.md entry and an App section in the Gantt chart.

**Waiting on the user:** approving the commits/push, setting their Apple ID Team in Xcode to run on their phone, and confirming the minimum iOS version.
