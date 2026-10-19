# Round 5: app v2 (redesign, camera, Play lookup)

**Date:** 2026-10-19. Planned by Coach, using the user's design answers: Apple Sports look, field diagram, camera input, favorite accent color.

- **Game Dev (UI):**
  - Redesign: Theme, FieldDiagramView, onboarding color picker, Settings.
  - Fixes after the user tested it:
    - an in-app appearance setting
    - accent color on all text
    - readable labels on accent fills
    - the checkmark root cause (a stale separate @AppStorage in Settings), proven with a UI test
  - Defense inputs.
- **Lens (Vision, new):** Live/Record/Upload camera and PlayerDetector. It uses Core ML directly, because the app module named "Vision" hides Apple's Vision framework.
- **Spotter (Vision):** yolo26n exported to Core ML (4.8 MB, 11/11 persons match PyTorch). The export was first blocked by numpy; Coach pinned ≤2.3.5.
- **App Architect (UI):** tabs, permissions, optional resources, UI-test target.
- **Data Wrangler (Data):** history.sqlite (342,499 plays, 55 MB; `desc` only for 2022+).
- **Historian (Data, new):** Play lookup (HistoryStore over the SQLite3 C API; 8-play parity with Python).
- **Coach:**
  - Reviews: XCTest + UI tests, Simulator screenshots in light/dark/purple.
  - Moved the dashboard to port 8770.
  - Committed and pushed appv2.
