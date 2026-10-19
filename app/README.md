# app/

The SwiftUI iPhone app (iOS 17+, portrait). It runs the Core ML play predictor on the device.
The Xcode project is generated from `project.yml` by XcodeGen, so only `project.yml` is committed;
`Vision.xcodeproj` and `build/` are git-ignored.

## Prerequisites

- Xcode (with an iOS Simulator runtime if you want to run in the Simulator)
- XcodeGen: `brew install xcodegen`
- The Core ML models. From the repo root run `uv run python -m scripts.export_coreml`. This writes
  `coreml/run_pass.mlmodel`, `direction.mlmodel`, `depth.mlmodel` and `predictor_schema.json`,
  which the app bundles. Project generation fails if they are missing.

## Generate the project

```
cd app && xcodegen generate
```

Run it again whenever you add or remove files or change `project.yml`.

Build from the command line (no signing needed for the Simulator):

```
xcodebuild -project Vision.xcodeproj -scheme Vision -destination 'generic/platform=iOS Simulator' build CODE_SIGNING_ALLOWED=NO
```

## Run it

- **Xcode:** `open Vision.xcodeproj`, pick an iPhone simulator, press Run (Cmd-R). SwiftUI live previews
  (the `#Preview` blocks) only work in Xcode.
- **VS Code:** install the **SweetPad** extension and the **Swift** extension, open the `app/` folder,
  then use SweetPad's "Build & Run" with the `Vision` scheme and a simulator. No previews here.

## Run on a real iPhone

1. `open Vision.xcodeproj`, select the Vision target, open Signing & Capabilities, and pick a Team.
   A free Apple ID works, but apps signed that way expire after 7 days (just run again from Xcode).
2. On the phone: Settings > Privacy & Security > Developer Mode, turn it on, and restart.
3. Plug the phone in, select it as the run destination, and press Run. The first time, trust the
   developer certificate on the phone under Settings > General > VPN & Device Management.

Regenerating the project resets the Team, so set it again after `xcodegen generate`
(or put `DEVELOPMENT_TEAM` in your own local copy of `project.yml` without committing it).
