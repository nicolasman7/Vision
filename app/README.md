# app/

The Swift/SwiftUI app: the Xcode project and all Swift source. Targets iOS first, possibly macOS later.

Three ways to give it video: upload a file, record with the camera, or point the camera at a live feed (a TV or a real field). It runs the Core ML models from `coreml/` on the device and draws the results over the video: player positions, a top-down field view, and the predicted play with its confidence.

Swift unit and UI tests live inside the Xcode project here, not in the top-level `tests/` folder.
