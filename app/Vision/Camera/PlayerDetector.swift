import CoreML
import VideoToolbox

/// One box from the detector. `rect` is normalized (0...1) with a top-left origin,
/// relative to the upright image, so SwiftUI can draw it directly.
struct Detection {
    let rect: CGRect
    let label: String
    let confidence: Float

    var isPerson: Bool { label == "person" }
}

/// YOLO (NMS baked into the export) run straight through Core ML.
///
/// Why not the Vision framework's VNCoreMLRequest: this app's module is also named `Vision`,
/// so `import Vision` resolves to the app itself and Apple's Vision types are unreachable.
/// `MLFeatureValue(cgImage:orientation:constraint:)` does the same rotate-and-scale for us.
/// Returns nil when `yolo_detector.mlmodelc` isn't in the bundle, so the camera still works without it.
final class PlayerDetector {
    /// One model in memory, shared by the camera and the clip player.
    static let shared = PlayerDetector()

    private let model: MLModel
    private let imageConstraint: MLImageConstraint
    private let classes: [String]
    private let lock = NSLock()   // camera and clip queues never overlap in practice; this makes sure

    init?() {
        guard let url = Bundle.main.url(forResource: "yolo_detector", withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url),
              let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        else { return nil }
        self.model = model
        imageConstraint = constraint
        // Ultralytics stores labels as a Python dict string: "{0: 'person', 1: 'bicycle', ...}".
        // Splitting on quotes puts the names at the odd indices.
        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
        let parts = (meta?["names"] ?? "").split(separator: "'", omittingEmptySubsequences: false)
        classes = stride(from: 1, to: parts.count, by: 2).map { String(parts[$0]) }
    }

    /// Runs synchronously on the caller's queue (never call it on the main thread).
    /// `orientation` says how to rotate the buffer so it's upright.
    func detect(_ buffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) -> [Detection] {
        lock.lock(); defer { lock.unlock() }
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &cgImage)
        // cropAndScale 1 = VNImageCropAndScaleOption.scaleFit: letterbox into the square input, like
        // Ultralytics does in training. Stretching instead (scaleFill) found ~20% fewer players.
        guard let cgImage,
              let image = try? MLFeatureValue(cgImage: cgImage, orientation: orientation,
                                              constraint: imageConstraint,
                                              options: [.cropAndScale: NSNumber(value: 1)]),
              let input = try? MLDictionaryFeatureProvider(dictionary: [
                  "image": image,
                  "iouThreshold": 0.45,
                  // A bit below the 0.3 person cut so faint ball detections survive.
                  "confidenceThreshold": 0.2,
              ]),
              let output = try? model.prediction(from: input),
              let scores = output.featureValue(for: "confidence")?.multiArrayValue,
              let boxes = output.featureValue(for: "coordinates")?.multiArrayValue
        else { return [] }

        // Undo the letterbox: the upright frame fills `cw` x `ch` of the square input, centered.
        let sideways = [.left, .right, .leftMirrored, .rightMirrored].contains(orientation)
        let w = CGFloat(sideways ? cgImage.height : cgImage.width)
        let h = CGFloat(sideways ? cgImage.width : cgImage.height)
        let cw = w / max(w, h), ch = h / max(w, h)

        var found: [Detection] = []
        for i in 0..<scores.shape[0].intValue {
            // Best class for this box.
            var best = 0
            var bestScore: Float = 0
            for c in 0..<scores.shape[1].intValue {
                let s = scores[[i, c] as [NSNumber]].floatValue
                if s > bestScore { best = c; bestScore = s }
            }
            let label = best < classes.count ? classes[best] : "\(best)"
            guard (label == "person" && bestScore >= 0.3) || label == "sports ball" else { continue }
            // Boxes are [center x, center y, width, height], normalized to the square input, top-left origin.
            let b = (0..<4).map { CGFloat(boxes[[i, $0] as [NSNumber]].floatValue) }
            let x = (b[0] - (1 - cw) / 2) / cw, y = (b[1] - (1 - ch) / 2) / ch
            let bw = b[2] / cw, bh = b[3] / ch
            found.append(Detection(rect: CGRect(x: x - bw / 2, y: y - bh / 2, width: bw, height: bh),
                                   label: label, confidence: bestScore))
        }
        return found
    }
}
