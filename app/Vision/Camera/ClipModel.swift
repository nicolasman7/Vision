import AVFoundation
import CoreTransferable
import UniformTypeIdentifiers

/// A video picked from Photos, copied to a temp file we can keep playing.
struct Movie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent(received.file.lastPathComponent)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return Movie(url: copy)
        }
    }
}

/// Plays an uploaded clip and runs the detector on its frames at about 5 fps.
final class ClipModel: ObservableObject {
    @Published var player: AVPlayer?
    @Published var detections: [Detection] = []
    @Published var imageSize = CGSize(width: 16, height: 9)   // upright display size
    @Published var fps = 0.0

    private var output: AVPlayerItemVideoOutput?
    private var orientation = CGImagePropertyOrientation.up
    private var timer: Timer?
    private var busy = false          // main thread only
    private var lastRun = 0.0
    private let queue = DispatchQueue(label: "clip.detect")

    @MainActor func load(_ url: URL) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let (natural, transform) = try? await track.load(.naturalSize, .preferredTransform) else { return }
        let upright = natural.applying(transform)
        imageSize = CGSize(width: abs(upright.width), height: abs(upright.height))
        orientation = Self.orientation(of: transform)

        let item = AVPlayerItem(asset: asset)
        let out = AVPlayerItemVideoOutput(pixelBufferAttributes:
            [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        item.add(out)
        output = out
        detections = []
        player?.pause()
        player = AVPlayer(playerItem: item)
        player?.play()

        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in self?.tick() }
    }

    private func tick() {
        guard !busy, let output, let detector = PlayerDetector.shared else { return }
        let time = output.itemTime(forHostTime: CACurrentMediaTime())
        // No new frame while paused, so the last boxes stay on screen.
        guard output.hasNewPixelBuffer(forItemTime: time),
              let buffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else { return }
        busy = true
        let orientation = self.orientation
        queue.async {
            let found = detector.detect(buffer, orientation: orientation)
            DispatchQueue.main.async {
                let now = CACurrentMediaTime(), dt = now - self.lastRun
                self.lastRun = now
                if dt < 2 { self.fps = self.fps * 0.8 + 0.2 / dt }
                self.detections = found
                self.busy = false
            }
        }
    }

    func pause() { player?.pause() }

    /// Phone videos are stored sideways with a rotation in preferredTransform; Vision needs that as an orientation.
    static func orientation(of t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch Int((atan2(t.b, t.a) * 180 / .pi).rounded()) {
        case 90: return .right
        case -90: return .left
        case 180, -180: return .down
        default: return .up
        }
    }
}
