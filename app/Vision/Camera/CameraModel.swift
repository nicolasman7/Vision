import AVFoundation
import CoreImage
import Photos

/// A frozen frame plus the boxes found on it.
struct Snapshot {
    let image: CGImage
    let detections: [Detection]
}

/// Owns the capture session: back wide camera at 1080p, live detection, and recording to Photos.
/// All @Published values change on the main thread; the session runs on its own queue.
final class CameraModel: NSObject, ObservableObject {
    enum Status { case starting, ready, noCamera, denied }

    @Published var status = Status.starting
    @Published var detections: [Detection] = []
    @Published var imageSize = CGSize(width: 1080, height: 1920)   // upright (portrait) frame size
    @Published var fps = 0.0
    @Published var recordingStart: Date?
    @Published var photosDenied = false
    @Published var notice: String?          // short toast, e.g. "Saved to Photos"

    let session = AVCaptureSession()
    let detector = PlayerDetector.shared

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoQueue = DispatchQueue(label: "camera.video")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let movieOutput = AVCaptureMovieFileOutput()
    private var audioInput: AVCaptureDeviceInput?
    private var configured = false

    // Touched only on videoQueue.
    private var lastRun = 0.0
    private var lastBuffer: CVPixelBuffer?
    private var lastDetections: [Detection] = []

    /// Asks for camera access if needed, then starts the session. Safe to call repeatedly.
    @MainActor func start() async {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            status = .noCamera   // the Simulator has no camera
            return
        }
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
            status = .denied
            return
        }
        sessionQueue.async {
            if !self.configured { self.configure(device) }
            if !self.session.isRunning { self.session.startRunning() }
        }
        status = .ready
    }

    func stop() {
        sessionQueue.async { self.session.stopRunning() }
    }

    private func configure(_ device: AVCaptureDevice) {
        session.beginConfiguration()
        if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
        if let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
            session.addInput(input)
        }
        videoOutput.alwaysDiscardsLateVideoFrames = true   // drop frames while the detector is busy
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        session.commitConfiguration()
        configured = true
    }

    /// Record mode swaps the detection output for the movie output. Older iPhones can't run
    /// both at once, and boxes aren't needed while recording.
    @MainActor func setRecordMode(_ on: Bool) async {
        var micAllowed = false
        if on {
            let photos = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            photosDenied = !(photos == .authorized || photos == .limited)
            micAllowed = await AVCaptureDevice.requestAccess(for: .audio)   // no mic: record silent video
        }
        sessionQueue.async {
            guard self.configured else { return }
            self.session.beginConfiguration()
            if on {
                self.session.removeOutput(self.videoOutput)
                if self.session.canAddOutput(self.movieOutput) { self.session.addOutput(self.movieOutput) }
                if let conn = self.movieOutput.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90   // save portrait video, matching the preview
                }
                if micAllowed, self.audioInput == nil,
                   let mic = AVCaptureDevice.default(for: .audio),
                   let input = try? AVCaptureDeviceInput(device: mic), self.session.canAddInput(input) {
                    self.session.addInput(input)
                    self.audioInput = input
                }
            } else {
                self.session.removeOutput(self.movieOutput)
                if let input = self.audioInput { self.session.removeInput(input); self.audioInput = nil }
                if self.session.canAddOutput(self.videoOutput) { self.session.addOutput(self.videoOutput) }
            }
            self.session.commitConfiguration()
        }
        if !on { detections = [] }
    }

    @MainActor func toggleRecording() {
        if recordingStart != nil {
            sessionQueue.async { self.movieOutput.stopRecording() }
            recordingStart = nil
            return
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
        sessionQueue.async { self.movieOutput.startRecording(to: url, recordingDelegate: self) }
        recordingStart = .now
    }

    /// Freezes the latest analysed frame and its boxes.
    func snapshot(_ done: @escaping (Snapshot) -> Void) {
        videoQueue.async {
            guard let buffer = self.lastBuffer else { return }
            let image = CIImage(cvPixelBuffer: buffer).oriented(.right)
            guard let cg = CIContext().createCGImage(image, from: image.extent) else { return }
            let found = self.lastDetections
            DispatchQueue.main.async { done(Snapshot(image: cg, detections: found)) }
        }
    }

    private func show(_ message: String) {
        DispatchQueue.main.async {
            self.notice = message
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                if self.notice == message { self.notice = nil }
            }
        }
    }
}

extension CameraModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Throttle to ~12 fps; the camera delivers 30. Detection runs right here on videoQueue,
        // and alwaysDiscardsLateVideoFrames drops whatever arrives while it's busy.
        let now = CACurrentMediaTime()
        let dt = now - lastRun
        guard let detector, dt >= 1.0 / 12, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastRun = now
        // Buffers arrive in sensor (landscape) orientation; .right makes Vision see the upright portrait frame.
        let found = detector.detect(buffer, orientation: .right)
        lastBuffer = buffer
        lastDetections = found
        let size = CGSize(width: CVPixelBufferGetHeight(buffer), height: CVPixelBufferGetWidth(buffer))
        DispatchQueue.main.async {
            self.detections = found
            self.imageSize = size
            if dt < 2 { self.fps = self.fps * 0.8 + 0.2 / dt }   // smoothed; skip the first gap
        }
    }
}

extension CameraModel: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        // A recording can end with an error yet still be a complete file (e.g. disk nearly full).
        let finished = (error as NSError?)?.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool ?? (error == nil)
        guard finished else {
            try? FileManager.default.removeItem(at: url)
            show("Recording failed")
            return
        }
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }) { saved, _ in
            try? FileManager.default.removeItem(at: url)
            self.show(saved ? "Saved to Photos" : "Couldn't save to Photos")
        }
    }
}
