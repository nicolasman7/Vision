import AVFoundation
import AVKit
import PhotosUI
import SwiftUI

/// The Live tab: full-bleed camera with player boxes, recording, or an uploaded clip.
struct LiveView: View {
    enum Mode: String, CaseIterable { case live = "Live", record = "Record", upload = "Upload" }

    @StateObject private var camera = CameraModel()
    @StateObject private var clip = ClipModel()
    @State private var mode = Mode.live
    @State private var pickedItem: PhotosPickerItem?
    @State private var loadingClip = false
    @State private var snapshot: Snapshot?

    var body: some View {
        ZStack {
            // Only camera/video pixels are dark; everything else follows the app appearance.
            Color(.systemBackground).ignoresSafeArea()
            if mode == .upload { uploadContent } else { cameraContent }
        }
        .overlay(alignment: .top) { pills.padding(.top, 8) }
        .overlay(alignment: .bottom) { controls.padding(.horizontal, 16).padding(.bottom, 12) }
        .onAppear { if mode != .upload { Task { await camera.start() } } }
        .onDisappear { camera.stop(); clip.pause() }
        .onChange(of: mode) { old, new in
            snapshot = nil
            if new == .upload { camera.stop() } else { clip.pause() }
            Task {
                if new != .upload { await camera.start() }
                if new == .record || old == .record { await camera.setRecordMode(new == .record) }
            }
        }
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            loadingClip = true
            Task {
                if let movie = try? await item.loadTransferable(type: Movie.self) { await clip.load(movie.url) }
                loadingClip = false
            }
        }
    }

    // MARK: Camera (Live and Record)

    @ViewBuilder private var cameraContent: some View {
        switch camera.status {
        case .starting:
            ProgressView()
        case .noCamera:
            MessageCard(icon: "video.slash", title: "Camera not available on Simulator. Use Upload.",
                        message: "Pick a clip from your library to see player boxes.")
        case .denied:
            MessageCard(icon: "camera", title: "Camera access is off",
                        message: "Allow camera access in Settings to film the play.", showsSettings: true)
        case .ready:
            if let snapshot {
                Image(decorative: snapshot.image, scale: 1)
                    .resizable().scaledToFill().ignoresSafeArea()
                DetectionOverlay(detections: snapshot.detections,
                                 imageSize: CGSize(width: snapshot.image.width, height: snapshot.image.height),
                                 fill: true).ignoresSafeArea()
            } else {
                CameraPreview(session: camera.session).ignoresSafeArea()
                if mode == .live {
                    DetectionOverlay(detections: camera.detections, imageSize: camera.imageSize, fill: true)
                        .ignoresSafeArea()
                }
            }
            if mode == .record && camera.photosDenied {
                MessageCard(icon: "photo.on.rectangle", title: "Photos access is off",
                            message: "Recordings can't be saved until Vision may add to your library.",
                            showsSettings: true)
            }
        }
    }

    // MARK: Upload

    @ViewBuilder private var uploadContent: some View {
        if let player = clip.player {
            VideoPlayer(player: player) {
                DetectionOverlay(detections: clip.detections, imageSize: clip.imageSize, fill: false)
            }
            .padding(.bottom, 130)   // keep the player's controls clear of ours
        } else if loadingClip {
            ProgressView("Loading clip")
        } else {
            MessageCard(icon: "film", title: "Analyze a clip",
                        message: "Choose a video from your library. Players are boxed as it plays.") {
                PhotosPicker(selection: $pickedItem, matching: .videos) { Text("Choose Video").onAccentLabel() }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    // MARK: Pills and controls

    private var detectionsShown: [Detection] {
        snapshot?.detections ?? (mode == .upload ? clip.detections : camera.detections)
    }

    @ViewBuilder private var pills: some View {
        HStack(spacing: 8) {
            if PlayerDetector.shared == nil {
                Pill(text: "Detector not installed", icon: "exclamationmark.triangle")
            } else if mode == .record {
                if let start = camera.recordingStart {
                    Pill(icon: "record.circle", tint: .red) { Text(timerInterval: start...Date.distantFuture, countsDown: false) }
                }
            } else if camera.status == .ready || mode == .upload {
                let players = detectionsShown.filter(\.isPerson).count
                Pill(text: "\(players) player\(players == 1 ? "" : "s")", icon: "person.2")
                if snapshot == nil {
                    Pill(text: "\(Int((mode == .upload ? clip.fps : camera.fps).rounded())) fps", icon: "speedometer")
                }
            }
            if let notice = camera.notice { Pill(text: notice, icon: "checkmark.circle") }
        }
        .animation(.default, value: camera.notice)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            if mode == .record && camera.status == .ready {
                RecordButton(isRecording: camera.recordingStart != nil) { camera.toggleRecording() }
            }
            if mode == .live && camera.status == .ready && PlayerDetector.shared != nil {
                Button(snapshot == nil ? "Snapshot" : "Back to Live",
                       systemImage: snapshot == nil ? "camera.viewfinder" : "play.fill") {
                    if snapshot == nil { camera.snapshot { snapshot = $0 } } else { snapshot = nil }
                }
                .buttonStyle(.bordered).background(.regularMaterial, in: Capsule())
            }
            if mode == .upload && clip.player != nil {
                PhotosPicker("Choose Another", selection: $pickedItem, matching: .videos)
                    .buttonStyle(.bordered).background(.regularMaterial, in: Capsule())
            }
            if mode != .record {
                Text("Boxes only for now. Reading the formation from video comes in a later round.")
                    .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(.regularMaterial, in: Capsule())
            }
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue) }
            }
            .pickerStyle(.segmented)
            .disabled(camera.recordingStart != nil)   // finish the recording before switching
            .padding(4)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }
}

// MARK: - Pieces

/// AVCaptureVideoPreviewLayer as a SwiftUI view.
private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}

/// Draws normalized boxes over an image shown aspect-fill (camera) or aspect-fit (video player).
struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize
    let fill: Bool

    var body: some View {
        GeometryReader { geo in
            // Where the image actually lands inside this view.
            let sx = geo.size.width / imageSize.width, sy = geo.size.height / imageSize.height
            let scale = fill ? max(sx, sy) : min(sx, sy)
            let w = imageSize.width * scale, h = imageSize.height * scale
            let ox = (geo.size.width - w) / 2, oy = (geo.size.height - h) / 2
            ForEach(detections.indices, id: \.self) { i in
                let r = detections[i].rect
                RoundedRectangle(cornerRadius: 6)
                    .stroke(detections[i].isPerson ? AnyShapeStyle(.tint) : AnyShapeStyle(.yellow), lineWidth: 2)
                    .frame(width: r.width * w, height: r.height * h)
                    .position(x: ox + r.midX * w, y: oy + r.midY * h)
            }
        }
        .allowsHitTesting(false)
    }
}

private struct Pill<Content: View>: View {
    var icon: String
    var tint: Color?   // nil: inherit the accent
    @ViewBuilder var content: Content

    var body: some View {
        Label { content } icon: { Image(systemName: icon).foregroundStyle(tint.map(AnyShapeStyle.init) ?? AnyShapeStyle(.tint)) }
            .font(.footnote.weight(.semibold).monospacedDigit())
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())   // keeps accent text readable over video
    }
}

extension Pill where Content == Text {
    init(text: String, icon: String) { self.init(icon: icon) { Text(text) } }
}

private struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().stroke(.tint, lineWidth: 4).frame(width: 72, height: 72)
                RoundedRectangle(cornerRadius: isRecording ? 6 : 30)
                    .fill(.red)
                    .frame(width: isRecording ? 30 : 60, height: isRecording ? 30 : 60)
            }
            .animation(.spring(duration: 0.25), value: isRecording)
        }
        .accessibilityLabel(isRecording ? "Stop recording" : "Start recording")
    }
}

/// Friendly centered card for empty and permission states.
private struct MessageCard<Actions: View>: View {
    let icon: String
    let title: String
    let message: String
    var showsSettings = false
    @ViewBuilder var actions: Actions

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon).font(.largeTitle).foregroundStyle(.tint)
            Text(title).font(.headline).multilineTextAlignment(.center)
            Text(message).font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
            if showsSettings, let url = URL(string: UIApplication.openSettingsURLString) {
                Link(destination: url) { Text("Open Settings").onAccentLabel() }.buttonStyle(.borderedProminent)
            }
            actions
        }
        .padding(24)
        .frame(maxWidth: 320)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

extension MessageCard where Actions == EmptyView {
    init(icon: String, title: String, message: String, showsSettings: Bool = false) {
        self.init(icon: icon, title: title, message: message, showsSettings: showsSettings) { EmptyView() }
    }
}

#Preview {
    LiveView()
}
