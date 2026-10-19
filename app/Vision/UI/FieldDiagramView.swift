import SwiftUI

/// Hero card: the run/pass headline plus a top-down field slice with the prediction drawn on it.
/// Pass depth = shaded bands past the line of scrimmage; run direction = arrows out of the backfield.
/// `prediction == nil` draws an empty field (e.g. while the camera has no read yet).
struct FieldDiagramView: View {
    let prediction: PlayPrediction?
    var ydstogo: Int? = nil                     // draws the yellow first-down line when known
    @AppStorage("accentColor") private var accent: AccentChoice = .blue
    @ScaledMetric(relativeTo: .largeTitle) private var heroSize: CGFloat = 60

    var body: some View {
        let probs = Probs(prediction)
        VStack(alignment: .leading, spacing: 14) {
            headline
            FieldCanvas(probs: probs, accent: accent.color, ink: accent.textColor, ydstogo: ydstogo, hasPrediction: prediction != nil)
                .aspectRatio(0.95, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .accessibilityElement()
                .accessibilityLabel(accessibilitySummary)
        }
        .animation(.spring(duration: 0.6, bounce: 0.15), value: probs)
    }

    private var headline: some View {
        let pPass = prediction?.pPass ?? 0.5
        let top = max(pPass, 1 - pPass)
        return HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(prediction == nil ? "–" : "\(Int((top * 100).rounded()))%")
                .font(.system(size: heroSize, weight: .heavy, design: .rounded))
                .monospacedDigit()
                .contentTransition(.numericText(value: top))
                .foregroundStyle(accent.textColor)
            VStack(alignment: .leading, spacing: 0) {
                Text(prediction == nil ? "No read yet" : pPass >= 0.5 ? "PASS" : "RUN")
                    .font(.title2.weight(.heavy))
                Text(prediction == nil ? "Set the situation or point the camera" : Self.strength(top))
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
            }
            .fontDesign(.rounded)
        }
        .accessibilityElement(children: .combine)
    }

    /// "Lean" under 60%, "Likely" 60–75%, "Strong" above 75%.
    static func strength(_ top: Double) -> String { top > 0.75 ? "Strong" : top >= 0.6 ? "Likely" : "Lean" }

    private var accessibilitySummary: String {
        guard let p = prediction else { return "Field diagram, no prediction yet" }
        func pct(_ v: Double?) -> String { "\(Int(((v ?? 0) * 100).rounded())) percent" }
        return "If pass: behind the line \(pct(p.depth["<=0"])), short \(pct(p.depth["1-9"])), deep \(pct(p.depth["10+"])). "
            + "If run: left \(pct(p.direction["left"])), middle \(pct(p.direction["middle"])), right \(pct(p.direction["right"]))."
    }
}

/// The seven numbers we animate between: pPass, run left/middle/right, depth behind/short/deep.
struct Probs: VectorArithmetic {
    var v = SIMD8<Double>()

    init(_ p: PlayPrediction?) {
        guard let p else { return }
        v = [p.pPass, p.direction["left"] ?? 0, p.direction["middle"] ?? 0, p.direction["right"] ?? 0,
             p.depth["<=0"] ?? 0, p.depth["1-9"] ?? 0, p.depth["10+"] ?? 0, 0]
    }
    private init(v: SIMD8<Double>) { self.v = v }

    static var zero: Probs { Probs(v: .zero) }
    static func + (a: Probs, b: Probs) -> Probs { Probs(v: a.v + b.v) }
    static func - (a: Probs, b: Probs) -> Probs { Probs(v: a.v - b.v) }
    mutating func scale(by rhs: Double) { v *= rhs }
    var magnitudeSquared: Double { (v * v).sum() }
}

private struct FieldCanvas: View, Animatable {
    var probs: Probs
    let accent: Color    // fills
    let ink: Color       // text
    let ydstogo: Int?
    let hasPrediction: Bool

    var animatableData: Probs {
        get { probs }
        set { probs = newValue }
    }

    var body: some View {
        Canvas { ctx, size in
            let w = size.width, h = size.height
            let yd = h / 30                       // 30 yards visible: 9 behind the LOS, 21 beyond
            let los = h - 9 * yd
            let y = { (yards: Double) in los - yards * yd }   // yards past the LOS -> canvas y
            let line = Color.primary.opacity(0.14)

            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.primary.opacity(0.05)))
            for mark in stride(from: -5.0, through: 20, by: 5) {
                ctx.fill(Path(CGRect(x: 0, y: y(mark) - 0.5, width: w, height: 1)), with: .color(line))
            }
            for mark in -9...21 where mark % 5 != 0 {          // hash ticks
                for x in [w * 0.38, w * 0.62] {
                    ctx.fill(Path(CGRect(x: x - 4, y: y(Double(mark)) - 0.5, width: 8, height: 1)), with: .color(line))
                }
            }

            let pPass = probs.v[0]
            // Smooth emphasis: bands light up as pass gets likelier, arrows as run does.
            let passEmph = 0.3 + 0.7 * min(max((pPass - 0.4) / 0.2, 0), 1)
            let runEmph = 0.3 + 0.7 * min(max((0.6 - pPass) / 0.2, 0), 1)

            var labels: [(Text, CGPoint, UnitPoint, Double)] = []   // drawn last, on pills, so lines never cross them
            if hasPrediction {
                let bands: [(String, Double, Double, Double)] = [   // label, p, from yards, to yards
                    ("DEEP 10+", probs.v[6], 9.5, 21), ("SHORT 1–9", probs.v[5], 0, 9.5), ("BEHIND LOS", probs.v[4], -3, 0),
                ]
                for (label, p, from, to) in bands {
                    let rect = CGRect(x: 0, y: y(to), width: w, height: (to - from) * yd)
                    ctx.fill(Path(rect), with: .color(accent.opacity((0.05 + 0.65 * p) * passEmph)))
                    ctx.fill(Path(CGRect(x: 0, y: rect.minY, width: w, height: 1)), with: .color(accent.opacity(0.25 * passEmph)))
                    let text = Text(label + "  ").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                        + Text(pct(p)).font(.system(.title3, design: .rounded, weight: .heavy)).foregroundStyle(ink)
                    let ty = label.hasPrefix("BEHIND") ? rect.midY : rect.minY + 16
                    labels.append((text, CGPoint(x: w - 10, y: ty), .trailing, max(passEmph, 0.45)))
                }
            }

            // Line of scrimmage and first-down line.
            ctx.fill(Path(CGRect(x: 0, y: los - 1, width: w, height: 2)), with: .color(.primary.opacity(0.55)))
            if let ydstogo, Double(ydstogo) <= 20.5 {
                ctx.fill(Path(CGRect(x: 0, y: y(Double(ydstogo)) - 1, width: w, height: 2)), with: .color(.yellow.opacity(0.9)))
            }

            // Offense: five linemen, QB and back. Small and quiet.
            let player = Color.primary.opacity(0.6), r = yd * 0.5
            for k in -2...2 {
                ctx.stroke(Path(ellipseIn: CGRect(x: w / 2 + Double(k) * yd * 1.5 - r, y: y(-1) - r, width: 2 * r, height: 2 * r)),
                           with: .color(player), lineWidth: 1.5)
            }
            let back = CGPoint(x: w / 2, y: y(-6.5))
            for p in [CGPoint(x: w / 2, y: y(-4)), back] {
                ctx.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)), with: .color(player))
            }

            defer { drawLabels(labels, in: &ctx, size: size) }
            guard hasPrediction else { return }
            let runs: [(Double, CGPoint)] = [
                (probs.v[1], CGPoint(x: w * 0.18, y: y(5))), (probs.v[2], CGPoint(x: w / 2, y: y(6.5))), (probs.v[3], CGPoint(x: w * 0.82, y: y(5))),
            ]
            let maxRun = max(runs.map(\.0).max() ?? 1, 0.01)
            for (p, tip) in runs {
                let thickness = 2 + 14 * p
                ctx.fill(arrow(from: CGPoint(x: back.x, y: back.y - r - 2), to: tip, thickness: thickness),
                         with: .color(accent.opacity(runEmph * (0.35 + 0.65 * p / maxRun))))
                let dx = tip.x - back.x, dy = tip.y - back.y, len = max(hypot(dx, dy), 1)
                labels.append((Text(pct(p)).font(.system(.headline, design: .rounded, weight: .heavy)).foregroundStyle(ink),
                               CGPoint(x: tip.x + dx / len * 16, y: tip.y + dy / len * 16), .center, max(runEmph, 0.45)))
            }
        }
    }

    private func drawLabels(_ labels: [(Text, CGPoint, UnitPoint, Double)], in ctx: inout GraphicsContext, size: CGSize) {
        for (text, point, anchor, opacity) in labels {
            let resolved = ctx.resolve(text)
            let s = resolved.measure(in: size)
            let pill = CGRect(x: point.x - s.width * anchor.x - 6, y: point.y - s.height * anchor.y - 2,
                              width: s.width + 12, height: s.height + 4)
            var c = ctx
            c.opacity = opacity
            c.fill(Path(roundedRect: pill, cornerRadius: pill.height / 2), with: .style(.background.opacity(0.75)))
            c.draw(resolved, at: point, anchor: anchor)
        }
    }

    private func pct(_ v: Double) -> String { "\(Int((v * 100).rounded()))%" }

    /// Filled arrow polygon: a shaft of the given thickness plus a head.
    private func arrow(from a: CGPoint, to b: CGPoint, thickness t: Double) -> Path {
        let dx = b.x - a.x, dy = b.y - a.y, len = max(hypot(dx, dy), 1)
        let ux = dx / len, uy = dy / len, nx = -uy, ny = ux
        let headLen = min(len * 0.4, 8 + t), headHalf = t / 2 + 6
        let base = CGPoint(x: b.x - ux * headLen, y: b.y - uy * headLen)
        var path = Path()
        path.move(to: CGPoint(x: a.x + nx * t / 2, y: a.y + ny * t / 2))
        path.addLine(to: CGPoint(x: base.x + nx * t / 2, y: base.y + ny * t / 2))
        path.addLine(to: CGPoint(x: base.x + nx * headHalf, y: base.y + ny * headHalf))
        path.addLine(to: b)
        path.addLine(to: CGPoint(x: base.x - nx * headHalf, y: base.y - ny * headHalf))
        path.addLine(to: CGPoint(x: base.x - nx * t / 2, y: base.y - ny * t / 2))
        path.addLine(to: CGPoint(x: a.x - nx * t / 2, y: a.y - ny * t / 2))
        path.closeSubpath()
        return path
    }
}

extension PlayPrediction {
    /// Sample predictions for previews (no models needed).
    static let mockPass = PlayPrediction(pPass: 0.74, direction: ["left": 0.30, "middle": 0.45, "right": 0.25],
                                         depth: ["<=0": 0.17, "1-9": 0.56, "10+": 0.27])
    static let mockRun = PlayPrediction(pPass: 0.36, direction: ["left": 0.22, "middle": 0.50, "right": 0.28],
                                        depth: ["<=0": 0.25, "1-9": 0.52, "10+": 0.23])
}

private struct FieldDiagramPreview: View {
    @State private var pass = true
    var body: some View {
        FieldDiagramView(prediction: pass ? .mockPass : .mockRun, ydstogo: 8)
            .padding(20)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .padding()
            .onTapGesture { pass.toggle() }
    }
}

#Preview("Pass, light") { FieldDiagramPreview() }
#Preview("Pass, dark") { FieldDiagramPreview().preferredColorScheme(.dark) }
#Preview("Run") { FieldDiagramView(prediction: .mockRun).padding() }
#Preview("No read") { FieldDiagramView(prediction: nil).padding() }
