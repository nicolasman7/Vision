import SwiftUI

/// The prediction: PASS vs RUN, then run direction and pass depth.
struct ResultCardView: View {
    let prediction: PlayPrediction?
    let known: Int
    let total: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let p = prediction {
                passRun(p.pPass)
                section("If run →", symbol: "figure.run", color: .green, rows: [
                    ("Left", p.direction["left"]), ("Middle", p.direction["middle"]), ("Right", p.direction["right"]),
                ])
                section("If pass →", symbol: "football.fill", color: .orange, rows: [
                    ("Behind LOS", p.depth["<=0"]), ("Short 1–9", p.depth["1-9"]), ("Deep 10+", p.depth["10+"]),
                ])
                Label("About 12% of dropbacks end in a sack or scramble (no throw).", systemImage: "info.circle")
                    .font(.footnote).foregroundStyle(.secondary)
            } else {
                Label("Waiting for the models…", systemImage: "hourglass").foregroundStyle(.secondary)
            }
            Text("\(known) of \(total) inputs set. Unknown ones are fine; the model fills the gaps.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 6)
    }

    private func passRun(_ pPass: Double) -> some View {
        let isPass = pPass >= 0.5
        let top = max(pPass, 1 - pPass)
        let strength = top > 0.75 ? "Strong" : top >= 0.6 ? "Likely" : "Lean"
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("\(strength) \(isPass ? "PASS" : "RUN")")
                    .font(.largeTitle.bold()).foregroundStyle(isPass ? .orange : .green)
                Spacer()
                Text(top, format: .percent.precision(.fractionLength(0))).font(.title2.monospacedDigit().bold())
            }
            GeometryReader { geo in
                HStack(spacing: 2) {
                    Rectangle().fill(.orange).frame(width: geo.size.width * pPass)
                    Rectangle().fill(.green)
                }
                .clipShape(Capsule())
            }
            .frame(height: 14)
            HStack {
                Text("PASS \(pct(pPass))").foregroundStyle(.orange)
                Spacer()
                Text("RUN \(pct(1 - pPass))").foregroundStyle(.green)
            }
            .font(.caption.bold().monospacedDigit())
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(strength) \(isPass ? "pass" : "run"). Pass \(pct(pPass)), run \(pct(1 - pPass)).")
    }

    private func section(_ title: String, symbol: String, color: Color, rows: [(String, Double?)]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: symbol).font(.headline)
            ForEach(rows, id: \.0) { name, value in
                HStack {
                    Text(name).font(.subheadline).frame(minWidth: 90, alignment: .leading)
                    ProgressView(value: value ?? 0).tint(color)
                    Text(value.map(pct) ?? "–").font(.subheadline.monospacedDigit()).frame(minWidth: 44, alignment: .trailing)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(title.dropLast(2)) \(name): \(value.map(pct) ?? "unknown")")
            }
        }
    }

    private func pct(_ v: Double) -> String { v.formatted(.percent.precision(.fractionLength(0))) }
}

let mockPrediction = PlayPrediction(pPass: 0.68,
                                    direction: ["left": 0.31, "middle": 0.42, "right": 0.27],
                                    depth: ["<=0": 0.18, "1-9": 0.55, "10+": 0.27])

#Preview("Prediction") {
    Form { ResultCardView(prediction: mockPrediction, known: 6, total: 21) }
}

#Preview("Lean run, dark") {
    Form { ResultCardView(prediction: PlayPrediction(pPass: 0.44, direction: mockPrediction.direction,
                                                     depth: mockPrediction.depth), known: 2, total: 21) }
        .preferredColorScheme(.dark)
}

#Preview("No models") {
    Form { ResultCardView(prediction: nil, known: 0, total: 21) }
}
