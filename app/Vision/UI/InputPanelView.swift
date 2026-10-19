import SwiftUI

/// Situation inputs. Every control has an unknown ("?") state that maps to nil.
struct InputPanelView: View {
    @Binding var inputs: PlayInputs
    @State private var showMore = false

    var body: some View {
        Section("Situation") {
            Picker("Down", selection: $inputs.down) {
                Text("?").tag(Int?.none)
                ForEach(1...4, id: \.self) { Text(ordinal($0)).tag(Int?($0)) }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Down")

            VStack(alignment: .leading, spacing: 8) {
                Stepper(value: Binding(get: { inputs.ydstogo ?? 10 }, set: { inputs.ydstogo = $0 }), in: 1...99) {
                    LabeledContent("Distance", value: inputs.ydstogo.map { "\($0) yd" } ?? "?")
                }
                ChipRow(options: [1, 2, 3, 5, 7, 10, 15].map { ($0 == 15 ? "15+" : "\($0)", $0) },
                        selection: $inputs.ydstogo, name: "Distance")
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    LabeledContent("Field position", value: inputs.yardline100.map(fieldPosition) ?? "?")
                    if inputs.yardline100 != nil { clearButton("field position") { inputs.yardline100 = nil } }
                }
                Slider(value: Binding(get: { Double(inputs.yardline100 ?? 50) },
                                      set: { inputs.yardline100 = Int($0.rounded()) }),
                       in: 1...99, step: 1) {
                    Text("Yards to end zone")
                } minimumValueLabel: { Text("Opp") } maximumValueLabel: { Text("Own") }
                .accessibilityValue(inputs.yardline100.map(fieldPosition) ?? "unknown")
            }
        }

        Section("Offense") {
            ChipRow(options: FormationFamily.allCases.map { ($0.label, $0) },
                    selection: $inputs.formationFamily, name: "Formation")

            Picker("Empty backfield", selection: $inputs.isEmpty) {
                Text("?").tag(Bool?.none)
                Text("Yes").tag(Bool?(true))
                Text("No").tag(Bool?(false))
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Personnel").font(.subheadline).foregroundStyle(.secondary)
                ChipRow(options: ["11", "12", "21", "13", "10", "22"].map { ($0, $0) },
                        selection: personnel, name: "Personnel")
            }
        }

        Section {
            DisclosureGroup("More", isExpanded: $showMore) {
                Stepper(value: Binding(get: { inputs.scoreDifferential ?? 0 }, set: { inputs.scoreDifferential = $0 }),
                        in: -50...50) {
                    LabeledContent("Score (offense − defense)",
                                   value: inputs.scoreDifferential.map { $0 > 0 ? "+\($0)" : "\($0)" } ?? "?")
                }
                if inputs.scoreDifferential != nil {
                    clearButton("score") { inputs.scoreDifferential = nil }
                }
                Picker("Quarter", selection: $inputs.qtr) {
                    Text("?").tag(Int?.none)
                    ForEach(1...4, id: \.self) { Text("Q\($0)").tag(Int?($0)) }
                    Text("OT").tag(Int?(5))
                }
                Picker("Hash", selection: $inputs.hash) {
                    Text("?").tag(HashMark?.none)
                    ForEach(HashMark.allCases, id: \.self) { Text($0.label).tag(HashMark?($0)) }
                }
            }
        }
    }

    /// Personnel code "XY" = X running backs, Y tight ends, the rest receivers. OL stays unknown.
    private var personnel: Binding<String?> {
        Binding(
            get: {
                guard let rb = inputs.offRB, let te = inputs.offTE else { return nil }
                return "\(rb)\(te)"
            },
            set: { code in
                let digits = code.map { $0.compactMap(\.wholeNumberValue) }
                inputs.offRB = digits?[0]
                inputs.offTE = digits?[1]
                inputs.offWR = digits.map { 5 - $0[0] - $0[1] }
            })
    }

    private func clearButton(_ what: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
            .buttonStyle(.borderless)
            .accessibilityLabel("Clear \(what)")
    }

    private func ordinal(_ n: Int) -> String { ["1st", "2nd", "3rd", "4th"][n - 1] }

    private func fieldPosition(_ toGo: Int) -> String {
        toGo == 50 ? "Midfield" : toGo > 50 ? "Own \(100 - toGo)" : "Opp \(toGo)"
    }
}

/// Horizontal row of tappable chips plus a leading "?" chip for unknown. Tapping the selected chip clears it.
struct ChipRow<Value: Hashable>: View {
    let options: [(String, Value)]
    @Binding var selection: Value?
    let name: String

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                chip("?", isOn: selection == nil, accessibility: "Unknown") { selection = nil }
                ForEach(options, id: \.0) { label, value in
                    chip(label, isOn: selection == value, accessibility: label) {
                        selection = selection == value ? nil : value
                    }
                }
            }
            .padding(.vertical, 2)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(name)
    }

    private func chip(_ label: String, isOn: Bool, accessibility: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.subheadline.bold())
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(isOn ? Color.accentColor : Color(.tertiarySystemFill), in: Capsule())
                .foregroundStyle(isOn ? Color.white : Color.primary)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel("\(name) \(accessibility)")
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

extension FormationFamily {
    var label: String {
        switch self {
        case .shotgun: "Shotgun"
        case .underCenter: "Under center"
        case .pistol: "Pistol"
        }
    }
}

extension HashMark {
    var label: String {
        switch self {
        case .left: "Left"
        case .middle: "Middle"
        case .right: "Right"
        }
    }
}

#Preview {
    @Previewable @State var inputs = PlayInputs(down: 3, ydstogo: 7, formationFamily: .shotgun)
    Form { InputPanelView(inputs: $inputs) }
}
