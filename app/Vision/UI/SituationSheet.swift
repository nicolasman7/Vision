import SwiftUI

/// Sheet for editing the situation. Every control has an unknown state ("?" / "Any") that maps to nil.
struct SituationSheet: View {
    @Binding var inputs: PlayInputs
    @Environment(\.dismiss) private var dismiss
    @State private var showMore = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Down & distance") {
                    ChipRow(options: (1...4).map { (PlayInputs.ordinal($0), $0) }, selection: $inputs.down, name: "Down")
                    ChipRow(options: [1, 2, 3, 5, 7, 10, 15].map { ($0 == 15 ? "15+" : "\($0)", $0) },
                            selection: $inputs.ydstogo, name: "Distance")
                    Stepper(value: Binding(get: { inputs.ydstogo ?? 10 }, set: { inputs.ydstogo = $0 }), in: 1...99) {
                        LabeledContent("Distance", value: inputs.ydstogo.map { "\($0) yd" } ?? "Any")
                    }
                }

                Section("Field position") {
                    HStack {
                        Text(inputs.fieldText ?? "Any")
                            .font(.system(.title3, design: .rounded, weight: .bold)).monospacedDigit()
                        Spacer()
                        if inputs.yardline100 != nil { ClearButton(what: "field position") { inputs.yardline100 = nil } }
                    }
                    Slider(value: Binding(get: { Double(100 - (inputs.yardline100 ?? 50)) },
                                          set: { inputs.yardline100 = 100 - Int($0.rounded()) }),
                           in: 1...99, step: 1) {
                        Text("Field position")
                    } minimumValueLabel: { Text("Own").font(.caption) } maximumValueLabel: { Text("Opp").font(.caption) }
                    .accessibilityValue(inputs.fieldText ?? "unknown")
                }

                Section("Offense") {
                    ChipRow(options: FormationFamily.allCases.map { ($0.label, $0) },
                            selection: $inputs.formationFamily, name: "Formation")
                    ChipRow(options: ["11", "12", "21", "13", "10", "22"].map { ($0, $0) },
                            selection: personnel, name: "Personnel")
                    Picker("Empty backfield", selection: $inputs.isEmpty) {
                        Text("Any").tag(Bool?.none)
                        Text("Yes").tag(Bool?(true))
                        Text("No").tag(Bool?(false))
                    }
                }

                Section("Defense") {
                    ChipRow(options: PlayInputs.defensePresets.map { ($0.name, $0.name) },
                            selection: defense, name: "Defense personnel")
                    ChipRow(options: [5, 6, 7, 8, 9].map { ($0 == 9 ? "9+" : "\($0)", $0) },
                            selection: $inputs.box, name: "Men in the box")
                }

                Section {
                    DisclosureGroup("More", isExpanded: $showMore) {
                        Stepper(value: Binding(get: { inputs.scoreDifferential ?? 0 }, set: { inputs.scoreDifferential = $0 }),
                                in: -50...50) {
                            HStack {
                                LabeledContent("Score margin",
                                               value: inputs.scoreDifferential.map { $0 > 0 ? "+\($0)" : "\($0)" } ?? "Any")
                                if inputs.scoreDifferential != nil {
                                    ClearButton(what: "score") { inputs.scoreDifferential = nil }
                                }
                            }
                        }
                        Picker("Quarter", selection: $inputs.qtr) {
                            Text("Any").tag(Int?.none)
                            ForEach(1...4, id: \.self) { Text("Q\($0)").tag(Int?($0)) }
                            Text("OT").tag(Int?(5))
                        }
                        Picker("Hash", selection: $inputs.hash) {
                            Text("Any").tag(HashMark?.none)
                            ForEach(HashMark.allCases, id: \.self) { Text($0.label).tag(HashMark?($0)) }
                        }
                    }
                }
            }
            .toolbar { ToolbarItem(placement: .principal) { Text("Situation").font(.headline) } }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Reset") { withAnimation { inputs = PlayInputs() } }
                        .disabled(InputCount.known(inputs) == 0)
                }
                ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() }.bold() }
            }
            .sensoryFeedback(.selection, trigger: inputs.qtr)
            .sensoryFeedback(.selection, trigger: inputs.hash)
            .sensoryFeedback(.selection, trigger: inputs.isEmpty)
        }
        .presentationDetents([.medium, .large])
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

    /// Defense preset name <-> DL/LB/DB counts. A custom mix that matches no preset shows as "Any".
    private var defense: Binding<String?> {
        Binding(
            get: { inputs.defensePresetName },
            set: { name in
                let preset = PlayInputs.defensePresets.first { $0.name == name }
                inputs.defDL = preset?.dl
                inputs.defLB = preset?.lb
                inputs.defDB = preset?.db
            })
    }
}

/// Row of pill chips with a leading "Any" chip for unknown. Tapping the selected chip clears it.
struct ChipRow<Value: Hashable>: View {
    let options: [(String, Value)]
    @Binding var selection: Value?
    let name: String
    @AppStorage("accentColor") private var accent: AccentChoice = .blue

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(name).font(.subheadline).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    chip("Any", value: nil)
                    ForEach(options, id: \.0) { label, value in chip(label, value: value) }
                }
                .padding(.vertical, 2)
            }
        }
        .sensoryFeedback(.selection, trigger: selection)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(name)
    }

    private func chip(_ label: String, value: Value?) -> some View {
        let isOn = selection == value
        return Button {
            withAnimation(.spring(duration: 0.25)) { selection = (isOn && value != nil) ? nil : value }
        } label: {
            Text(label)
                .font(.system(.subheadline, design: .rounded, weight: .semibold))
                .padding(.horizontal, 14).padding(.vertical, 8)
                .frame(minWidth: 44)
                .background(isOn ? AnyShapeStyle(.tint) : AnyShapeStyle(.fill.tertiary), in: Capsule())
                .foregroundStyle(isOn ? AnyShapeStyle(accent.onAccent) : AnyShapeStyle(.primary))
        }
        .buttonStyle(.borderless)
        .accessibilityLabel("\(name) \(label)")
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

private struct ClearButton: View {
    let what: String
    let action: () -> Void
    var body: some View {
        Button(action: action) { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
            .buttonStyle(.borderless)
            .accessibilityLabel("Clear \(what)")
    }
}

extension PlayInputs {
    static func ordinal(_ n: Int) -> String { ["1st", "2nd", "3rd", "4th"][max(0, min(n, 4) - 1)] }

    /// "3rd & 8", "3rd down", "& 8", or nil when both are unknown.
    var downDistanceText: String? {
        switch (down, ydstogo) {
        case let (d?, y?): "\(Self.ordinal(d)) & \(y >= 15 ? "15+" : "\(y)")"
        case let (d?, nil): "\(Self.ordinal(d)) down"
        case let (nil, y?): "& \(y)"
        default: nil
        }
    }

    /// "OPP 35", "OWN 25", "50", or nil. yardline100 = yards to the opponent's end zone.
    var fieldText: String? {
        yardline100.map { $0 == 50 ? "50" : $0 < 50 ? "OPP \($0)" : "OWN \(100 - $0)" }
    }

    /// "Shotgun · 11 personnel · Empty", or nil when nothing is set.
    var offenseText: String? {
        var parts: [String] = []
        if let f = formationFamily { parts.append(f.label) }
        if let rb = offRB, let te = offTE { parts.append("\(rb)\(te) personnel") }
        if isEmpty == true { parts.append("Empty") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    static let defensePresets: [(name: String, dl: Int, lb: Int, db: Int)] = [
        ("Base 4-3", 4, 3, 4), ("Base 3-4", 3, 4, 4), ("Nickel", 4, 2, 5), ("Dime", 4, 1, 6), ("Goal line", 6, 3, 2),
    ]

    var defensePresetName: String? {
        Self.defensePresets.first { $0.dl == defDL && $0.lb == defLB && $0.db == defDB }?.name
    }

    /// "Nickel · 6 in box", or nil when nothing is set.
    var defenseText: String? {
        var parts: [String] = []
        if let name = defensePresetName { parts.append(name) }
        if let box { parts.append(box >= 9 ? "9+ in box" : "\(box) in box") }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
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

#Preview("Sheet") {
    @Previewable @State var inputs = PlayInputs(down: 3, ydstogo: 8, yardline100: 35, formationFamily: .shotgun)
    SituationSheet(inputs: $inputs).appTheme()
}

#Preview("Sheet, dark") {
    @Previewable @State var inputs = PlayInputs()
    SituationSheet(inputs: $inputs).appTheme().preferredColorScheme(.dark)
}
