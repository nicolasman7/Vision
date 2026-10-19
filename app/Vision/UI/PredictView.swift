import SwiftUI

/// Predict tab: the field diagram as the hero, a scorebug-style situation strip below it.
struct PredictView: View {
    @State private var inputs = PlayInputs()
    @State private var predictor: PlayPredictor?
    @State private var loadError: String?
    @State private var editing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    ScreenTitle("Predict")
                    VStack(spacing: 8) {
                        SectionLabel("Prediction")
                        Group {
                            if let loadError {
                                ContentUnavailableView("Models not loaded", systemImage: "exclamationmark.triangle",
                                                       description: Text(loadError))
                            } else {
                                // Recomputed on every body pass, so it follows every input change.
                                FieldDiagramView(prediction: try? predictor?.predict(inputs), ydstogo: inputs.ydstogo)
                            }
                        }
                        .padding(18)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
                    }

                    VStack(spacing: 8) {
                        SectionLabel("Situation")
                        SituationStrip(inputs: inputs) { editing = true }
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Label("About 12% of dropbacks end in a sack or scramble (no throw).", systemImage: "info.circle")
                        Text("\(InputCount.known(inputs)) of \(InputCount.total(inputs)) inputs set. Unknowns are fine; the model fills the gaps.")
                    }
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .background(Color(.systemGroupedBackground))
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $editing) { SituationSheet(inputs: $inputs).appTheme() }
        }
        .onAppear(perform: loadModels)
    }

    private func loadModels() {
        guard predictor == nil, loadError == nil else { return }
        do { predictor = try PlayPredictor() } catch {
            loadError = "The prediction models are missing from the app bundle. Export them with "
                + "`uv run python -m scripts.export_coreml`, then rebuild.\n\n(\(error.localizedDescription))"
        }
    }
}

/// Scorebug-style summary: "3rd & 8 · OPP 35". Tap to edit.
struct SituationStrip: View {
    let inputs: PlayInputs
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 8) {
                        Text(inputs.downDistanceText ?? "Any down")
                        if let field = inputs.fieldText {
                            Text("·").foregroundStyle(.tertiary)
                            Text(field)
                        }
                    }
                    .font(.system(.title2, design: .rounded, weight: .bold))
                    .monospacedDigit()
                    if inputs.offenseText == nil && inputs.defenseText == nil {
                        Text("Tap to set formation and personnel").font(.subheadline).foregroundStyle(.secondary)
                    }
                    ForEach([inputs.offenseText, inputs.defenseText].compactMap { $0 }, id: \.self) {
                        Text($0).font(.subheadline).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Image(systemName: "slider.horizontal.3")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.tint)
            }
            .padding(.horizontal, 18).padding(.vertical, 14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityHint("Edit the situation")
    }
}

/// Counts filled-in inputs by reflection, so new PlayInputs fields are counted automatically.
enum InputCount {
    static func known(_ inputs: PlayInputs) -> Int {
        Mirror(reflecting: inputs).children.filter { !Mirror(reflecting: $0.value).children.isEmpty }.count
    }
    static func total(_ inputs: PlayInputs) -> Int { Mirror(reflecting: inputs).children.count }
}

#Preview("Light") { PredictView().appTheme() }
#Preview("Dark") { PredictView().appTheme().preferredColorScheme(.dark) }
#Preview("Strip") {
    SituationStrip(inputs: PlayInputs(down: 3, ydstogo: 8, yardline100: 35, formationFamily: .shotgun,
                                      offRB: 1, offTE: 1, offWR: 3, defDL: 4, defLB: 2, defDB: 5, box: 6)) {}
        .padding().appTheme()
}
