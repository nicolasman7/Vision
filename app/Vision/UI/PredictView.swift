import SwiftUI

/// Root screen: situation inputs at the bottom, live prediction on top.
struct PredictView: View {
    @State private var inputs = PlayInputs()
    @State private var predictor: PlayPredictor?
    @State private var loadError: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let loadError {
                        ContentUnavailableView("Models not loaded", systemImage: "exclamationmark.triangle",
                                               description: Text(loadError))
                    } else {
                        // Recomputed on every body pass, so it follows every input change.
                        ResultCardView(prediction: try? predictor?.predict(inputs),
                                       known: InputCount.known(inputs), total: InputCount.total(inputs))
                    }
                }
                InputPanelView(inputs: $inputs)
            }
            .navigationTitle("Play Call")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Reset", systemImage: "arrow.counterclockwise") { inputs = PlayInputs() }
                        .disabled(InputCount.known(inputs) == 0)
                }
            }
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

/// Counts filled-in inputs by reflection, so new PlayInputs fields are counted automatically.
enum InputCount {
    static func known(_ inputs: PlayInputs) -> Int {
        Mirror(reflecting: inputs).children.filter { !Mirror(reflecting: $0.value).children.isEmpty }.count
    }
    static func total(_ inputs: PlayInputs) -> Int { Mirror(reflecting: inputs).children.count }
}

#Preview {
    PredictView()
}
