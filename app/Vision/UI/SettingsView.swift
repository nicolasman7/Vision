import SwiftUI

/// Settings tab: appearance, accent color, and an About page.
struct SettingsView: View {
    @AppStorage("accentColor") private var accent: AccentChoice = .blue
    @AppStorage("appearance") private var appearance: AppearanceChoice = .system

    var body: some View {
        NavigationStack {
            Form {
                ScreenTitle("Settings")
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets())
                Section {
                    AppearancePicker(appearance: $appearance).padding(.vertical, 4)
                    AccentPicker(accent: $accent).padding(.vertical, 8)
                } header: {
                    Text("Look")
                } footer: {
                    Text("System follows your iPhone's Light/Dark setting.")
                }
                Section {
                    NavigationLink("About") { AboutView() }
                }
            }
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

private struct AboutView: View {
    var body: some View {
        List {
            ScreenTitle("About")
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets())
            Section("Model") {
                LabeledContent("Type", value: "Gradient-boosted trees")
                LabeledContent("Runs", value: "On device (Core ML)")
                LabeledContent("Run vs pass", value: "76% on 2025 games")
                LabeledContent("Always guess pass", value: "60%")
            }
            Section {
                Text("Run direction (left / middle / right) and pass depth (behind the line, 1–9, 10+ yards) are separate models. "
                     + "Every input is optional: with nothing filled in, the model still makes its best guess.")
                    .font(.callout)
            }
            Section("Data sources") {
                LabeledContent("Play-by-play", value: "nflverse, 2016–2025")
                LabeledContent("Formation & personnel", value: "nflverse participation")
                LabeledContent("Charting", value: "FTN via nflverse")
            }
            Section {
                LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "–")
            }
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview("Light") { SettingsView().appTheme() }
#Preview("Dark") { SettingsView().appTheme().preferredColorScheme(.dark) }
