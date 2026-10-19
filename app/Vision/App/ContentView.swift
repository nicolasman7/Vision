import SwiftUI

struct ContentView: View {
    @AppStorage("hasOnboarded") private var hasOnboarded = false

    var body: some View {
        TabView {
            LiveView()
                .tabItem { Label("Live", systemImage: "camera.viewfinder") }
            PredictView()
                .tabItem { Label("Predict", systemImage: "sportscourt") }
            HistoryView()
                .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .appTheme()
        // Shown until onboarding finishes; the binding inverts the stored flag.
        .fullScreenCover(isPresented: Binding(get: { !hasOnboarded }, set: { hasOnboarded = !$0 })) {
            OnboardingView(onDone: { hasOnboarded = true })
        }
    }
}

#Preview {
    ContentView()
}
