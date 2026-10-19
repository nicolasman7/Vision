import SwiftUI

/// First launch: pick a favorite color. The caller decides when to show it and what `onDone` does.
struct OnboardingView: View {
    let onDone: () -> Void
    @AppStorage("accentColor") private var accent: AccentChoice = .blue

    var body: some View {
        VStack(spacing: 28) {
            Spacer()
            Image(systemName: "football.fill")
                .font(.system(size: 64))
                .foregroundStyle(accent.color.gradient)
                .symbolEffect(.bounce, value: accent)
            VStack(spacing: 8) {
                Text("Pick your color")
                    .font(.system(.largeTitle, design: .rounded, weight: .heavy))
                Text("Used for every number, bar and highlight. You can change it later in Settings.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            AccentPicker(accent: $accent)
                .padding(20)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            Spacer()
            Button(action: onDone) {
                Text("Continue").font(.headline).frame(maxWidth: .infinity).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .onAccentLabel()
            .controlSize(.large)
        }
        .padding(24)
        .background(Color(.systemGroupedBackground))
        .appTheme()
    }
}

#Preview("Light") { OnboardingView {} }
#Preview("Dark") { OnboardingView {}.preferredColorScheme(.dark) }
