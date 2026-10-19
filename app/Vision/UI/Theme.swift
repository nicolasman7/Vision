import SwiftUI
import UIKit

/// The user's favorite color. Stored by name under @AppStorage("accentColor").
enum AccentChoice: String, CaseIterable, Identifiable {
    case red, orange, yellow, green, mint, teal, blue, purple, pink

    var id: String { rawValue }
    var name: String { rawValue.capitalized }

    /// Fill color: swatches, bars, bands, selected chips.
    var color: Color {
        switch self {
        case .red: .red
        case .orange: .orange
        case .yellow: .yellow
        case .green: .green
        case .mint: .mint
        case .teal: .teal
        case .blue: .blue
        case .purple: .purple
        case .pink: .pink
        }
    }

    /// Label color on an accent fill (prominent buttons, selected chips, swatch checkmark).
    var onAccent: Color { self == .yellow || self == .mint ? .black : .white }

    /// Text color: the same hue, darkened in light mode where the bright shade is hard to read on white.
    var textColor: Color {
        let lightModeBrightness: CGFloat = switch self {
        case .yellow: 0.62
        case .mint, .green: 0.68
        case .teal: 0.75
        case .orange: 0.85
        default: 1
        }
        let base = UIColor(color)
        return Color(UIColor { traits in
            let c = base.resolvedColor(with: traits)
            guard traits.userInterfaceStyle != .dark, lightModeBrightness < 1 else { return c }
            var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            c.getHue(&h, saturation: &s, brightness: &b, alpha: &a)
            return UIColor(hue: h, saturation: s, brightness: b * lightModeBrightness, alpha: a)
        })
    }
}

/// System / Light / Dark. Stored under @AppStorage("appearance").
enum AppearanceChoice: String, CaseIterable, Identifiable {
    case system, light, dark

    var id: String { rawValue }
    var name: String { rawValue.capitalized }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }
}

private struct AppThemeModifier: ViewModifier {
    @AppStorage("accentColor") private var accent: AccentChoice = .blue
    @AppStorage("appearance") private var appearance: AppearanceChoice = .system

    func body(content: Content) -> some View {
        content
            .foregroundStyle(accent.textColor)      // all text; .secondary/.tertiary become faded accent
            .tint(accent.color)                      // controls, selected chips, bars
            .preferredColorScheme(appearance.colorScheme)
            .onChange(of: accent, initial: true) { _, new in applyUIKitAccent(new) }
            .onChange(of: appearance, initial: true) { _, new in
                // preferredColorScheme(nil) doesn't always switch back to System once Light/Dark was forced,
                // so also set the window style directly.
                let style: UIUserInterfaceStyle = switch new {
                case .system: .unspecified
                case .light: .light
                case .dark: .dark
                }
                for case let scene as UIWindowScene in UIApplication.shared.connectedScenes {
                    scene.windows.forEach { $0.overrideUserInterfaceStyle = style }
                }
            }
    }
}

/// UIKit-drawn controls (segmented pickers, tab bar) ignore SwiftUI's foregroundStyle, so color them here:
/// the appearance proxies cover controls created later, the view walk updates ones already on screen.
private func applyUIKitAccent(_ accent: AccentChoice) {
    let text = UIColor(accent.textColor)
    let faded = UIColor { text.resolvedColor(with: $0).withAlphaComponent(0.5) }
    func style(_ s: UISegmentedControl) {
        s.setTitleTextAttributes([.foregroundColor: text], for: .normal)
        s.setTitleTextAttributes([.foregroundColor: text], for: .selected)
    }
    style(UISegmentedControl.appearance())
    UITabBar.appearance().unselectedItemTintColor = faded
    func walk(_ v: UIView) {
        if let s = v as? UISegmentedControl { style(s) }
        if let t = v as? UITabBar { t.unselectedItemTintColor = faded }
        v.subviews.forEach(walk)
    }
    for case let scene as UIWindowScene in UIApplication.shared.connectedScenes { scene.windows.forEach(walk) }
}

private struct OnAccentLabel: ViewModifier {
    @AppStorage("accentColor") private var accent: AccentChoice = .blue
    func body(content: Content) -> some View { content.foregroundStyle(accent.onAccent) }
}

extension View {
    /// Contrasting label for controls filled with the accent (e.g. `.borderedProminent`). Needed because the
    /// root accent foregroundStyle from `appTheme()` would otherwise paint the label the same color as its fill.
    func onAccentLabel() -> some View { modifier(OnAccentLabel()) }

    /// The app's look: accent-colored text and tint, plus the chosen appearance (System / Light / Dark).
    func appTheme() -> some View { modifier(AppThemeModifier()) }
}

/// Big rounded screen title, used instead of navigation-bar titles (UIKit bar titles can't follow the accent).
struct ScreenTitle: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.system(.largeTitle, design: .rounded, weight: .heavy))
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Small uppercase label above a card, like Apple Sports section headers.
struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased())
            .font(.footnote.weight(.semibold))
            .foregroundStyle(.secondary)
            .tracking(0.6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Grid of color swatches. The screen passes its own `@AppStorage("accentColor")` binding in, so the checkmark
/// and the stored color share one source of truth. (When the picker owned a second @AppStorage instance, it
/// never re-rendered inside a Form row: the color changed but the checkmark stayed on the launch value.)
struct AccentPicker: View {
    @Binding var accent: AccentChoice

    var body: some View {
        let all = AccentChoice.allCases
        Grid(horizontalSpacing: 12, verticalSpacing: 12) {
            ForEach([Array(all[..<5]), Array(all[5...])], id: \.first) { row in
                GridRow {
                    ForEach(row) { choice in
                        Swatch(choice: choice, isSelected: accent == choice) {
                            withAnimation(.spring(duration: 0.3)) { accent = choice }
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)
        .sensoryFeedback(.selection, trigger: accent)
    }
}

private struct Swatch: View {
    let choice: AccentChoice
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Circle()
                .fill(choice.color.gradient)
                .frame(width: 48, height: 48)
                .overlay {
                    if isSelected {
                        Image(systemName: "checkmark").font(.headline.weight(.bold)).foregroundStyle(choice.onAccent)
                    }
                }
                .padding(4)
                .overlay(Circle().strokeBorder(isSelected ? choice.color : .clear, lineWidth: 2.5))
                .scaleEffect(isSelected ? 1.06 : 1)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(choice.name)
        .accessibilityIdentifier("swatch-\(choice.rawValue)")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// System / Light / Dark segmented picker. Takes the screen's `@AppStorage("appearance")` binding (see AccentPicker).
struct AppearancePicker: View {
    @Binding var appearance: AppearanceChoice

    var body: some View {
        Picker("Appearance", selection: $appearance) {
            ForEach(AppearanceChoice.allCases) { Text($0.name).tag($0) }
        }
        .pickerStyle(.segmented)
        .sensoryFeedback(.selection, trigger: appearance)
    }
}
