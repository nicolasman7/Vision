import XCTest

/// The color picker's checkmark (the `.isSelected` trait) must follow the swatch the user taps,
/// on both screens that show the picker.
final class AccentPickerUITests: XCTestCase {
    override func setUp() { continueAfterFailure = false }

    func testOnboardingCheckmarkFollowsTap() {
        let app = XCUIApplication()
        app.launchArguments = ["-hasOnboarded", "NO"]   // shows onboarding; accentColor itself is never pinned
        app.launch()
        assertCheckmarkFollowsTaps(in: app)
    }

    func testSettingsCheckmarkFollowsTap() {
        let app = XCUIApplication()
        app.launchArguments = ["-hasOnboarded", "YES"]
        app.launch()
        app.tabBars.buttons["Settings"].tap()
        assertCheckmarkFollowsTaps(in: app)
    }

    private func assertCheckmarkFollowsTaps(in app: XCUIApplication) {
        // Start from a known color by tapping it (a `-accentColor` launch argument would pin the value:
        // UserDefaults' argument domain overrides every later write).
        for (tapped, others) in [("blue", ["purple", "green"]), ("purple", ["blue", "green"]), ("green", ["blue", "purple"])] {
            let swatch = app.buttons["swatch-\(tapped)"]
            XCTAssertTrue(swatch.waitForExistence(timeout: 5), "swatch-\(tapped) missing")
            swatch.tap()
            XCTAssertTrue(swatch.isSelected, "\(tapped) should be selected after tapping it")
            for other in others {
                XCTAssertFalse(app.buttons["swatch-\(other)"].isSelected, "\(other) should not be selected after tapping \(tapped)")
            }
        }
    }
}
