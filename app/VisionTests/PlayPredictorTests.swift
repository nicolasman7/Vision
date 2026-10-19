import XCTest
@testable import Vision

/// Swift + Core ML must match Python (predictor.predict) on the scenarios in Fixtures/parity.json.
/// Regenerate them with: uv run python -m scripts.make_app_fixtures
final class PlayPredictorTests: XCTestCase {
    private static let ints: [String: WritableKeyPath<PlayInputs, Int?>] = [
        "down": \.down, "ydstogo": \.ydstogo, "yardline100": \.yardline100, "qtr": \.qtr,
        "quarterSecondsRemaining": \.quarterSecondsRemaining, "scoreDifferential": \.scoreDifferential,
        "posteamTimeouts": \.posteamTimeouts, "defteamTimeouts": \.defteamTimeouts,
        "backfieldCount": \.backfieldCount, "offOL": \.offOL, "offRB": \.offRB, "offTE": \.offTE, "offWR": \.offWR,
        "defDL": \.defDL, "defLB": \.defLB, "defDB": \.defDB, "box": \.box,
    ]
    private static let bools: [String: WritableKeyPath<PlayInputs, Bool?>] = ["noHuddle": \.noHuddle, "isEmpty": \.isEmpty]

    private func playInputs(_ json: [String: Any]) throws -> PlayInputs {
        var p = PlayInputs()
        for (key, value) in json {
            if let path = Self.ints[key] { p[keyPath: path] = try XCTUnwrap(value as? Int, key) }
            else if let path = Self.bools[key] { p[keyPath: path] = try XCTUnwrap(value as? Bool, key) }
            else if key == "formationFamily" { p.formationFamily = try XCTUnwrap(FormationFamily(rawValue: value as? String ?? "")) }
            else if key == "hash" { p.hash = try XCTUnwrap(HashMark(rawValue: value as? String ?? "")) }
            else { XCTFail("fixture has unknown input \(key)") }
        }
        return p
    }

    func testMatchesPythonFixtures() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "parity", withExtension: "json"))
        let fixtures = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(fixtures.count, 20)

        let predictor = try PlayPredictor()
        for f in fixtures {
            let name = f["name"] as? String ?? "?"
            let got = try predictor.predict(playInputs(f["inputs"] as? [String: Any] ?? [:]))
            XCTAssertEqual(got.pPass, try XCTUnwrap(f["p_pass"] as? Double), accuracy: 1e-4, name)
            for head in ["direction", "depth"] {
                let expected = try XCTUnwrap(f[head] as? [String: Double], name)
                let actual = head == "direction" ? got.direction : got.depth
                XCTAssertEqual(Set(actual.keys), Set(expected.keys), "\(name) \(head)")
                for (c, p) in expected {
                    XCTAssertEqual(actual[c] ?? .nan, p, accuracy: 1e-4, "\(name) \(head) \(c)")
                }
            }
        }
    }

    func testProbabilitiesSumToOneAndAllNilWorks() throws {
        let got = try PlayPredictor().predict(PlayInputs())
        XCTAssert((0...1).contains(got.pPass))
        XCTAssertEqual(got.direction.values.reduce(0, +), 1, accuracy: 1e-9)
        XCTAssertEqual(got.depth.values.reduce(0, +), 1, accuracy: 1e-9)
    }
}
