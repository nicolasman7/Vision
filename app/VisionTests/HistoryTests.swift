import XCTest
@testable import Vision

/// Plays loaded from the bundled history.sqlite must predict exactly what Python predicted for the same rows
/// (Fixtures/history_parity.json, written by scripts/build_history_db.py). This checks the row -> PlayInputs mapping.
final class HistoryTests: XCTestCase {
    private func fixtures() throws -> [[String: Any]] {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "history_parity", withExtension: "json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
    }

    func testMatchesPythonOnRealPlays() throws {
        let store = try HistoryStore()
        let predictor = try PlayPredictor()
        let rows = try fixtures()
        XCTAssertFalse(rows.isEmpty)
        for f in rows {
            let gameID = try XCTUnwrap(f["game_id"] as? String), playID = try XCTUnwrap(f["play_id"] as? Int)
            let name = "\(gameID) #\(playID)"
            let play = try XCTUnwrap(store.play(gameID: gameID, playID: playID), name)
            let got = try predictor.predict(play.inputs)
            XCTAssertEqual(got.pPass, try XCTUnwrap(f["p_pass"] as? Double), accuracy: 1e-4, name)
            for head in ["direction", "depth"] {
                let expected = try XCTUnwrap(f[head] as? [String: Double], name)
                let actual = head == "direction" ? got.direction : got.depth
                for (c, p) in expected { XCTAssertEqual(actual[c] ?? .nan, p, accuracy: 1e-4, "\(name) \(head) \(c)") }
            }
        }
    }

    /// The clock lookup returns the last play snapped at or before the requested moment: its clock is at or
    /// above the request, and the next snap in the same quarter is already past it.
    func testClockLookupReturnsPlayAtOrBefore() throws {
        let store = try HistoryStore()
        for f in try fixtures() {
            let gameID = try XCTUnwrap(f["game_id"] as? String), playID = try XCTUnwrap(f["play_id"] as? Int)
            let anchor = try XCTUnwrap(store.play(gameID: gameID, playID: playID))
            let qtr = try XCTUnwrap(anchor.inputs.qtr), clock = try XCTUnwrap(anchor.inputs.quarterSecondsRemaining)

            for request in [clock, max(0, clock - 3)] {
                let got = try XCTUnwrap(store.play(gameID: gameID, qtr: qtr, clock: request))
                XCTAssertEqual(got.inputs.qtr, qtr)
                XCTAssertGreaterThanOrEqual(try XCTUnwrap(got.inputs.quarterSecondsRemaining), request)
                XCTAssertGreaterThanOrEqual(got.playID, anchor.playID, "never earlier than a play that qualifies")
                if let next = store.next(got), next.inputs.qtr == qtr {
                    XCTAssertLessThan(try XCTUnwrap(next.inputs.quarterSecondsRemaining), request)
                }
            }
        }
    }

    func testPrevNextAndTimeParsing() throws {
        let store = try HistoryStore()
        let gameID = try XCTUnwrap(try fixtures().first?["game_id"] as? String)
        let first = try XCTUnwrap(store.play(gameID: gameID, qtr: 1, clock: 900))   // before kickoff -> first snap
        XCTAssertNil(store.previous(first))
        let second = try XCTUnwrap(store.next(first))
        XCTAssertEqual(store.previous(second)?.playID, first.playID)

        XCTAssertEqual(HistoryView.seconds("8:42"), 522)
        XCTAssertEqual(HistoryView.seconds("15:00"), 900)
        XCTAssertNil(HistoryView.seconds("16:00"))
        XCTAssertNil(HistoryView.seconds("abc"))
    }
}
