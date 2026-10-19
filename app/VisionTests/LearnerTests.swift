import XCTest
@testable import Vision

/// The online learner must replay Modeler's Python sequence (Fixtures/learner_sequence.json) exactly.
@MainActor final class LearnerTests: XCTestCase {
    private var fileURL: URL!

    override func setUp() async throws {
        fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("learner-\(UUID().uuidString).json")
    }

    override func tearDown() async throws { try? FileManager.default.removeItem(at: fileURL) }

    private var config: LearnerConfig { .load(bundle: Bundle(for: OnlineLearner.self)) }

    func testReplaysPythonSequence() throws {
        // Fixture = Modeler's config + 30 plays (inputs, p_base, x, y, p_adj_before) + w_final.
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "learner_sequence", withExtension: "json"))
        let data = try Data(contentsOf: url)
        let seqConfig = try JSONDecoder().decode(LearnerConfig.self, from: data)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let plays = try XCTUnwrap(json["plays"] as? [[String: Any]])
        let wFinal = try XCTUnwrap(json["w_final"] as? [Double])
        XCTAssertEqual(plays.count, 30)

        let learner = OnlineLearner(config: seqConfig, fileURL: fileURL)
        for (n, play) in plays.enumerated() {
            let raw = try XCTUnwrap(play["inputs"] as? [String: Any])
            func int(_ k: String) -> Int? { (raw[k] as? Double).map { Int($0) } }
            var i = PlayInputs()
            i.down = int("down"); i.ydstogo = int("ydstogo"); i.yardline100 = int("yardline_100")
            i.formationFamily = (raw["formation_family"] as? String).flatMap(FormationFamily.init(rawValue:))
            i.isEmpty = int("is_empty").map { $0 == 1 }
            i.box = int("ftn_box"); i.scoreDifferential = int("score_differential"); i.qtr = int("qtr")

            // Swift features from PlayInputs must equal Python's x.
            let x = OnlineLearner.features(inputs: i, pPass: try XCTUnwrap(play["p_base"] as? Double),
                                           clipLogit: seqConfig.clipLogit)
            let want = try XCTUnwrap(play["x"] as? [Double])
            XCTAssertEqual(x.count, want.count)
            for (k, (a, b)) in zip(x, want).enumerated() { XCTAssertEqual(a, b, accuracy: 1e-9, "play \(n) x[\(k)]") }

            let y = try XCTUnwrap(play["y"] as? Int)
            XCTAssertEqual(learner.step(x: x, y: Double(y)), try XCTUnwrap(play["p_adj_before"] as? Double),
                           accuracy: 1e-9, "play \(n)")
        }
        XCTAssertEqual(learner.weights.count, wFinal.count)
        for (got, want) in zip(learner.weights, wFinal) { XCTAssertEqual(got, want, accuracy: 1e-9) }
    }

    func testFeatureVector() {
        var i = PlayInputs()
        i.down = 3; i.ydstogo = 25; i.yardline100 = 40; i.formationFamily = .pistol; i.isEmpty = true
        i.box = 7; i.scoreDifferential = -30; i.qtr = 4
        let x = OnlineLearner.features(inputs: i, pPass: 0.5, clipLogit: 3)
        XCTAssertEqual(x, [1, 0, 0, 0, 1, 0, 2, 1, 0.4, 0, 0, 1, 1, 0.25, -1.5, 1])
        XCTAssertEqual(OnlineLearner.features(inputs: PlayInputs(), pPass: 1, clipLogit: 3),
                       [1, 3] + Array(repeating: 0, count: 14))
    }

    func testLearnIsIdempotentPerPlay() {
        let learner = OnlineLearner(config: config, fileURL: fileURL)
        let base = PlayPrediction(pPass: 0.8, direction: [:], depth: [:])
        let first = learner.learn(gameId: "g", playId: 1, inputs: PlayInputs(), base: base, wasPass: false)
        let w = learner.weights
        let again = learner.learn(gameId: "g", playId: 1, inputs: PlayInputs(), base: base, wasPass: true)
        XCTAssertEqual(learner.weights, w)
        XCTAssertEqual(again.wasPass, first.wasPass)
        XCTAssertEqual(learner.stats.learned, 1)
        XCTAssertTrue(learner.hasLearned(gameId: "g", playId: 1))
        XCTAssertFalse(first.baseCorrect)

        // Persisted: a fresh learner on the same file remembers it.
        let reloaded = OnlineLearner(config: config, fileURL: fileURL)
        XCTAssertTrue(reloaded.hasLearned(gameId: "g", playId: 1))
        XCTAssertEqual(reloaded.weights, w)
    }

    func testResetRestoresW0() {
        let learner = OnlineLearner(config: config, fileURL: fileURL)
        let w0 = learner.weights
        XCTAssertEqual(w0, config.w0)
        learner.learn(gameId: "g", playId: 7, inputs: PlayInputs(), base: PlayPrediction(pPass: 0.3, direction: [:], depth: [:]), wasPass: true)
        XCTAssertNotEqual(learner.weights, w0)
        learner.reset()
        XCTAssertEqual(learner.weights, w0)
        XCTAssertEqual(learner.stats.learned, 0)
        XCTAssertFalse(learner.hasLearned(gameId: "g", playId: 7))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fileURL.path))
    }

    func testMissExplainerSortedAndSkipsUnknown() throws {
        var i = PlayInputs()
        i.down = 3; i.ydstogo = 8; i.yardline100 = 60; i.formationFamily = .shotgun; i.box = 6; i.hash = .left
        let factors = MissExplainer.factors(predictor: try PlayPredictor(), inputs: i)
        XCTAssertEqual(Set(factors.map(\.name)), ["Down & distance", "Field position", "Formation", "Box", "Hash"])
        XCTAssertEqual(factors.map { abs($0.delta) }, factors.map { abs($0.delta) }.sorted(by: >))
        XCTAssertTrue(MissExplainer.factors(predictor: try PlayPredictor(), inputs: PlayInputs()).isEmpty)
    }
}
