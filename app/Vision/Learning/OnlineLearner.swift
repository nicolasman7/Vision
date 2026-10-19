import Foundation

// "Learns from its mistakes": a tiny online logistic regression on top of the frozen run/pass model.
// It sees the base model's logit plus a few inputs, and nudges its weights after each revealed play.
// Math must match Modeler's Python (fixture: VisionTests/Fixtures/learner_sequence.json).

struct LearnOutcome: Codable {
    let baseP: Double
    let adjustedP: Double
    let wasPass: Bool
    var baseCorrect: Bool { (baseP >= 0.5) == wasPass }
    var adjustedCorrect: Bool { (adjustedP >= 0.5) == wasPass }
}

struct LearnerStats {
    let learned: Int
    let baseAccuracy: Double
    let adjustedAccuracy: Double
    let baseLogLoss: Double
    let adjustedLogLoss: Double
}

/// learner_config.json (written by Modeler).
struct LearnerConfig: Decodable {
    let features: [String]
    let w0: [Double]
    let lr: Double
    let l2: Double
    let clipLogit: Double
    enum CodingKeys: String, CodingKey { case features, w0, lr, l2, clipLogit = "clip_logit" }

    /// Spec fallback when the bundle has no config: w0 = identity (weight 1 on the base logit), so p_adj == p_base.
    // ponytail: clip_logit 8 is a guess for the fallback only; the bundled config wins.
    static let fallback = LearnerConfig(features: OnlineLearner.featureNames,
                                        w0: [0, 1] + Array(repeating: 0, count: 14), lr: 0.01, l2: 0.01, clipLogit: 8)

    init(features: [String], w0: [Double], lr: Double, l2: Double, clipLogit: Double) {
        self.features = features; self.w0 = w0; self.lr = lr; self.l2 = l2; self.clipLogit = clipLogit
    }

    static func load(bundle: Bundle = .main) -> LearnerConfig {
        guard let url = bundle.url(forResource: "learner_config", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(LearnerConfig.self, from: data),
              config.w0.count == OnlineLearner.featureNames.count else { return .fallback }
        return config
    }
}

@MainActor final class OnlineLearner: ObservableObject {
    static let shared = OnlineLearner(config: .load(), fileURL: OnlineLearner.defaultFileURL)

    nonisolated static let featureNames = ["bias", "base_logit", "down_1", "down_2", "down_3", "down_4",
                               "ytg", "ytg_known", "field", "form_shotgun", "form_under_center", "form_pistol",
                               "empty", "box", "score", "late"]

    @Published private(set) var stats: LearnerStats

    private struct State: Codable {
        var w: [Double]
        var outcomes: [String: LearnOutcome] = [:]   // key "gameId#playId"
    }

    private let config: LearnerConfig
    private let fileURL: URL
    private var state: State

    init(config: LearnerConfig, fileURL: URL) {
        self.config = config
        self.fileURL = fileURL
        let saved = (try? Data(contentsOf: fileURL)).flatMap { try? JSONDecoder().decode(State.self, from: $0) }
        state = saved?.w.count == config.w0.count ? saved! : State(w: config.w0)
        stats = Self.stats(state.outcomes.values)
    }

    var weights: [Double] { state.w }

    func adjustedPPass(base: PlayPrediction, inputs: PlayInputs) -> Double {
        sigmoid(dot(state.w, Self.features(inputs: inputs, pPass: base.pPass, clipLogit: config.clipLogit)))
    }

    func hasLearned(gameId: String, playId: Int) -> Bool { state.outcomes[Self.key(gameId, playId)] != nil }

    @discardableResult
    func learn(gameId: String, playId: Int, inputs: PlayInputs, base: PlayPrediction, wasPass: Bool) -> LearnOutcome {
        let key = Self.key(gameId, playId)
        if let done = state.outcomes[key] { return done }

        let x = Self.features(inputs: inputs, pPass: base.pPass, clipLogit: config.clipLogit)
        let outcome = LearnOutcome(baseP: base.pPass, adjustedP: step(x: x, y: wasPass ? 1 : 0), wasPass: wasPass)
        state.outcomes[key] = outcome
        stats = Self.stats(state.outcomes.values)
        save()
        return outcome
    }

    /// One prequential SGD step: returns p_adj scored BEFORE the update, then
    /// w <- w - lr * ((p - y) * x + l2 * (w - w0))   (log-loss gradient + pull back toward w0).
    @discardableResult
    func step(x: [Double], y: Double) -> Double {
        let p = sigmoid(dot(state.w, x))
        for k in state.w.indices {
            state.w[k] -= config.lr * ((p - y) * x[k] + config.l2 * (state.w[k] - config.w0[k]))
        }
        return p
    }

    func reset() {
        state = State(w: config.w0)
        stats = Self.stats([])
        try? FileManager.default.removeItem(at: fileURL)
    }

    // MARK: Features (order = featureNames; must match Modeler's Python)

    nonisolated static func features(inputs i: PlayInputs, pPass: Double, clipLogit: Double) -> [Double] {
        func flag(_ b: Bool) -> Double { b ? 1 : 0 }
        let logit = log(pPass / (1 - pPass))   // ±inf at 0/1 is fine: the clip catches it
        var x = [1, min(max(logit, -clipLogit), clipLogit)]
        x += (1...4).map { flag(i.down == $0) }
        x += [i.ydstogo.map { Double(min($0, 20)) / 10 } ?? 0, flag(i.ydstogo != nil)]
        x += [i.yardline100.map { Double($0) / 100 } ?? 0]
        x += [FormationFamily.shotgun, .underCenter, .pistol].map { flag(i.formationFamily == $0) }
        x += [flag(i.isEmpty == true)]
        x += [i.box.map { (Double($0) - 6.5) / 2 } ?? 0]
        x += [i.scoreDifferential.map { Double(min(max($0, -21), 21)) / 14 } ?? 0]
        x += [flag((i.qtr ?? 0) >= 4)]
        return x
    }

    // MARK: Helpers

    private static let defaultFileURL: URL = {
        let dir = (try? FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                 appropriateFor: nil, create: true)) ?? FileManager.default.temporaryDirectory
        return dir.appendingPathComponent("online_learner.json")
    }()

    private static func key(_ gameId: String, _ playId: Int) -> String { "\(gameId)#\(playId)" }

    private func save() {
        // Losing this file only loses learning progress, so a failed write is ignored.
        try? JSONEncoder().encode(state).write(to: fileURL, options: .atomic)
    }

    private static func stats<C: Collection>(_ outcomes: C) -> LearnerStats where C.Element == LearnOutcome {
        let n = Double(max(outcomes.count, 1))
        func acc(_ correct: (LearnOutcome) -> Bool) -> Double { Double(outcomes.filter(correct).count) / n }
        func logLoss(_ p: (LearnOutcome) -> Double) -> Double {
            outcomes.reduce(0) { sum, o in
                let q = min(max(p(o), 1e-15), 1 - 1e-15)
                return sum - log(o.wasPass ? q : 1 - q)
            } / n
        }
        return LearnerStats(learned: outcomes.count,
                            baseAccuracy: acc(\.baseCorrect), adjustedAccuracy: acc(\.adjustedCorrect),
                            baseLogLoss: logLoss(\.baseP), adjustedLogLoss: logLoss(\.adjustedP))
    }
}

private func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
private func sigmoid(_ z: Double) -> Double { 1 / (1 + exp(-z)) }

// MARK: - Why did it miss?

/// How much one group of inputs moved the base p_pass: p(all inputs) - p(this group left blank).
struct Factor: Identifiable { let id = UUID(); let name: String; let delta: Double }

enum MissExplainer {
    /// Clears one optional input; returns whether it was known.
    private static func clear<T>(_ kp: WritableKeyPath<PlayInputs, T?>) -> (inout PlayInputs) -> Bool {
        { p in defer { p[keyPath: kp] = nil }; return p[keyPath: kp] != nil }
    }

    private static let groups: [(String, [(inout PlayInputs) -> Bool])] = [
        ("Down & distance", [clear(\.down), clear(\.ydstogo)]),
        ("Field position", [clear(\.yardline100)]),
        ("Scoreboard", [clear(\.qtr), clear(\.quarterSecondsRemaining), clear(\.scoreDifferential),
                        clear(\.posteamTimeouts), clear(\.defteamTimeouts)]),
        ("Formation", [clear(\.formationFamily), clear(\.isEmpty), clear(\.backfieldCount), clear(\.noHuddle)]),
        ("Offense personnel", [clear(\.offOL), clear(\.offRB), clear(\.offTE), clear(\.offWR)]),
        ("Defense personnel", [clear(\.defDL), clear(\.defLB), clear(\.defDB)]),
        ("Box", [clear(\.box)]),
        ("Hash", [clear(\.hash)]),
    ]

    static func factors(predictor: PlayPredictor, inputs: PlayInputs) -> [Factor] {
        guard let full = try? predictor.predict(inputs).pPass else { return [] }
        return groups.compactMap { name, clears -> Factor? in
            var blank = inputs
            let known = clears.map { $0(&blank) }   // map, not contains: every field must be cleared
            guard known.contains(true), let p = try? predictor.predict(blank).pPass else { return nil }
            return Factor(name: name, delta: full - p)
        }
        .sorted { abs($0.delta) > abs($1.delta) }
    }
}
