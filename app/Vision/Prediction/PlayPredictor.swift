import CoreML
import Foundation

enum PlayPredictorError: Error {
    case missingResource(String)
    case badOutput(String)
}

/// Runs the three Core ML heads (run_pass, direction, depth) on whatever inputs are known.
/// Mirrors pipeline/prediction/predictor.py; parity is checked by VisionTests/PlayPredictorTests.
final class PlayPredictor {
    private struct Schema: Decodable {
        struct OneHot: Decodable { let columns: [String] }
        struct Head: Decodable { let file: String }
        let features: [String]
        let onehot: [String: OneHot]
        let heads: [String: Head]
    }

    private let schema: Schema
    private let runPass: MLModel, direction: MLModel, depth: MLModel

    init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: "predictor_schema", withExtension: "json") else {
            throw PlayPredictorError.missingResource("predictor_schema.json")
        }
        let schema = try JSONDecoder().decode(Schema.self, from: Data(contentsOf: url))
        self.schema = schema

        // Xcode compiles each bundled .mlmodel into a .mlmodelc folder; codegen is off, so load by URL.
        func model(_ head: String) throws -> MLModel {
            let name = ((schema.heads[head]?.file ?? head) as NSString).deletingPathExtension
            guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
                throw PlayPredictorError.missingResource("\(name).mlmodelc")
            }
            return try MLModel(contentsOf: url)
        }
        runPass = try model("run_pass")
        direction = try model("direction")
        depth = try model("depth")
    }

    func predict(_ inputs: PlayInputs) throws -> PlayPrediction {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["features": features(inputs)])

        // run_pass has Int64 class keys; they come back as NSNumber, so match by value, not by AnyHashable.
        let rp = try probs(runPass, provider)
        guard let pPass = rp.first(where: { ($0.key as? NSNumber)?.int64Value == 1 })?.value.doubleValue else {
            throw PlayPredictorError.badOutput("run_pass has no class 1")
        }
        return PlayPrediction(pPass: pPass,
                              direction: try stringProbs(direction, provider),
                              depth: try stringProbs(depth, provider))
    }

    /// PlayInputs -> the feature vector, in the schema's order. Unknown = NaN, never 0.
    private func features(_ i: PlayInputs) throws -> MLMultiArray {
        func d(_ v: Int?) -> Double? { v.map(Double.init) }
        func d(_ v: Bool?) -> Double? { v.map { $0 ? 1 : 0 } }

        var v: [String: Double?] = [
            "down": d(i.down), "ydstogo": d(i.ydstogo), "yardline_100": d(i.yardline100),
            "qtr": d(i.qtr), "quarter_seconds_remaining": d(i.quarterSecondsRemaining),
            "score_differential": d(i.scoreDifferential),
            "posteam_timeouts_remaining": d(i.posteamTimeouts), "defteam_timeouts_remaining": d(i.defteamTimeouts),
            "no_huddle": d(i.noHuddle), "is_empty": d(i.isEmpty), "ftn_n_offense_backfield": d(i.backfieldCount),
            "off_ol": d(i.offOL), "off_rb": d(i.offRB), "off_te": d(i.offTE), "off_wr": d(i.offWR),
            "def_dl": d(i.defDL), "def_lb": d(i.defLB), "def_db": d(i.defDB), "ftn_box": d(i.box),
        ]

        // Derived inputs (schema "derived"): only when everything they depend on is known.
        if let togo = i.ydstogo, let yl = i.yardline100 { v["goal_to_go"] = togo == yl ? 1 : 0 }
        if let q = i.qtr, let qs = i.quarterSecondsRemaining {
            v["half_seconds_remaining"] = Double(qs + ([1, 3].contains(q) ? 900 : 0))
            v["game_seconds_remaining"] = Double(qs + 900 * max(0, 4 - q))
        }

        // One-hot: the known value's column is 1, its siblings 0; unknown leaves every column NaN.
        let categories = ["formation_family": i.formationFamily?.rawValue, "ftn_starting_hash": i.hash?.rawValue]
        for (col, value) in categories {
            guard let value, let columns = schema.onehot[col]?.columns, columns.contains("\(col)=\(value)") else { continue }
            for c in columns { v[c] = c == "\(col)=\(value)" ? 1 : 0 }
        }

        let array = try MLMultiArray(shape: [NSNumber(value: schema.features.count)], dataType: .double)
        for (k, name) in schema.features.enumerated() {
            array[k] = NSNumber(value: (v[name] ?? nil) ?? .nan)
        }
        return array
    }

    private func probs(_ model: MLModel, _ provider: MLFeatureProvider) throws -> [AnyHashable: NSNumber] {
        guard let p = try model.prediction(from: provider).featureValue(for: "probs")?.dictionaryValue else {
            throw PlayPredictorError.badOutput("no probs output")
        }
        return p
    }

    private func stringProbs(_ model: MLModel, _ provider: MLFeatureProvider) throws -> [String: Double] {
        var out: [String: Double] = [:]
        for (k, p) in try probs(model, provider) {
            guard let key = k as? String else { throw PlayPredictorError.badOutput("non-string class \(k)") }
            out[key] = p.doubleValue
        }
        return out
    }
}
