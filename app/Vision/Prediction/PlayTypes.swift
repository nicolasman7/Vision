// Inputs and outputs of the play predictor. Contract: coreml/predictor_schema.json.

enum FormationFamily: String, CaseIterable { case shotgun, underCenter = "under_center", pistol }
enum HashMark: String, CaseIterable { case left = "L", middle = "M", right = "R" }

struct PlayInputs {            // nil = unknown -> NaN feature (never 0)
    var down: Int?; var ydstogo: Int?; var yardline100: Int?   // yardline100 = yards to opponent end zone
    var qtr: Int?; var quarterSecondsRemaining: Int?
    var scoreDifferential: Int?; var posteamTimeouts: Int?; var defteamTimeouts: Int?
    var noHuddle: Bool?
    var formationFamily: FormationFamily?; var isEmpty: Bool?; var backfieldCount: Int?
    var offOL: Int?; var offRB: Int?; var offTE: Int?; var offWR: Int?
    var defDL: Int?; var defLB: Int?; var defDB: Int?
    var box: Int?; var hash: HashMark?
}

struct PlayPrediction {
    let pPass: Double
    let direction: [String: Double]   // "left","middle","right" — given it's a run
    let depth: [String: Double]       // "<=0","1-9","10+" — given it's a throw
}
