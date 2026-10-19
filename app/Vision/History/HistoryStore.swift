import Foundation
import SQLite3

/// One game in the bundled history database.
struct HistoryGame: Identifiable, Hashable {
    let id: String              // nflverse game_id, e.g. "2024_06_KC_LV"
    let season: Int, week: Int
    let home: String, away: String
    let date: String?           // "YYYY-MM-DD"

    /// "KC @ LV · Oct 13"
    var label: String {
        let matchup = "\(away) @ \(home)"
        guard let date, let d = Self.iso.date(from: date) else { return matchup }
        return "\(matchup) · \(Self.short.string(from: d))"
    }

    private static let iso: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    private static let short: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "MMM d"; return f
    }()
}

/// One real play: what the model could see before the snap (`inputs`) and what actually happened.
struct HistoryPlay {
    let gameID: String, playID: Int
    let posteam: String?, defteam: String?
    let inputs: PlayInputs      // qtr and clock live here too
    let isPass: Bool?
    let direction: String?      // "left" / "middle" / "right" (runs)
    let airYards: Double?
    let depthBucket: String?    // "<=0" / "1-9" / "10+" (throws)
    let desc: String?
    let inTraining: Bool        // season ≤ 2023
}

enum HistoryStoreError: Error { case missing, open(String) }

/// Read-only access to history.sqlite (built by scripts on the Python side). Plain SQLite3 C API, no packages.
final class HistoryStore {
    private var db: OpaquePointer?

    convenience init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: "history", withExtension: "sqlite") else { throw HistoryStoreError.missing }
        try self.init(path: url.path)
    }

    init(path: String) throws {
        guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            let msg = db.map { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            sqlite3_close(db)
            throw HistoryStoreError.open(msg)
        }
    }

    deinit { sqlite3_close(db) }

    // MARK: Queries

    func seasons() -> [Int] {
        query("SELECT DISTINCT season FROM games ORDER BY season DESC").compactMap { $0.int("season") }
    }

    func weeks(season: Int) -> [Int] {
        query("SELECT DISTINCT week FROM games WHERE season = ? ORDER BY week", [season]).compactMap { $0.int("week") }
    }

    func teams(season: Int) -> [String] {
        query("SELECT home AS t FROM games WHERE season = ? UNION SELECT away FROM games WHERE season = ? ORDER BY t",
              [season, season]).compactMap { $0.string("t") }
    }

    /// Games in a season, narrowed by week and/or team (nil = any).
    func games(season: Int, week: Int? = nil, team: String? = nil) -> [HistoryGame] {
        query("""
            SELECT * FROM games WHERE season = ?1 AND (?2 IS NULL OR week = ?2)
              AND (?3 IS NULL OR home = ?3 OR away = ?3)
            ORDER BY week, game_date, game_id
            """, [season, week, team]).compactMap { r in
            guard let id = r.string("game_id"), let s = r.int("season"), let w = r.int("week"),
                  let home = r.string("home"), let away = r.string("away") else { return nil }
            return HistoryGame(id: id, season: s, week: w, home: home, away: away, date: r.string("game_date"))
        }
    }

    /// The play at or just before a moment: the last play of that quarter snapped with at least `clock` seconds
    /// left. Ordered by play_id, not clock, so odd clock values in the data can't reorder the game.
    /// Before the quarter's first snap (e.g. 15:00), falls back to that first snap.
    func play(gameID: String, qtr: Int, clock: Int) -> HistoryPlay? {
        playRow("WHERE game_id = ? AND qtr = ? AND clock_s >= ? ORDER BY play_id DESC", [gameID, qtr, clock])
            ?? playRow("WHERE game_id = ? AND qtr = ? ORDER BY play_id", [gameID, qtr])
    }

    func play(gameID: String, playID: Int) -> HistoryPlay? {
        playRow("WHERE game_id = ? AND play_id = ?", [gameID, playID])
    }

    func previous(_ p: HistoryPlay) -> HistoryPlay? {
        playRow("WHERE game_id = ? AND play_id < ? ORDER BY play_id DESC", [p.gameID, p.playID])
    }

    func next(_ p: HistoryPlay) -> HistoryPlay? {
        playRow("WHERE game_id = ? AND play_id > ? ORDER BY play_id", [p.gameID, p.playID])
    }

    // MARK: Row -> model

    private func playRow(_ clause: String, _ args: [Any?]) -> HistoryPlay? {
        guard let r = query("SELECT * FROM plays \(clause) LIMIT 1", args).first,
              let gameID = r.string("game_id"), let playID = r.int("play_id") else { return nil }
        let inputs = PlayInputs(
            down: r.int("down"), ydstogo: r.int("ydstogo"), yardline100: r.int("yardline_100"),
            qtr: r.int("qtr"), quarterSecondsRemaining: r.int("clock_s"),
            scoreDifferential: r.int("score_differential"),
            posteamTimeouts: r.int("posteam_timeouts"), defteamTimeouts: r.int("defteam_timeouts"),
            noHuddle: r.bool("no_huddle"),
            formationFamily: r.string("formation_family").flatMap(FormationFamily.init(rawValue:)),
            isEmpty: r.bool("is_empty"), backfieldCount: r.int("backfield"),
            offOL: r.int("off_ol"), offRB: r.int("off_rb"), offTE: r.int("off_te"), offWR: r.int("off_wr"),
            defDL: r.int("def_dl"), defLB: r.int("def_lb"), defDB: r.int("def_db"),
            box: r.int("box"), hash: r.string("hash").flatMap(HashMark.init(rawValue:)))
        return HistoryPlay(gameID: gameID, playID: playID, posteam: r.string("posteam"), defteam: r.string("defteam"),
                           inputs: inputs, isPass: r.bool("is_pass"), direction: r.string("direction"),
                           airYards: r.double("air_yards"), depthBucket: r.string("depth_bucket"),
                           desc: r.string("desc"), inTraining: r.bool("in_training") ?? false)
    }

    // MARK: SQLite plumbing

    /// A result row keyed by column name. NULL columns are simply absent, so every getter returns nil for them.
    private struct Row {
        var values: [String: Any] = [:]
        func int(_ k: String) -> Int? {
            switch values[k] { case let v as Int64: Int(v); case let v as Double: Int(v.rounded()); default: nil }
        }
        func double(_ k: String) -> Double? {
            switch values[k] { case let v as Double: v; case let v as Int64: Double(v); default: nil }
        }
        func bool(_ k: String) -> Bool? { int(k).map { $0 != 0 } }
        func string(_ k: String) -> String? { values[k] as? String }
    }

    private func query(_ sql: String, _ args: [Any?] = []) -> [Row] {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }

        // SQLITE_TRANSIENT: make SQLite copy the string, since the Swift buffer is temporary.
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        for (i, arg) in args.enumerated() {
            let idx = Int32(i + 1)
            switch arg {
            case let v as Int: sqlite3_bind_int64(stmt, idx, Int64(v))
            case let v as String: sqlite3_bind_text(stmt, idx, v, -1, transient)
            default: sqlite3_bind_null(stmt, idx)
            }
        }

        var rows: [Row] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            var row = Row()
            for c in 0..<sqlite3_column_count(stmt) {
                let name = String(cString: sqlite3_column_name(stmt, c))
                switch sqlite3_column_type(stmt, c) {
                case SQLITE_INTEGER: row.values[name] = sqlite3_column_int64(stmt, c)
                case SQLITE_FLOAT: row.values[name] = sqlite3_column_double(stmt, c)
                case SQLITE_TEXT: row.values[name] = String(cString: sqlite3_column_text(stmt, c))
                default: break   // NULL -> unknown
                }
            }
            rows.append(row)
        }
        return rows
    }
}
