import SwiftUI

/// History tab: pick a real NFL play by game and moment, see what the model would have said before the snap,
/// and compare it with what actually happened.
struct HistoryView: View {
    @State private var store: HistoryStore?
    @State private var predictor: PlayPredictor?
    @State private var loadError: String?

    @State private var seasons: [Int] = []
    @State private var season = 0
    @State private var teams: [String] = []
    @State private var team: String?          // nil = any team
    @State private var weeks: [Int] = []
    @State private var week: Int?             // nil = all weeks
    @State private var games: [HistoryGame] = []
    @State private var gameID: String?
    @State private var qtr = 1
    @State private var time = "15:00"

    @State private var play: HistoryPlay?
    @State private var notFound = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    ScreenTitle("History")
                    if let loadError {
                        ContentUnavailableView("Play history not available", systemImage: "externaldrive.badge.xmark",
                                               description: Text(loadError))
                    } else {
                        finder
                        if let play { result(play) }
                        else if notFound {
                            ContentUnavailableView("No play found", systemImage: "magnifyingglass",
                                                   description: Text("No snaps in that quarter of this game."))
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 24)
            }
            .background(Color(.systemGroupedBackground))
            .toolbar(.hidden, for: .navigationBar)
        }
        .onAppear(perform: load)
    }

    // MARK: Finder

    private var finder: some View {
        VStack(spacing: 8) {
            SectionLabel("Find a play")
            VStack(spacing: 12) {
                row("Season") {
                    Picker("Season", selection: $season) { ForEach(seasons, id: \.self) { Text(String($0)).tag($0) } }
                }
                row("Team") {
                    Picker("Team", selection: $team) {
                        Text("Any team").tag(String?.none)
                        ForEach(teams, id: \.self) { Text($0).tag(String?($0)) }
                    }
                }
                row("Week") {
                    Picker("Week", selection: $week) {
                        Text("All weeks").tag(Int?.none)
                        ForEach(weeks, id: \.self) { Text("Week \($0)").tag(Int?($0)) }
                    }
                }
                row("Game") {
                    Picker("Game", selection: $gameID) {
                        if games.isEmpty { Text("No games").tag(String?.none) }
                        ForEach(games) { Text($0.label).tag(String?($0.id)) }
                    }
                }
                Picker("Quarter", selection: $qtr) {
                    ForEach(1...5, id: \.self) { Text($0 == 5 ? "OT" : "Q\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                row("Time left") {
                    TextField("MM:SS", text: $time)
                        .keyboardType(.numbersAndPunctuation)
                        .multilineTextAlignment(.trailing)
                        .font(.body.monospacedDigit())
                        .frame(maxWidth: 90)
                }
                Button(action: find) {
                    Label("Find play", systemImage: "magnifyingglass").frame(maxWidth: .infinity).onAccentLabel()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(gameID == nil || Self.seconds(time) == nil)
            }
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
        .onChange(of: season) { refreshFilters() }
        .onChange(of: team) { refreshGames() }
        .onChange(of: week) { refreshGames() }
    }

    private func row(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        HStack {
            Text(title).foregroundStyle(.secondary)
            Spacer()
            content()
        }
    }

    // MARK: Result

    private func result(_ play: HistoryPlay) -> some View {
        let prediction = try? predictor?.predict(play.inputs)
        return VStack(spacing: 22) {
            VStack(spacing: 8) {
                SectionLabel("Situation")
                VStack(alignment: .leading, spacing: 6) {
                    Text(Self.situation(play))
                        .font(.system(.title3, design: .rounded, weight: .bold)).monospacedDigit()
                    let lines = [play.inputs.offenseText, play.inputs.defenseText,
                                 play.inputs.hash.map { "\($0.label) hash" }].compactMap { $0 }
                    Text(lines.isEmpty ? "No formation data" : lines.joined(separator: "\n"))
                        .font(.subheadline).foregroundStyle(.secondary)
                    badge(play.inTraining)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            }

            VStack(spacing: 8) {
                SectionLabel("Model, before the snap")
                FieldDiagramView(prediction: prediction, ydstogo: play.inputs.ydstogo)
                    .padding(18)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            }

            VStack(spacing: 8) {
                SectionLabel("What actually happened")
                VStack(alignment: .leading, spacing: 8) {
                    Text(Self.outcome(play)).font(.system(.title3, design: .rounded, weight: .bold))
                    if let verdict = Self.verdict(prediction, play) {
                        Text(verdict.text).font(.headline).foregroundStyle(verdict.hit ? .green : .red)
                    }
                    if let desc = play.desc { Text(desc).font(.callout).foregroundStyle(.secondary) }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            }

            HStack {
                Button { step(store?.previous(play)) } label: { Label("Prev", systemImage: "chevron.left") }
                    .disabled(store?.previous(play) == nil)
                Spacer()
                Button { step(store?.next(play)) } label: {
                    Label("Next", systemImage: "chevron.right").labelStyle(TrailingIcon())
                }
                .disabled(store?.next(play) == nil)
            }
            .buttonStyle(.bordered)
        }
    }

    private func badge(_ inTraining: Bool) -> some View {
        Label(inTraining ? "In training data" : "Unseen by the model",
              systemImage: inTraining ? "books.vertical" : "sparkles")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(.tint.opacity(0.15), in: Capsule())
    }

    // MARK: Text

    /// "Q3 8:42 · 3rd & 6 · KC ball at LV 34"
    static func situation(_ p: HistoryPlay) -> String {
        let i = p.inputs
        var parts: [String] = []
        if let q = i.qtr {
            let clock = i.quarterSecondsRemaining.map { String(format: " %d:%02d", $0 / 60, $0 % 60) } ?? ""
            parts.append((q >= 5 ? "OT" : "Q\(q)") + clock)
        }
        if let d = i.down, let togo = i.ydstogo {
            parts.append("\(PlayInputs.ordinal(d)) & \(togo == i.yardline100 ? "Goal" : String(togo))")
        }
        if let yl = i.yardline100 {
            let spot = yl == 50 ? "the 50" : yl < 50 ? "\(p.defteam ?? "OPP") \(yl)" : "\(p.posteam ?? "OWN") \(100 - yl)"
            parts.append("\(p.posteam ?? "Offense") ball at \(spot)")
        }
        return parts.joined(separator: " · ")
    }

    /// "Pass · 10+ (14 air yds)" or "Run · left".
    static func outcome(_ p: HistoryPlay) -> String {
        guard let isPass = p.isPass else { return "Unknown" }
        guard isPass else { return ["Run", p.direction].compactMap { $0 }.joined(separator: " · ") }
        let detail = p.depthBucket.map { b in b + (p.airYards.map { " (\(Int($0)) air yds)" } ?? "") }
        return ["Pass", detail ?? "no throw"].joined(separator: " · ")
    }

    /// Run/pass call vs. reality; the percentage shown is the side the model picked.
    static func verdict(_ pred: PlayPrediction?, _ p: HistoryPlay) -> (text: String, hit: Bool)? {
        guard let pred, let isPass = p.isPass else { return nil }
        let callsPass = pred.pPass >= 0.5
        let pct = Int((max(pred.pPass, 1 - pred.pPass) * 100).rounded())
        let call = "\(pct)% \(callsPass ? "pass" : "run")"
        return callsPass == isPass ? ("✓ Model called it (\(call))", true) : ("✗ Model missed (\(call))", false)
    }

    /// "8:42" -> 522; also "8" -> 480. Capped at a full quarter.
    static func seconds(_ text: String) -> Int? {
        let parts = text.split(separator: ":").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard (1...2).contains(parts.count), !parts.contains(nil) else { return nil }
        let s = parts[0]! * 60 + (parts.count == 2 ? parts[1]! : 0)
        return (0...900).contains(s) ? s : nil
    }

    // MARK: Actions

    private func load() {
        guard store == nil, loadError == nil else { return }
        do { store = try HistoryStore() } catch {
            loadError = "history.sqlite is missing from the app bundle. Build it on the Python side, "
                + "then run `xcodegen generate` and rebuild."
            return
        }
        predictor = try? PlayPredictor()
        seasons = store?.seasons() ?? []
        season = seasons.first ?? 0
        refreshFilters()
    }

    private func refreshFilters() {
        guard let store else { return }
        teams = store.teams(season: season)
        weeks = store.weeks(season: season)
        if let t = team, !teams.contains(t) { team = nil }
        if let w = week, !weeks.contains(w) { week = nil }
        if week == nil, team == nil { week = weeks.first }   // all 272 games at once is too long a menu
        refreshGames()
    }

    private func refreshGames() {
        games = store?.games(season: season, week: week, team: team) ?? []
        if !games.contains(where: { $0.id == gameID }) { gameID = games.first?.id }
    }

    private func find() {
        guard let gameID, let clock = Self.seconds(time) else { return }
        step(store?.play(gameID: gameID, qtr: qtr, clock: clock))
    }

    private func step(_ next: HistoryPlay?) {
        withAnimation(.snappy) {
            play = next
            notFound = next == nil
        }
    }
}

/// "Next ›": icon after the title.
private struct TrailingIcon: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 6) { configuration.title; configuration.icon }
    }
}

#Preview { HistoryView().appTheme() }
