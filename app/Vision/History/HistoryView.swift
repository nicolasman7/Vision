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
    @State private var outcome: LearnOutcome?   // set when the current play was revealed (and learned) in this visit
    @State private var confirmReset = false
    @ObservedObject private var learner = OnlineLearner.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    ScreenTitle("History")
                    if let loadError {
                        ContentUnavailableView("Play history not available", systemImage: "externaldrive.badge.xmark",
                                               description: Text(loadError))
                    } else {
                        learningCard
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

    // MARK: Learning card

    private var learningCard: some View {
        let s = learner.stats
        return VStack(spacing: 8) {
            SectionLabel("Learning")
            VStack(alignment: .leading, spacing: 6) {
                if s.learned == 0 {
                    Text("Reveal unseen plays to teach the model.").font(.headline)
                    Text("Each guess is scored before it learns, so the numbers stay honest.")
                        .font(.footnote).foregroundStyle(.secondary)
                } else {
                    Text("Learned from \(s.learned) play\(s.learned == 1 ? "" : "s") · Base \(Self.pct1(s.baseAccuracy)) → Adjusted \(Self.pct1(s.adjustedAccuracy))")
                        .font(.headline).monospacedDigit()
                    Text(String(format: "Log loss %.3f → %.3f (lower is better)", s.baseLogLoss, s.adjustedLogLoss))
                        .font(.footnote).foregroundStyle(.secondary).monospacedDigit()
                    Button("Reset learning", role: .destructive) { confirmReset = true }
                        .font(.footnote.weight(.semibold))
                        .confirmationDialog("Forget everything learned from revealed plays?",
                                            isPresented: $confirmReset, titleVisibility: .visible) {
                            Button("Reset learning", role: .destructive) { learner.reset() }
                        }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
    }

    // MARK: Result

    private enum Mode { case training, learned, unseen }

    private func mode(_ play: HistoryPlay) -> Mode {
        if play.inTraining { return .training }
        // A play revealed in this visit stays in "unseen" mode so its before-learning verdicts remain visible.
        if outcome == nil, learner.hasLearned(gameId: play.gameID, playId: play.playID) { return .learned }
        return .unseen
    }

    private func result(_ play: HistoryPlay) -> some View {
        let prediction = try? predictor?.predict(play.inputs)
        let mode = mode(play)
        let shown = mode != .unseen || outcome != nil
        // Before reveal: the learner's live value. After reveal: the value it had before learning this play.
        let adjusted = outcome?.adjustedP ?? (mode == .unseen ? prediction.map { learner.adjustedPPass(base: $0, inputs: play.inputs) } : nil)
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
                    badge(mode)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            }

            VStack(spacing: 8) {
                SectionLabel("Model, before the snap")
                VStack(alignment: .leading, spacing: 10) {
                    FieldDiagramView(prediction: prediction, ydstogo: play.inputs.ydstogo)
                    if let prediction, let adjusted, abs(adjusted - prediction.pPass) >= 0.005 {
                        Label("Adjusted by what it learned: \(Self.call(adjusted))", systemImage: "arrow.triangle.2.circlepath")
                            .font(.footnote.weight(.semibold)).foregroundStyle(.secondary)
                    }
                }
                .padding(18)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            }

            if shown {
                VStack(spacing: 8) {
                    SectionLabel("What actually happened")
                    VStack(alignment: .leading, spacing: 8) {
                        Text(Self.outcome(play)).font(.system(.title3, design: .rounded, weight: .bold))
                        if let prediction, let isPass = play.isPass {
                            verdictLine(mode == .unseen ? "Base" : "Model", prediction.pPass, isPass)
                            if let adjusted, mode == .unseen { verdictLine("Adjusted", adjusted, isPass) }
                        }
                        if let desc = play.desc { Text(desc).font(.callout).foregroundStyle(.secondary) }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                }
                if mode != .training, let predictor, let prediction, let isPass = play.isPass {
                    why(MissExplainer.factors(predictor: predictor, inputs: play.inputs),
                        missed: (prediction.pPass >= 0.5) != isPass)
                }
            } else {
                Button { reveal(play, prediction) } label: {
                    Label("Reveal what happened", systemImage: "eye").frame(maxWidth: .infinity).onAccentLabel()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(prediction == nil || play.isPass == nil)
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

    private func verdictLine(_ who: String, _ pPass: Double, _ isPass: Bool) -> some View {
        let hit = (pPass >= 0.5) == isPass
        return Text("\(hit ? "✓" : "✗") \(who) \(hit ? "called it" : "missed") (\(Self.call(pPass)))")
            .font(.headline).foregroundStyle(hit ? .green : .red)
    }

    /// Top 4 factor groups as signed bars; positive pushes toward pass.
    private func why(_ factors: [Factor], missed: Bool) -> some View {
        let top = Array(factors.sorted { abs($0.delta) > abs($1.delta) }.prefix(4))
        let biggest = max(top.map { abs($0.delta) }.max() ?? 0, 1e-9)
        return VStack(spacing: 8) {
            SectionLabel(missed ? "Why it missed" : "What drove the call")
            VStack(alignment: .leading, spacing: 10) {
                if top.isEmpty { Text("No single input stood out.").foregroundStyle(.secondary) }
                ForEach(top) { f in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(f.name).font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(Self.signedPct(f.delta)).font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
                        }
                        GeometryReader { g in
                            Capsule().fill(f.delta >= 0 ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                                .frame(width: max(4, g.size.width * abs(f.delta) / biggest))
                        }
                        .frame(height: 6)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
    }

    private func badge(_ mode: Mode) -> some View {
        let (text, icon) = switch mode {
        case .training: ("In training data", "books.vertical")
        case .learned: ("Learned from", "checkmark.seal")
        case .unseen: ("Unseen by the model", "sparkles")
        }
        return Label(text, systemImage: icon)
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

    /// "74% pass" / "68% run": the side the probability picks, with its share.
    static func call(_ pPass: Double) -> String {
        "\(Int((max(pPass, 1 - pPass) * 100).rounded()))% \(pPass >= 0.5 ? "pass" : "run")"
    }

    /// 0.7604 -> "76.0%"
    static func pct1(_ v: Double) -> String { String(format: "%.1f%%", v * 100) }

    /// 0.12 -> "+12% toward pass", -0.06 -> "−6%"
    static func signedPct(_ d: Double) -> String {
        let n = Int((abs(d) * 100).rounded())
        return d >= 0 ? "+\(n)% toward pass" : "−\(n)%"
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
            outcome = nil
        }
    }

    /// Show what happened and learn from it, once. Only unseen, not-yet-learned plays get here.
    private func reveal(_ play: HistoryPlay, _ prediction: PlayPrediction?) {
        guard outcome == nil, let prediction, let isPass = play.isPass,
              !learner.hasLearned(gameId: play.gameID, playId: play.playID) else { return }
        withAnimation(.snappy) {
            outcome = learner.learn(gameId: play.gameID, playId: play.playID, inputs: play.inputs,
                                    base: prediction, wasPass: isPass)
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
