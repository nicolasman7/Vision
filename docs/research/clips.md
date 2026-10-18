# Openly licensed football clips

_Film Scout, 2026-10-18. Manifest: `docs/clips_manifest.json`. Rebuild: `uv run python -m scripts.fetch_clips`._

**On disk now** (`data/raw/clips/`, git-ignored, never committed): 3 high-school clips, about 53 MB, each 27–30 s from huddle break through the offense setting and the snap.

| id | angle | res | license |
|---|---|---|---|
| belmont_woburn_2023_q2_snap | sideline_elevated | 1080p | CC0 1.0 |
| belmont_woburn_2023_q3_snap | sideline_elevated | 1080p | CC0 1.0 |
| wincam_winchester_2021_q2_snap | high_wide (nearest to All-22) | 480p | Public Domain Mark |

**Angles still missing:** a true end zone view, true All-22 (a fixed coaches' camera), and phone-at-TV (no open source exists, so the user will have to film it).

**Sources tried:**
- **Internet Archive:** works with no login; `archive.org/metadata/<id>` returned 200. The Belmont Media Center collection has about 70 CC0 or Public Domain Mark games. Many other high-school games there are CC BY-NC-ND, which I skipped. NFL uploads tagged CC BY were skipped as NFL broadcasts.
- **Wikimedia Commons:** the API returned 200. Its upload server returned **HTTP 429 (Retry-After 600)** after OpenCV seeked over HTTP, so the script downloads short Commons files once instead of seeking. Candidates are checked but not used yet:
  - Milton TD pass and Baker kickoff (ground/stands, CC BY-SA)
  - Army–Navy 1930 (public domain)
  - SMU–Louisville 2025 (CC BY-SA 4.0)
  - Chiefs fan clips (CC BY-SA 2.0)
- **Pexels, Pixabay, DVIDS:** HTTP 403, bot protection, or an API key is required, so I skipped them.

**License obligations:**
- CC0 and Public Domain Mark ask nothing; the manifest keeps credit anyway.
- Any CC BY or BY-SA clip added later needs the manifest's `attribution_text` shown wherever it is used.
- Personal use only. Never commit the video or frames.
- **Doubt:** the PD/CC0 marks were applied by the *uploader*. For the WinCam game the uploader (Belmont Media Center) is not the producer (WinCam).

**For later:** Ultralytics YOLO weights are AGPL-3.0. That's fine for personal use, but revisit it before distributing the app.
