# Detection datasets for American football

_Research by Film Scout, 2026-10-17. Options with evidence only. The user decides._

**What this is for:** the `detection` stage needs a YOLO model that draws boxes around players (and ideally the ball and referees) in football video. YOLO learns from labeled examples: images where a person has already drawn the boxes. This page lists where those labeled examples could come from.

**How access was checked:** "no login" claims come from a header-only request (`curl -sIL -o /dev/null -w "%{http_code}"`) on the actual file. No data was downloaded and no accounts were used.

## The short version

Public, labeled **American football** detection data is rare. One 2025 paper says there was no public American football tracking dataset before theirs ([arXiv 2511.09455](https://arxiv.org/abs/2511.09455)). Every football-specific set below needs an account. The only things you can get with no login are general-purpose datasets and models that were not built for football.

## Football-specific datasets

### 1. NFL 1st and Future: Impact Detection (Kaggle, 2020–21)
- **URL:** https://www.kaggle.com/competitions/nfl-impact-detection
- **What:** **helmet** boxes only, not full-body players and not the ball. 9,947 labeled still images (4,958 sideline, 4,989 end zone) with 193,736 helmet boxes. There are also videos with helmet boxes on every frame ([AWS write-up](https://aws.amazon.com/blogs/machine-learning/helmet-detection-error-analysis-in-football-videos-using-amazon-sagemaker)).
- **Classes:** Helmet, Helmet-Blurred, Helmet-Sideline, Helmet-Partial, Helmet-Difficult.
- **Format:** CSV (`image_labels.csv`: image, left, top, width, height, label). Converting it to YOLO format takes a few lines of code.
- **Views:** NFL sideline and end zone cameras (tight, high resolution). No broadcast, no All-22.
- **License:** Kaggle competition rules, not an open license.
- **Access:** needs account (Kaggle). The competition page returned 200; the data needs a login and accepting the rules.

### 2. NFL Health & Safety: Helmet Assignment (Kaggle, 2021)
- **URL:** https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment
- **What:** 120 training videos (60 plays from 50 games, each filmed from sideline and end zone), about 52k frames. Includes helmet boxes, which player each helmet belongs to, and NGS tracking data for the same plays ([arXiv 2205.09878](https://arxiv.org/abs/2205.09878)). It also reuses the image set from #1.
- **Format:** CSV. **Views:** sideline and end zone. **License:** Kaggle competition rules.
- **Why it matters for us:** it pairs **video with field-coordinate tracking**, which could help check the `field_mapping` stage later.
- **Access:** needs account (Kaggle).

### 3. NFL 1st and Future: Player Contact Detection (Kaggle, 2022–23)
- **URL:** https://www.kaggle.com/competitions/nfl-player-contact-detection
- **What:** four videos per play (Sideline and Endzone synced, plus an **All-29** wide view that is not guaranteed to be synced), tracking data, contact labels, and baseline helmet boxes ([NFL announcement](https://www.nfl.com/playerhealthandsafety/equipment-and-innovation/innovation-challenges/nfl-enters-next-frontier-of-predicting-injury-with-contact-detection-challenge)).
- **Note:** it is the only listed source with an All-22-style view. The boxes are helmets, not bodies.
- **Access:** needs account (Kaggle). I did not verify frame counts or file sizes.

### 4. AFMOT: Hand Held Multi-Object Tracking in American Football (2025)
- **URL:** https://huggingface.co/datasets/rinost081/AFMOT ([paper](https://arxiv.org/abs/2511.09455), [project page](https://rinost081.github.io/MOTAF_page))
- **What:** 11,411 frames, 188,711 **full-body** boxes, 505 track IDs, 20 videos (10 plays × 2 views). Classes are players and referees; the paper lists no ball class.
- **Format:** MOT Challenge layout (`gt/gt.txt` plus `img1/*.png` per sequence, seen in the file listing).
- **Views:** handheld cameras at university games. "Wide" shows the whole field and is the closest thing here to All-22 or a view from the stands; "Tight" follows the ball.
- **License:** CC BY-NC-SA 4.0, academic non-commercial use only, no redistribution.
- **Access:** needs account (Hugging Face) **and** an access request (name, institutional email, purpose). Checked with a header request: `gt.txt` returned **HTTP 401**. This set is "on request" as well as behind a login.
- **Why it matters:** it is the closest match to what we need: full-body players and refs, tracking IDs, and a wide angle.

### 5. Roboflow Universe community sets
Small sets uploaded by individual users. Downloading needs a Roboflow account (API key). The pages are behind a Cloudflare check, so I could only read search-result snippets. **Image counts, views, and label quality are not verified.**

| Dataset | Classes (from snippets) | Size | License |
|---|---|---|---|
| [darshana/nfl-player-detection](https://universe.roboflow.com/darshana/nfl-player-detection-06af8) | Ball, Player, Referee | not shown | CC BY 4.0 |
| [fbtrainingmodels/nfl-final](https://universe.roboflow.com/fbtrainingmodels/nfl-final) | player, referee | 2,044 images | CC BY 4.0 |
| [rizan/nfl-9wqrs](https://universe.roboflow.com/rizan/nfl-9wqrs) | Ball, player, player_with_ball | ~25 images | CC BY 4.0 |
| [rizan/keypointsnfl](https://universe.roboflow.com/rizan/keypointsnfl) | field keypoints (meaning not stated) | ~9 images | CC BY 4.0 |

- **Format:** Roboflow exports to YOLO, COCO, and other formats. **Access:** needs account (Roboflow).

### 6. Field lines and keypoints
I found **no real public American football field-line or hash-mark dataset**. The only hit is `rizan/keypointsnfl` (about 9 images). For soccer, **SoccerNet** has camera-calibration and pitch-line labels, but the pitch markings are different and the videos need an NDA password ([SoccerNet on PyPI](https://www.pypi.org/project/SoccerNet/)). Access: needs account/NDA (SoccerNet). A 2013 CVPR workshop paper built 800+ formation clips and promised to release them ([Atmosukarto et al.](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W19/html/Atmosukarto_Automatic_Recognition_of_2013_CVPR_paper.html)), but I found no download link. Treat that one as on request at best.

### Not detection, but related
- **Big Data Bowl** (e.g. https://www.kaggle.com/competitions/nfl-big-data-bowl-2025): tracking data in field coordinates, **no video**. It is for the `prediction` stage. Access: needs account (Kaggle).

## General-purpose sources (no login)

### 7. Zero-label fallback: COCO-pretrained YOLO
- **What:** Ultralytics YOLO11 weights trained on COCO already detect `person` (class 0) and `sports ball` (class 32). No labeling needed.
- **Access:** no login, verified **HTTP 200** on the actual files: [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) and [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt).
- **License:** AGPL-3.0. Fine for personal use. Shipping it in a distributed app would mean open-sourcing the app or buying a license.
- **What it probably can't do:**
  - **Tiny players in All-22.** A player in a 1080p All-22 frame is roughly 20–40 px tall. YOLO shrinks the image to 640 px, so the player ends up only about 12–25 px tall. COCO models are weakest on objects that small, so expect missed players, especially in piles at the line. Cutting the frame into tiles helps but costs speed.
  - **The football.** COCO's `sports ball` is mostly round balls. A brown oval football that is small, blurred, and usually hidden in someone's hands will rarely be found.
  - **Offense vs. defense.** `person` is the only class it has. Telling teams apart needs an extra step, such as clustering jersey colors from each box.
  - **Referees, coaches, and fans.** All of them are also `person`. They need filtering, for example "inside the field mask", plus a check for striped shirts.
- **Best use:** a day-one baseline, and a pre-labeler that speeds up hand-labeling (#9).

### 8. Open Images V7
- **What:** 1.9M general photos with 16M boxes over 600 classes, including `Football helmet` and `Football` ([Ultralytics docs](https://docs.ultralytics.com/datasets/detect/open-images-v7)). These are mostly Flickr photos, not game film. I did not verify whether `Football` means an American football or a soccer ball, or how many boxes each class has.
- **License:** annotations CC BY 4.0; images are listed as CC BY 2.0.
- **Access:** no login, verified **HTTP 200** on the actual file [oidv6-train-annotations-bbox.csv](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) (also 200 on [oidv7-class-descriptions-boxable.csv](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv)).

## 9. Hand-labeling route
- **Tools that need no account:** [CVAT](https://github.com/cvat-ai/cvat) (self-hosted in Docker) or [Label Studio](https://github.com/HumanSignal/label-studio) (`pip install`, runs locally). Both export YOLO format.
- **Work per frame:** about 22 players, 3–7 officials, and 1 ball, so roughly 30 boxes.
- **Rough hours for 500 frames:**
  - **From scratch** at about 5–8 s per box (2.5–4 min per frame): about **20–35 hours**.
  - **Pre-labeled with COCO YOLO (#7), then corrected** (about 1–1.5 min per frame): about **8–12 hours**.
  - Adding an offense/defense label to each box adds about 20–30%.
- **Tips:** take frames from many games, views, and lighting conditions, not consecutive frames. Include phone-at-TV shots, since no public dataset has them. These are your own frames, so they stay out of git (copyright).
- **License:** yours, for personal use. **Access:** no account needed, local only.

## Comparison

| # | Source | Boxes on | Views | Size | Access (HTTP code) |
|---|---|---|---|---|---|
| 1 | Kaggle Impact Detection | helmets | sideline, end zone | 9,947 imgs + videos | needs account (Kaggle) |
| 2 | Kaggle Helmet Assignment | helmets + tracking | sideline, end zone | 120 videos, ~52k frames | needs account (Kaggle) |
| 3 | Kaggle Contact Detection | helmets + tracking | sideline, end zone, All-29 | not verified | needs account (Kaggle) |
| 4 | AFMOT | players, refs (bodies) | handheld wide + tight (college) | 11,411 frames | needs account (HF) + on request (401) |
| 5 | Roboflow NFL sets | player, ref, ball | not verified | 25–2,044 imgs | needs account (Roboflow) |
| 6 | Field lines (keypointsnfl / SoccerNet) | field points | n/a | ~9 imgs / soccer | needs account (Roboflow / SoccerNet NDA) |
| 7 | COCO YOLO11 (zero-label) | person, sports ball | any (weak on small) | pretrained | no login, verified HTTP 200 on the actual file |
| 8 | Open Images V7 | football helmet, football | general photos | 1.9M imgs total | no login, verified HTTP 200 on the actual file |
| 9 | Hand-label ~500 frames | whatever you choose | whatever you film | 500 frames | local, no account |

## Recommendation (you can overrule this)

1. **Start with #7 (COCO YOLO11, zero labels)** to get the pipeline running end to end today, with no accounts. Measure where it fails (All-22 small players, the ball).
2. **Then hand-label about 500 of your own frames (#9), pre-labeled by #7.** This takes about 8–12 hours. It is the only option that covers *your* views (broadcast, phone-at-TV, stands) and the classes you want (player, ref, ball, maybe offense/defense). Fine-tune YOLO on it.
3. **If you're willing to make accounts,** AFMOT (#4) is the most useful extra data (full-body boxes, wide view, tracking IDs), but it is gated and non-commercial. A Kaggle account unlocks #1–3, which mostly help with helmets and with checking field mapping, not body detection.
4. **Field lines:** no usable public dataset exists. Plan on classic OpenCV line detection, or label your own keypoints.

## Could not verify
- Roboflow image counts and views (pages blocked by Cloudflare; snippets only).
- Contact Detection dataset size; Kaggle licenses beyond "competition rules".
- What Open Images `Football` means and how many boxes each class has.
- SoccerNet: which calibration files, if any, can be downloaded without the NDA.
