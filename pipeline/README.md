# pipeline/

The Python ML pipeline. Each sub-folder is one stage. Data flows through them in this order:

1. **view_classification/**: Works out what kind of shot a frame is (All-22, broadcast, sideline, end zone, phone filming a TV, stands). This tells later stages what to expect and how much to trust them.
2. **detection/**: YOLO model that draws boxes around players and the ball.
3. **tracking/**: ByteTrack links boxes across frames so each player keeps the same ID.
4. **field_mapping/**: Finds yard lines and hash marks, computes a homography with OpenCV, and converts pixel positions to field coordinates in yards.
5. **formation/**: Reads the pre-snap alignment from field positions and identifies the formation.
6. **prediction/**: Predicts the play (run/pass, direction, depth) with a confidence score. Trained on NFL Big Data Bowl tracking data and nflverse play-by-play.
7. **export/**: Converts trained models to Core ML with coremltools so the app can use them.

Stages 1–4 turn video into player positions. Stages 5–6 only see those positions and never touch pixels.
