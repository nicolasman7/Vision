"""Export a small COCO YOLO detector to coreml/yolo_detector.mlpackage (the app loads it by that name).

    uv run python -m scripts.export_yolo_coreml

nms=True bakes NMS into the model, so iOS Vision returns VNRecognizedObjectObservation directly.
Then a parity check: person count from Core ML vs PyTorch on one frame of a local clip.
"""
import shutil
import sys
from pathlib import Path

import coremltools as ct
import cv2
from PIL import Image
from ultralytics import YOLO

WEIGHTS = Path("models/yolo/yolo26n.pt")  # YOLO("models/yolo/x.pt") downloads into models/yolo/
OUT = Path("coreml/yolo_detector.mlpackage")
SIZE = 640
CONF = 0.25


def export(half):
    model = YOLO(str(WEIGHTS))
    src = Path(model.export(format="coreml", nms=True, imgsz=SIZE, half=half))
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.move(src, OUT)
    return model


def main():
    try:
        model = export(half=True)
    except Exception as e:  # coremltools is untested with torch 2.14: retry once in fp32
        print(f"half=True export failed: {e!r}; retrying with half=False", file=sys.stderr)
        model = export(half=False)

    ml = ct.models.MLModel(str(OUT))
    spec = ml.get_spec()
    print("inputs:", [i.name for i in spec.description.input])
    print("outputs:", [o.name for o in spec.description.output])
    labels = list(spec.pipeline.models[-1].nonMaximumSuppression.stringClassLabels.vector)
    print("labels:", len(labels), "person" in labels, "sports ball" in labels)
    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"size: {size_mb:.1f} MB")

    # Parity on one frame, squashed to 640x640 for both (Core ML input is a fixed 640x640 image).
    clip = sorted(Path("data/raw/clips").glob("*.mp4"))[0]
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
    ok, frame = cap.read()
    cap.release()
    rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    out = ml.predict({"image": Image.fromarray(rgb), "iouThreshold": 0.45, "confidenceThreshold": CONF})
    ml_persons = int((out["confidence"].argmax(1) == labels.index("person")).sum()) if len(out["confidence"]) else 0
    pt = model.predict(rgb[:, :, ::-1], imgsz=SIZE, conf=CONF, classes=[0], device="cpu", verbose=False)[0]
    print(f"{clip.stem} frame 200 persons: coreml={ml_persons} pytorch={len(pt.boxes)}")


if __name__ == "__main__":
    main()
