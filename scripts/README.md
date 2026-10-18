# scripts/

Command-line tools you run by hand: downloading datasets, extracting frames from video, starting a training run, evaluating a model, exporting to Core ML. Scripts should stay thin. They call code in `pipeline/` instead of holding the logic themselves.
