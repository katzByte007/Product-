# OSNet-AIN x1.0 Person Re-ID

## Proposed baseline

- Task: person re-identification across cameras; equipment/product Re-ID is out of scope.
- Architecture: `osnet_ain_x1_0` from Torchreid.
- Expected Re-ID input: 256 x 128 pixels, RGB, ImageNet mean/std normalization.
- Runtime output: one L2-normalized float32 feature vector per crop.
- Operating mode: candidate association and operator confirmation only.
- Automatic identity confirmation: disabled.
- Plant similarity threshold: unset until derived from the labeled validation dataset.

## Artifact status

No checkpoint is included in this repository. Place a reviewed Re-ID checkpoint
at `models/reid/osnet_ain_x1_0/model.pth` or set `VISION_REID_WEIGHTS` to its
location. The provider deliberately does not fall back to ImageNet
classification weights.

Install the optional Torchreid source dependency with
`python -m pip install -r requirements-reid.txt` in the application's existing
environment. The dependency does not download or approve model weights.

Before approval, record the exact checkpoint source/model-zoo entry, training
dataset, retrieval date, SHA-256 checksum, and the checkpoint's own license or
terms. Torchreid's repository license is MIT, but that does not establish the
license for a separately distributed checkpoint or its training data. Obtain
the applicable artifact and training-data approvals independently.

Upstream references:

- Torchreid source: https://github.com/KaiyangZhou/deep-person-reid
- Torchreid model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
- OSNet-AIN paper: https://arxiv.org/abs/1910.06827

## Validation and approval

Use plant-specific cross-camera positive and negative pairs. The validation
report must name the dataset/version, checkpoint checksum, camera coverage,
pair counts, selected maximum false-match rate, resulting threshold, false
match rate, false reject rate, precision, and recall. Do not copy a threshold
from an upstream benchmark.

`backend.reid.validation` reads a CSV with columns `crop_a,crop_b,same` and
optional `camera_a,camera_b`. Paths are relative to the CSV file. Run:

```text
python -m backend.reid.validation validation_pairs.csv \
  --weights models/reid/osnet_ain_x1_0/model.pth \
  --dataset-id plant-reid-validation-v1 \
  --max-false-match-rate 0.01
```

The JSON report includes the CSV and checkpoint SHA-256 hashes, positive and
negative pair counts, covered camera pairs, and the selected operating-point
metrics. Preserve the report as the approval record; the command does not
change runtime configuration.

The target false-match rate is an explicit validation policy input, not a
universal product default. After review, configure the application with:

- `VISION_REID_ENABLED=1`
- `VISION_REID_VALIDATION_STATUS=approved`
- `VISION_REID_VALIDATION_DATASET=<reviewed dataset/version>`
- `VISION_REID_SIMILARITY_THRESHOLD=<threshold from reviewed report>`
- `VISION_REID_WEIGHTS_SHA256=<checkpoint hash from reviewed report>`
- `VISION_REID_WEIGHTS=<reviewed checkpoint path>`

Provider enablement does not enable automatic confirmation. Candidates still
require operator confirmation, and identity confirmation must never be treated
as an employee-name or access-control identity.