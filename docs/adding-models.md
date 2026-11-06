# Add a model

Each catalog entry produces one image for one model and chip. The catalog is
shared by the packaging tool, Docker build, and GitHub Actions build matrix.
The HTTP routes are independent of the task adapter.

## 1. Choose an adapter

`rkapi/backends/yolo.py` implements the current detection adapter using the
optimized C++ RGA/DMA/RKNN engine. The native postprocessor currently handles
a decoded YOLO head, separate scores/boxes, and a nine-output INT8 DFL layout.
A model family name alone does not establish compatibility: inspect the RKNN
input/output tensors and verify detections using a real image.

YOLOv8, YOLOv10, and YOLO11 exports are future additions. Reuse `yolo` only when
the export matches a supported layout. Different layouts need native
postprocessor changes or a separate adapter. In particular, do not assume
that every end-to-end/NMS-free export is already supported.

For classification, segmentation, OCR, or another task, add an adapter as
described below. No placeholder models are advertised as supported.

## 2. Add a catalog entry

Add an entry to `models/catalog.json`. For example, the existing RK3588 entry
uses this structure (replace the example checksum with the actual SHA-256):

```json
{
  "id": "yolo26n-rk3588",
  "name": "YOLO26n",
  "family": "yolo26",
  "chip": "rk3588",
  "task": "detection",
  "backend": "yolo",
  "tag": "rk3588-yolo26",
  "file": "models/yolo26n-rk3588/yolo26n_rk3588.rknn",
  "sha256": "<64 lowercase hexadecimal characters>",
  "metadata": "models/yolo26n-rk3588/metadata.yaml",
  "nms": true
}
```

- IDs and tags must be unique lowercase identifiers. Use distinct tags for
  model sizes, variants, and chips, for example `rk3588-yolov8n`.
- `chip` is `rk3576` or `rk3588`. Export a separate RKNN model for each chip.
- `backend` selects `rkapi/backends/<backend>.py`; it must exist.
- Provide exactly one of `file` or `url`. Local paths are relative to the
  repository root. `url` must be HTTPS and accessible during CI without secrets.
- For large weights, upload a model asset to a GitHub Release first, then use
  its permanent `https://github.com/.../releases/download/.../model.rknn` URL.
  Avoid `latest` URLs: pin the asset and its checksum for reproducible builds.
- `metadata` is an optional local export/provenance file. `labels` is an
  optional local UTF-8 file with one label per class. The YOLO adapter checks
  the label count; otherwise it uses COCO names for 80-class models.
- `nms` controls the YOLO adapter's default. Set `false` only for a verified
  NMS-free export. NMS runs on the API server, not in the client.

Compute the checksum with `sha256sum model.rknn` on Linux or
`shasum -a 256 model.rknn` on macOS. Keep the model's license and export
metadata alongside the entry.

## 3. Build and verify

```bash
python3 tools/models.py validate
python3 tools/models.py list
python3 tools/models.py matrix --model yolo26n-rk3588

docker build --platform linux/arm64 --target model \
  --build-arg MODEL_ID=yolo26n-rk3588 \
  -t rockchip-cv-api:rk3588-yolo26 .
```

The generic `model` target packages only the selected model in the final
image. It checks the weights against the catalog SHA-256; startup checks
again. SDK and native compilation layers are shared between model builds.
Neither the Dockerfile nor workflow needs a new target or matrix entry.

For inspection without Docker:

```bash
python3 tools/models.py prepare --model yolo26n-rk3588 --output /tmp/cv-model-package
```

This generates `package.json`, `model.rknn`, and any metadata/labels. Packaging
refuses to replace a nonempty directory it did not create. A custom package
can be mounted read-only with `-v /path/to/package:/opt/model:ro` using an
image that contains its adapter. Do not override a weight file without
updating its manifest and checksum.

Run the image on its matching board using the README command. Check
`/health`, `/api/models`, and one real image through `/api/predict`. Verify
that the output is correct for that task. Hosted CI checks builds, imports,
and package integrity; it cannot validate NPU inference without hardware.

After merging, GitHub Actions discovers the entry automatically. Publishing
a release also attaches a Docker image archive for offline installation.

## Adapter contract for other tasks

Create `rkapi/backends/<name>.py` and define `class Backend(BaseBackend)` from
`rkapi.backends.base`. The loader imports only the selected adapter, so the
shared API has no direct dependency on YOLO tensor layouts or detection boxes.

Implement:

- `__init__(package)`: call `super().__init__(package)`, load
  `package["file"]`, and set `self.description` to JSON-compatible runtime
  information. Package asset paths are absolute and checksum-verified.
- `image(image, conf, iou, source)`: accept a decoded BGR NumPy image and return
  a JSON-compatible task result. Include `success`, `source`, and a `timing`
  dictionary; use task-appropriate fields, such as classification `results`
  instead of detection `predictions`. The HTTP layer adds `model` metadata
  and upload/decode/process timing. `conf`/`iou` are detection controls and
  can be ignored for tasks where they do not apply.
- `close()`: release any resources owned by the adapter. Concurrent HTTP
  calls run in worker threads; protect shared native contexts appropriately.

Live support is optional. The base class returns HTTP 501 for unsupported
live operations. A live adapter implements `predict_realtime(conf, iou)`,
`preview()`, `source_status()`, and optionally `annotate(jpeg, raw)` and
`configure(values)`. `preview()` returns `(state, jpeg_bytes)`; state contains
`result` (with monotonically increasing `frame`), `error`, and `ended`.
Raise `SourceUnavailable` for a configured source that is not ready (503),
or `UnsupportedOperation` for an unsupported feature (501).

The current image provides Rockchip libraries and the YOLO native extension.
If another adapter requires additional native bindings or dependencies, add
those to the shared build explicitly. The adapter interface does not make
arbitrary model formats work automatically. Keep runtime dependencies in the
image so application developers continue to need only HTTP.
