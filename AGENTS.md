# Rockchip CV API

This repository ships ARM64 Docker images for RK3576 and RK3588. Each image
bundles one model and its userspace runtime. Application developers call
`POST /api/predict` over HTTP. The board OS supplies the kernel drivers.

## Where to work

- `rkapi/app.py` owns HTTP routes, uploads, and request timing. Preserve the
  existing detection response and compatibility routes when adding tasks.
- `rkapi/backends/` owns task preprocessing, postprocessing, and results.
  `yolo.py` wraps the native engine and formats results across YOLO tasks.
  `rkapi/runtime.py` selects the adapter from the model package.
- `native/engine/` contains the C++ RKNN, RGA, DMA, and MPP pipeline.
  `task_model.cpp`, `task_decode.cpp`, and `task_dense.cpp` decode the
  segmentation, pose, oriented-box, classification, semantic, and depth heads;
  `postprocess.cpp` keeps the optimized detection paths. Reuse it for hardware
  inference. Keep application clients HTTP-only.
- Manifests under `models/<family>/<task>/<size>/<chip>/<precision>/` define
  supported packages and image tags.
  See the existing entries for the package fields.

## Model and image changes

- Check which tasks upstream actually releases for each family and size.
  Add catalog entries after verifying the compiled model on its target chip.
- Match the conversion toolkit to the runtime bundled by the Dockerfile.
  The current release uses RKNN Toolkit and runtime 2.3.2. Keep conversion
  environments separate from the API image.
- Use a distinct tag per chip, model family, task, size, and precision. Bundle
  one model package per image and keep existing aliases working. Pin the RKNN
  file and its metadata to an immutable Hugging Face revision. Record both URLs
  and SHA-256 checksums in `model.json`. Keep provenance and labels with the
  Hugging Face model entry.
- Declare the tensor contract in each catalog entry: `classes`, `end2end`,
  `coordinates`, `preprocess`, `dense_resize`, and `kpt_shape` for pose models.
  These become the native model options and must match the conversion sidecar.
- GitHub Actions builds images from the catalog. Verify the pinned Hugging Face
  URLs are publicly reachable before pushing entries that depend on them.
- Publish each model name, including its task suffix when present, under
  `ghcr.io/<owner>/rockchip-cv-api/<model>` with `<chip>-<precision>` tags.
  Keep the existing
  `rockchip-cv-api` image references as compatibility tags for the same build.

## Shared Docker layers

- Keep the API, native engine, and userspace dependencies in the shared ARM64
  `runtime` stage. Build every variant with the generic `model` target. Keep
  chip, task, size, precision, and `MODEL_ID` inputs confined to model packaging.
- Keep the model package copy to `/opt/model` as the final filesystem layer.
  Put the RKNN weights, manifest, and any model-specific labels or metadata in
  that package. Keep package preparation and downloads in the `package` stage.
- Reuse the shared runtime build cache in CI. For a given runtime revision and
  build configuration, model images must share the same runtime layers so a
  board can reuse them when pulling another tag. Changes to the runtime or its
  dependencies can create new shared layers.
- When changing the Dockerfile or image workflow, build two different model
  variants with the same runtime and compare their `RootFS.Layers` with
  `docker image inspect`. Only the final model package layer should differ.

## Validation

Run `python3 tools/models.py validate` after catalog changes. To build one
image, use the generic Docker target with its catalog ID:

```sh
docker build --platform linux/arm64 --target model \
  --build-arg MODEL_ID=yolo26n-rk3588-int8 \
  -t yolo26n:rk3588-int8 .
```

For changes that affect inference, use a focused smoke test on the matching
board. Check `/health` and `/api/models`, then send one real image suited to
the task to `/api/predict` and verify the result. Check both boards when
claiming support for both. Report separately what passed in CI, what ran on
hardware, and whether the hardware test used the published image.

Use temporary containers and files for smoke tests. Keep fixtures,
checkpoints, ONNX intermediates, logs, and generated packages outside the
repository. Preserve existing workspace edits and running services.
