# Rockchip CV API

**Run computer vision on Rockchip boards. Build your application with HTTP.**

Rockchip CV API packages a hardware-accelerated C++ inference engine behind a
REST API. Start a Docker image on your board, send an image, and get results
back as JSON—from Python, curl, or any application that supports HTTP.

**Your application needs HTTP, not an inference environment.** Model weights,
RKNN userspace runtime, hardware acceleration libraries, and API dependencies
are packaged in the image. No RKNN toolkit, Python inference environment, or
model setup is needed on your application machine.

The current release provides **YOLO26 object detection on RK3576 and RK3588**.
Each Docker image includes the matching YOLO26n model and starts the API
service automatically.

[Quick start](#quick-start) · [Python example](#use-it-from-python) · [API reference](#api-reference) · [Camera and RTSP](#camera-and-rtsp) · [Configuration](#configuration) · [Troubleshooting](#troubleshooting)

## What you can do

- **Detect objects in images** by uploading a JPEG or PNG.
- **Inspect a video frame** at a chosen timestamp.
- **Process camera and RTSP streams** and retrieve the latest detections.
- **Get an annotated MJPEG feed** for a configured video source.
- **Adjust confidence and NMS thresholds** without rebuilding the image.
- **Inspect latency** for decoding, preprocessing, inference, NMS, and response handling.

Image preprocessing uses RGA and inference runs on the Rockchip NPU. Continuous
H.264/H.265 video uses MPP hardware decoding. Your application only needs to
call the API; it does not need RKNN or the model installed locally.

## Models and images

Each model/chip combination has its own Docker tag. One container serves one
model; applications use the same `/api/predict` endpoint across images.

| Chip | Bundled model | Catalog ID | Image tag |
| --- | --- | --- | --- |
| RK3576 | YOLO26n | `yolo26n-rk3576` | `rk3576-yolo26` |
| RK3588 | YOLO26n | `yolo26n-rk3588` | `rk3588-yolo26` |

These are the currently supported models. YOLOv8, YOLOv10, YOLO11, and other
vision tasks can be added through the [model catalog and backend adapters](docs/adding-models.md);
they are not bundled or verified yet. Changing the model family may change
its result fields; `/api/models` reports the loaded model and task.

## Quick start

### Before you begin

Use an RK3576 or RK3588 board with **ARM64 Linux, Docker, and a supported board
OS image** that already provides working NPU, RGA, and MPP kernel drivers.
The container bundles the userspace runtime; kernel drivers belong to the
board OS and cannot be installed by the application container. On a supported
board OS, users do not need to install an inference SDK or configure NPU
libraries manually.

### 1. Start the service

Images are built by GitHub Actions and published to the repository's
[GitHub Container Registry](https://github.com/Hanzo-Huang/rockchip-cv-api/pkgs/container/rockchip-cv-api).
Choose the tag matching your board:

```bash
# RK3588. For RK3576, use :rk3576-yolo26 instead.
IMAGE="ghcr.io/hanzo-huang/rockchip-cv-api:rk3588-yolo26"

docker run -d --name cv-api --restart unless-stopped \
  --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  "$IMAGE"
```

Docker pulls the image if needed and starts the API on **port 8000**. There is
no separate model download, model mount, or Python startup command. The run
command grants access to the board's hardware and uses the host network.

You can also [build from source](#build-from-source). Published GitHub
Releases include image archives for [offline installation](#offline-installation).

### 2. Check that it is ready

```bash
curl -fsS http://localhost:8000/health
```

Look for `"status": "ok"`. The response also reports the chip, model layout,
NMS setting, and RKNN runtime/driver versions.

Open **`http://<board-ip>:8000/docs`** in your browser to try the endpoints
interactively. The OpenAPI schema is available at `/openapi.json`.

### 3. Send your first image

Replace `image.jpg` with an image on the machine where you run curl:

```bash
curl -fsS http://localhost:8000/api/predict \
  -F "file=@image.jpg"
```

Running the client on another computer? Replace `localhost` with your board's
IP address. This applies to all examples below.

Example response, with timing fields abbreviated:

```json
{
  "success": true,
  "source": "image",
  "predictions": [
    {
      "class": "person",
      "class_id": 0,
      "confidence": 0.92,
      "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
    }
  ],
  "image": {"width": 1280, "height": 720},
  "timing": {
    "preprocess_ms": 1.2,
    "inference_ms": 11.0,
    "postprocess_ms": 1.4,
    "nms_ms": 0.02
  }
}
```

Box coordinates refer to the **original image**, not the model's resized input.
The bundled models use COCO class names. An image with no detections returns an
empty `predictions` array.

## Use it from Python

Install `requests` on your application machine:

```bash
pip install requests
```

Then call the API:

```python
import requests

api_url = "http://localhost:8000"  # Use the board's IP for a remote client.

with open("image.jpg", "rb") as image:
    response = requests.post(
        f"{api_url}/api/predict",
        files={"file": image},
        data={"conf": 0.3},
        timeout=30,
    )

response.raise_for_status()

for detection in response.json()["predictions"]:
    print(detection["class"], detection["confidence"], detection["box"])
```

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/models` | Describe the loaded model, task, chip, and adapter |
| `POST` | `/api/predict` | Predict an uploaded image, a video frame, or the latest source frame |
| `GET` | `/api/config` | Read the default confidence and NMS thresholds |
| `POST` | `/api/config` | Update the default thresholds |
| `GET` | `/api/video_feed` | Get an annotated MJPEG stream from the configured source |
| `GET` | `/health` | Check model/runtime status and continuous-source state |
| `GET` | `/docs` | Open the interactive API documentation |
| `GET` | `/openapi.json` | Get the OpenAPI schema |

`/api/models/{model_id}/predict` also accepts the loaded catalog ID or family;
other model names return 404. Existing `/api/models/yolo26/predict` and
`/api/models/yolo11/predict` clients continue to work with the YOLO26 images.
This alias does not mean a YOLO11 model is loaded.

### Prediction options

Send prediction parameters as multipart form fields:

| Field | Default | Description |
| --- | --- | --- |
| `file` | — | Image upload |
| `video` | — | Video upload; use instead of `file` |
| `timestamp` | `0` | Frame time in seconds for a video upload |
| `conf` | `0.25` | Confidence threshold, from `0` to `1` |
| `iou` | `0.45` | NMS IoU threshold, from `0` to `1` |
| `realtime` | `false` | Use the latest frame from `RK_SOURCE` |

`conf` and `iou` fall back to the current `/api/config` defaults when omitted.
Overrides apply to one request. Choose one input: `file`, `video`, or the
configured continuous source. A request with no upload uses that source.

**Set thresholds for one image:**

```bash
curl -fsS http://localhost:8000/api/predict \
  -F "file=@image.jpg" -F "conf=0.3" -F "iou=0.5"
```

**Detect a frame from an uploaded video:**

```bash
curl -fsS http://localhost:8000/api/predict \
  -F "video=@clip.mp4" -F "timestamp=1.5"
```

**Change the service defaults:**

```bash
curl -fsS http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"obj_thresh": 0.3, "nms_thresh": 0.5}'
```

NMS runs in C++ on the API server and is **enabled by default**. For a verified
NMS-free export, start the container with `-e RK_NMS=off`. The number of model
output tensors alone does not determine whether NMS is needed.

## Camera and RTSP

Set `RK_SOURCE` when starting the container to enable continuous video
processing. If the quick-start container is already running, stop and remove
it first:

```bash
docker stop cv-api
docker rm cv-api
```

Then start the service with your camera URL. For example, on RK3588:

```bash
docker run -d --name cv-api --restart unless-stopped \
  --privileged --network host \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible:ro \
  -e RK_SOURCE=rtsp://192.168.1.10/live \
  "$IMAGE"
```

On RK3576, use the `rk3576-yolo26` image tag.

To use a local video, mount its directory and set its path **inside** the
container:

```bash
-v "$PWD/media:/media:ro" -e RK_SOURCE=/media/input.mp4
```

For an H.264 V4L2 camera, set `RK_SOURCE=/dev/video0` to the appropriate device.

Once a frame is available, request its detections:

```bash
curl -fsS -X POST http://localhost:8000/api/predict
```

View the MJPEG output at **`http://<board-ip>:8000/api/video_feed`**, or embed it
in an application with an HTML `<img>` element.

Continuous sources support **H.264/H.265**. Live inference automatically uses
two independent contexts on RK3576 and three on RK3588. Local video playback
follows media timestamps and stops at EOF, retaining the last frame for
prediction. Raw/MJPEG V4L2 input and automatic source reconnection are not
supported. Uploaded video frame extraction uses OpenCV; continuous input uses
MPP hardware decoding.

## Configuration

Pass settings with `docker run -e NAME=value`:

| Variable | Default | Description |
| --- | --- | --- |
| `RK_PACKAGE` | `/opt/model/package.json` | Validated model package manifest |
| `RK_NMS` | Model catalog setting (`on` for YOLO26) | Server-side NMS: `on` or `off` |
| `RK_SOURCE` | Empty | RTSP URL, mounted video path, or H.264 V4L2 device |
| `RK_WORKERS` | `0` | Continuous-source workers: automatic (`0`) or one all-core context (`1`) |
| `RK_CLASSES` | COCO names for 80 classes | Text file with one class name per line |
| `RK_MAX_UPLOAD_MB` | `128` | Maximum individual upload size in MB |

The packaged manifest selects the chip and adapter. Startup verifies the model
checksum and checks the mounted board identity before loading the model. Keep
one Uvicorn process per container; it owns the request context and optional
live inference pool. Upload inference is serialized within its native context.
Temporary uploads are removed after each request.

The threshold, NMS, class-label, and live-source controls above apply to the
YOLO adapter. Other task adapters define their own result fields and may not
support live input; unsupported live operations return HTTP 501.

## Build from source

Build on an ARM64 machine (the NPU is only required when running inference):

```bash
python3 tools/models.py list

docker build --platform linux/arm64 --target model \
  --build-arg MODEL_ID=yolo26n-rk3588 \
  -t rockchip-cv-api:rk3588-yolo26 .
```

For RK3576, use `MODEL_ID=yolo26n-rk3576` and tag `rk3576-yolo26`.
Use that local image name in the quick-start run command.

Optional build arguments `UBUNTU_MIRROR` and `PIP_INDEX_URL` select alternate
package mirrors. Models, runtime dependencies, and adapters are selected at
build time; users only choose an image tag.

## Build and publish with GitHub Actions

The [image workflow](.github/workflows/images.yml) reads `models/catalog.json`
and builds each entry on a native ARM64 runner:

- **Pull request:** build every model and check imports/package integrity; no publishing.
- **Push to `main`:** publish each model tag to `ghcr.io/hanzo-huang/rockchip-cv-api`.
- **Run workflow manually:** build all models or select one catalog ID.
- **Publish a release:** publish model tags plus versioned tags, and attach a
  compressed Docker image archive for each model to the GitHub Release.

For example, release `v0.3.0` adds `:rk3588-yolo26-v0.3.0` alongside
`:rk3588-yolo26`. Use a versioned tag when pinning an application deployment.
The workflow uses `GITHUB_TOKEN`; no separate registry password is needed.
After the first publish, set each GHCR package's visibility to **Public** if
users should pull without signing in. ARM64 runner availability and Actions
permissions must be enabled for the repository.

### Offline installation

Download the matching `rk3588-yolo26.tar.gz` or `rk3576-yolo26.tar.gz` asset
from a published release, transfer it to the board, then load it:

```bash
docker load -i rk3588-yolo26.tar.gz
```

Use the image name printed by `docker load` in the quick-start run command.

## Add a model

Add a catalog entry with the chip, image tag, weights, checksum, and backend.
The existing Dockerfile and workflow automatically package/build that entry.
Weights can be stored in the repository or fetched from an HTTPS release URL.

A compatible YOLO export can reuse the `yolo` adapter. A model with a different
output layout or a different task needs an adapter that implements its own
preprocessing, inference, and result formatting. See the
[model contributor guide](docs/adding-models.md) for the catalog format and
adapter contract.

## Understanding latency

The `timing` object breaks down the work performed by the API:

| Field | What it measures |
| --- | --- |
| `decode_ms` | Read/decode the uploaded image or video frame, including thread dispatch |
| `color_convert_ms` | Convert and pack the image into NV12 |
| `context_wait_ms` | Wait for the native inference context |
| `dma_upload_ms` | Prepare the DMA buffer and copy the image into it |
| `preprocess_ms` | RGA resizing, color conversion, and letterboxing |
| `rknn_run_ms` | Execute the model through RKNN |
| `output_get_ms` | Retrieve and release model outputs |
| `postprocess_ms` | Filter detections, apply NMS, map coordinates, and build detection JSON |
| `nms_ms` | Candidate ranking and class-aware suppression, within `postprocess_ms` |

`inference_ms` includes RKNN execution and output retrieval. `native_call_ms`
contains the native stages, and `process_ms` covers image preparation, native
processing, and response construction. These are nested timings; adding the
subtotals to their components would count the same work twice.

The `Server-Timing` response header reports `request_parse`, `receive_wait`,
`response_encode`, and `server_total`. The server total starts at ASGI request
dispatch and ends before sending response headers. Network delivery and client
work are outside this interval. Measure the complete HTTP round trip in your
application when you need end-to-end latency.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| The container exits during startup | Run `docker logs cv-api`. Check the image tag, board drivers, and access to hardware devices. |
| RKNN model initialization fails | Confirm the model was exported for your chip and that you selected the matching image. |
| Port 8000 is already in use | Use `docker ps` to find an existing API container, then stop it before starting another. |
| A realtime request returns HTTP 503 | Configure `RK_SOURCE` and wait for its first frame. Image uploads work without a continuous source. |
| A camera stream cannot be opened | Check the source URL/device and make sure the stream uses H.264/H.265; V4L2 cameras must output H.264. |
| Custom class names are rejected | Check that the class-name file has one nonempty line for every model class. |

Useful service commands:

```bash
docker logs -f cv-api   # Follow logs
docker restart cv-api   # Restart the API
docker stop cv-api      # Stop the API
```

## Project layout

```text
.
├── rkapi/
│   ├── app.py              # Shared HTTP API, uploads, timing, OpenAPI
│   ├── runtime.py          # Model package loader and adapter selection
│   ├── packages.py         # Catalog and package validation
│   └── backends/           # Task adapters (currently YOLO detection)
├── native/                 # Optimized C++ Rockchip inference engine
├── models/catalog.json     # Model packages and Docker tags
├── tools/models.py         # Validate, list, package, generate CI matrix
├── tools/setup-sdk.sh      # Pinned Rockchip SDK build dependencies
├── .github/workflows/      # ARM64 image publishing and release archives
├── docs/adding-models.md    # Contributor guide
└── Dockerfile              # Shared runtime and generic model target
```

## Acknowledgements

The native engine is adapted from
[Hanzo-Huang/rockchip-yolo-cpp](https://github.com/Hanzo-Huang/rockchip-yolo-cpp)
at commit [`76035d3`](https://github.com/Hanzo-Huang/rockchip-yolo-cpp/commit/76035d3da87f66f6147d01a49d49bc3c5fe5fdbf).
The HTTP interface follows
[Seeed's original API](https://github.com/Seeed-Projects/reComputer-RK-CV/tree/main/src/rk3588_yolo11).
Model author, export, and license information is preserved in the
[RK3576](models/yolo26n-rk3576/metadata.yaml) and
[RK3588](models/yolo26n-rk3588/metadata.yaml) metadata files.
