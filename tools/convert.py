#!/usr/bin/env python3
"""Convert supported COCO ONNX exports on x86 Linux using RKNN Toolkit 2.3.2.

This maintainer tool is separate from the API image and its dependencies.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=[family + size for family in ("yolo26", "yolo11") for size in "nsm"])
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--chip", required=True, choices=("rk3576", "rk3588"))
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Calibration list; relative image paths resolve against this list")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-url", required=True, help="ONNX download URL or export project URL")
    args = parser.parse_args()

    import onnx
    from rknn.api import RKNN

    version = importlib.metadata.version("rknn-toolkit2")
    if version != "2.3.2":
        raise ValueError("Use RKNN Toolkit 2.3.2 to match the image runtime")
    model = onnx.load(str(args.onnx), load_external_data=False)
    inputs = [i for i in model.graph.input if i.name not in {v.name for v in model.graph.initializer}]
    shape = lambda value: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
    if len(inputs) != 1 or shape(inputs[0]) != [1, 3, 640, 640]:
        raise ValueError("Expected a static batch-1 640x640 RGB input")
    outputs = None
    transformations = []
    if args.model.startswith("yolo26"):
        outputs = ["/model.23/Sigmoid_output_0", "/model.23/Mul_2_output_0"]
        available = {output for node in model.graph.node for output in node.output}
        if not set(outputs).issubset(available):
            raise ValueError("YOLO26 export lacks the supported scores/xyxy truncation nodes")
        # These packages intentionally use the one-to-many head and server NMS.
        if any("one2one" in node.name for node in model.graph.node):
            raise ValueError("Export YOLO26 with end2end=False for these NMS-enabled packages")
        # Ultralytics one-to-many exports concatenate (center, size). Keep the
        # already-computed anchor corners instead, before stride multiplication.
        # This makes the split boxes explicitly xyxy, as the native API expects.
        nodes = {node.name: node for node in model.graph.node}
        concat = nodes.get("/model.23/Concat_2")
        if (concat is None or list(concat.input) !=
                ["/model.23/Div_1_output_0", "/model.23/Sub_1_output_0"]):
            raise ValueError("Unrecognized YOLO26 box graph; verify coordinates before converting")
        concat.input[:] = ["/model.23/Sub_output_0", "/model.23/Add_1_output_0"]
        onnx.checker.check_model(model)
        transformations.append("one-to-many xywh head replaced with stride-scaled xyxy corners")
        layout = "scores-xyxy"
    else:
        expected = [[1, channels, grid, grid] for grid in (80, 40, 20) for channels in (64, 80, 1)]
        if [shape(output) for output in model.graph.output] != expected:
            raise ValueError("Use the Rockchip-optimized YOLO11 export with nine DFL outputs")
        layout = "dfl-9-int8"

    images = []
    for line in args.dataset.read_text().splitlines():
        if line.strip():
            path = (args.dataset.parent / line.strip()).resolve()
            if not path.is_file():
                raise ValueError(f"Calibration image missing: {path}")
            images.append(path)
    if len(images) < 20:
        raise ValueError("Provide at least 20 representative calibration images")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.model}_{args.chip}.rknn"
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}; choose a fresh output directory")
    # RKNN resolves dataset paths against cwd. Use a temporary absolute list.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        calibration = Path(temporary) / "calibration.txt"
        calibration.write_text("\n".join(map(str, images)) + "\n")
        compiled_onnx = args.onnx
        if transformations:
            compiled_onnx = Path(temporary) / "model-xyxy.onnx"
            onnx.save(model, str(compiled_onnx))
        compiled_sha256 = digest(compiled_onnx)
        rknn = RKNN(verbose=False)
        try:
            def check(result, stage):
                if result != 0:
                    raise RuntimeError(f"RKNN {stage} failed: {result}")
            check(rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                              target_platform=args.chip), "config")
            kwargs = {"outputs": outputs} if outputs else {}
            check(rknn.load_onnx(model=str(compiled_onnx), **kwargs), "load")
            check(rknn.build(do_quantization=True, dataset=str(calibration)), "build")
            check(rknn.export_rknn(str(target)), "export")
        finally:
            rknn.release()
    metadata = {
        "model": args.model, "author": "Ultralytics", "license": "AGPL-3.0",
        "source_url": args.source_url, "onnx_sha256": digest(args.onnx),
        "compiled_onnx_sha256": compiled_sha256, "transformations": transformations,
        "chip": args.chip, "toolkit": version, "precision": "INT8", "input_shape": [1, 3, 640, 640],
        "output_layout": layout, "nms": True, "classes": 80, "dataset": "COCO",
        "calibration_images": len(images),
        "calibration_images_sha256": hashlib.sha256("\n".join(digest(p) for p in images).encode()).hexdigest(),
        "sha256": digest(target),
        "onnx_metadata": {entry.key: entry.value for entry in model.metadata_props},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"model": args.model, "chip": args.chip, "file": str(target), "sha256": metadata["sha256"]}))


if __name__ == "__main__":
    main()
