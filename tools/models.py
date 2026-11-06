#!/usr/bin/env python3
"""Validate the catalog, generate CI matrices, and prepare model image assets."""

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rkapi.packages import checked_path, load_catalog, sha256


def prepare(spec, output):
    output = output.resolve()
    marker = ".rockchip-model-package"
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("Package output must not replace the repository or its parent")
    if output.exists() and any(output.iterdir()) and not (output / marker).is_file():
        raise ValueError("Refusing to replace a directory not created by this tool")
    for key in ("file", "metadata", "labels"):
        if spec.get(key) and checked_path(ROOT, spec[key]).is_relative_to(output):
            raise ValueError("Package output must not contain source assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        model = staged / "model.rknn"
        if spec.get("file"):
            shutil.copyfile(checked_path(ROOT, spec["file"]), model)
        else:
            with urllib.request.urlopen(spec["url"], timeout=120) as response:
                if not response.url.startswith("https://"):
                    raise ValueError("Model download redirected to an insecure URL")
                with model.open("wb") as target:
                    shutil.copyfileobj(response, target)
        if sha256(model) != spec["sha256"]:
            raise ValueError("Downloaded/copied model checksum mismatch")
        package = {key: value for key, value in spec.items() if key not in ("file", "url", "metadata", "labels")}
        package.update({"schema_version": 1, "file": "model.rknn"})
        for key, filename in (("metadata", "metadata.yaml"), ("labels", "classes.txt")):
            if spec.get(key):
                shutil.copyfile(checked_path(ROOT, spec[key]), staged / filename)
                package[key] = filename
        (staged / "package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        (staged / marker).touch()
        if output.exists():
            output.rmdir() if not any(output.iterdir()) else shutil.rmtree(output)
        shutil.move(str(staged), str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "list", "matrix", "prepare"))
    parser.add_argument("--model", help="Catalog model ID; omit for the complete CI matrix")
    parser.add_argument("--output", type=Path, default=ROOT / ".model-package")
    args = parser.parse_args()
    models = load_catalog(ROOT / "models/catalog.json", ROOT)
    selected = [spec for spec in models if not args.model or spec["id"] == args.model]
    if not selected:
        raise ValueError(f"Unknown model ID: {args.model}")
    if args.command == "matrix":
        print(json.dumps({"include": [{key: spec[key] for key in ("id", "tag", "chip", "backend")} for spec in selected]}))
    elif args.command == "list":
        for spec in selected:
            print(f"{spec['id']:24} {spec['tag']:24} {spec['task']:16} {spec['backend']}")
    elif args.command == "validate":
        print(f"Validated {len(models)} model packages")
    else:
        if not args.model:
            raise ValueError("prepare requires --model")
        prepare(selected[0], args.output)
        print(f"Prepared {selected[0]['id']} in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
