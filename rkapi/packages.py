"""Model catalog and packaged-model validation; usable without inference libraries."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def checked_path(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path must stay inside {root}: {value}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_spec(spec):
    for key in ("id", "family", "chip", "task", "backend", "tag"):
        if not isinstance(spec.get(key), str) or not IDENTIFIER.fullmatch(spec[key]):
            raise ValueError(f"Invalid model {key}: {spec.get(key)!r}")
    if spec["chip"] not in ("rk3576", "rk3588"):
        raise ValueError(f"Unsupported chip: {spec['chip']}")
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")):
        raise ValueError(f"{spec['id']}: a SHA-256 checksum is required")
    if not isinstance(spec.get("nms", True), bool):
        raise ValueError(f"{spec['id']}: nms must be a boolean")


def load_catalog(path, root):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise ValueError("Expected catalog schema_version 1 with a models list")
    models = document["models"]
    if not models:
        raise ValueError("The model catalog is empty")
    ids, tags = set(), set()
    for spec in models:
        validate_spec(spec)
        if spec["id"] in ids or spec["tag"] in tags:
            raise ValueError("Model IDs and image tags must be unique")
        ids.add(spec["id"])
        tags.add(spec["tag"])
        if bool(spec.get("file")) == bool(spec.get("url")):
            raise ValueError(f"{spec['id']}: provide exactly one of file or url")
        if spec.get("url") and urlsplit(spec["url"]).scheme != "https":
            raise ValueError(f"{spec['id']}: model URL must use HTTPS")
        if spec.get("file"):
            path = checked_path(root, spec["file"])
            if not path.is_file() or sha256(path) != spec["sha256"]:
                raise ValueError(f"{spec['id']}: model file missing or checksum mismatch")
        for key in ("metadata", "labels"):
            if spec.get(key) and not checked_path(root, spec[key]).is_file():
                raise ValueError(f"{spec['id']}: {key} file missing")
        backend = checked_path(root, f"rkapi/backends/{spec['backend']}.py")
        if not backend.is_file():
            raise ValueError(f"{spec['id']}: backend module missing: {spec['backend']}")
    return models


def load_package(path):
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported model package schema")
    validate_spec(spec)
    for key in ("file", "labels", "metadata"):
        if spec.get(key):
            resolved = checked_path(path.parent, spec[key])
            if not resolved.is_file():
                raise ValueError(f"Packaged {key} is missing: {resolved}")
            spec[key] = str(resolved)
    if not spec.get("file") or sha256(spec["file"]) != spec["sha256"]:
        raise ValueError("Packaged model checksum mismatch")
    return spec
