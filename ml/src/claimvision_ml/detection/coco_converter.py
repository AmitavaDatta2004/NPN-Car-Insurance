"""coco_converter.py — COCO JSON → YOLO TXT conversion with audit and assertions.

Task ID : DET-COCO-001
Phase   : 8
Owner   : Member 4 / Antigravity
Dataset : COCO Car Damage Detection Dataset
          59 train / 11 val / 8 test images
          Categories: damage (generic) + headlamp, rear bumper, door, hood, front bumper

Public API
----------
ValidationResult          dataclass with audit output
COCOtoYOLOConverter
    load_coco_json()          load COCO JSON from disk
    validate_structure()      audit images, boxes, categories
    draw_coco_boxes()         draw labelled rectangles on an image copy
    convert_bbox_to_yolo()    convert one [x,y,w,h] bbox to YOLO normalised format
    convert_split()           convert a full split and write labels/ and images/
    write_data_yaml()         write data.yaml for Ultralytics YOLO
    run_conversion_assertions() verify all label files are well-formed
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Summary of a COCO JSON audit for one split."""

    is_valid: bool
    image_count: int
    annotation_count: int
    category_counts: dict[str, int] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)
    invalid_boxes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # noqa: D401
        lines = [
            f"ValidationResult(is_valid={self.is_valid})",
            f"  images         : {self.image_count}",
            f"  annotations    : {self.annotation_count}",
            f"  categories     : {self.category_counts}",
            f"  missing files  : {len(self.missing_files)}",
            f"  invalid boxes  : {len(self.invalid_boxes)}",
            f"  warnings       : {len(self.warnings)}",
        ]
        if self.missing_files:
            lines.append(f"  missing        : {self.missing_files[:5]}")
        if self.invalid_boxes:
            lines.append(f"  invalid sample : {self.invalid_boxes[:3]}")
        if self.warnings:
            lines.append(f"  warning sample : {self.warnings[:3]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Converter class
# ---------------------------------------------------------------------------


class COCOtoYOLOConverter:
    """Audit and convert COCO object-detection annotations to YOLO TXT format.

    All methods are stateless class methods so they can be called without
    instantiation.  The class groups related utilities under one namespace.
    """

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load_coco_json(cls, json_path: str | Path) -> dict[str, Any]:
        """Load and return a COCO JSON annotation file as a Python dict.

        Parameters
        ----------
        json_path:
            Absolute or relative path to the ``_annotations.coco.json`` file.

        Returns
        -------
        dict
            The raw COCO structure with keys ``images``, ``annotations``,
            ``categories``, etc.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        json.JSONDecodeError
            When the file is not valid JSON.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"COCO JSON not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    @classmethod
    def validate_structure(
        cls,
        coco: dict[str, Any],
        image_dir: str | Path,
    ) -> ValidationResult:
        """Audit a COCO dict for structural integrity.

        Checks performed
        ----------------
        - Required top-level keys: ``images``, ``annotations``, ``categories``.
        - Every image file listed in the JSON exists on disk.
        - Every annotation references a valid ``image_id``.
        - Bounding boxes ``[x, y, w, h]`` have positive area and lie within the
          declared image dimensions.

        Parameters
        ----------
        coco:
            Loaded COCO dict.
        image_dir:
            Directory that contains the image files listed in ``coco["images"]``.

        Returns
        -------
        ValidationResult
            ``is_valid`` is ``True`` when no missing files or invalid boxes are
            found (warnings do not affect validity).
        """
        image_dir = Path(image_dir)
        warnings: list[str] = []
        missing_files: list[str] = []
        invalid_boxes: list[dict[str, Any]] = []

        # Required keys
        for key in ("images", "annotations", "categories"):
            if key not in coco:
                return ValidationResult(
                    is_valid=False,
                    image_count=0,
                    annotation_count=0,
                    warnings=[f"Missing required COCO key: '{key}'"],
                )

        # Build lookup maps
        image_map: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        cat_map: dict[int, str] = {cat["id"]: cat["name"] for cat in coco["categories"]}
        category_counts: dict[str, int] = {name: 0 for name in cat_map.values()}

        # Check file existence
        for img in coco["images"]:
            fpath = image_dir / img["file_name"]
            if not fpath.exists():
                missing_files.append(img["file_name"])

        # Validate annotations
        for ann in coco["annotations"]:
            img_id = ann.get("image_id")
            if img_id not in image_map:
                warnings.append(
                    f"annotation id={ann.get('id')} references unknown image_id={img_id}"
                )
                continue

            img_info = image_map[img_id]
            img_w: int = img_info.get("width", 0)
            img_h: int = img_info.get("height", 0)
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])

            # Validate box
            if w <= 0 or h <= 0:
                invalid_boxes.append(
                    {"ann_id": ann.get("id"), "image_id": img_id, "reason": "zero or negative area"}
                )
            elif x < 0 or y < 0:
                invalid_boxes.append(
                    {"ann_id": ann.get("id"), "image_id": img_id, "reason": "negative origin"}
                )
            elif img_w > 0 and (x + w > img_w):
                invalid_boxes.append(
                    {
                        "ann_id": ann.get("id"),
                        "image_id": img_id,
                        "reason": f"bbox exceeds image width ({x+w:.1f} > {img_w})",
                    }
                )
            elif img_h > 0 and (y + h > img_h):
                invalid_boxes.append(
                    {
                        "ann_id": ann.get("id"),
                        "image_id": img_id,
                        "reason": f"bbox exceeds image height ({y+h:.1f} > {img_h})",
                    }
                )

            # Count per category
            cat_name = cat_map.get(ann.get("category_id", -1), "unknown")
            if cat_name in category_counts:
                category_counts[cat_name] += 1
            else:
                warnings.append(
                    f"annotation id={ann.get('id')} has unknown category_id={ann.get('category_id')}"
                )

        is_valid = len(missing_files) == 0 and len(invalid_boxes) == 0

        return ValidationResult(
            is_valid=is_valid,
            image_count=len(coco["images"]),
            annotation_count=len(coco["annotations"]),
            category_counts=category_counts,
            missing_files=missing_files,
            invalid_boxes=invalid_boxes,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    @classmethod
    def draw_coco_boxes(
        cls,
        image: np.ndarray,
        annotations: list[dict[str, Any]],
        categories: dict[int, str],
        *,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw labelled COCO bounding boxes on a copy of an image.

        The original array is never modified.

        Parameters
        ----------
        image:
            BGR or RGB image as a NumPy array (H × W × 3).
        annotations:
            List of COCO annotation dicts each with ``bbox`` and ``category_id``.
        categories:
            Mapping of ``category_id`` → ``category_name``.
        thickness:
            Rectangle line thickness in pixels.

        Returns
        -------
        np.ndarray
            Copy of ``image`` with rectangles and labels drawn.
        """
        out = image.copy()
        colour_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 165, 0), (128, 0, 128), (0, 255, 255),
        ]
        cat_ids = sorted(categories.keys())
        colour_map = {cid: colour_palette[i % len(colour_palette)] for i, cid in enumerate(cat_ids)}

        for ann in annotations:
            x, y, w, h = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            cat_id = ann.get("category_id", -1)
            label = categories.get(cat_id, str(cat_id))
            colour = colour_map.get(cat_id, (200, 200, 200))
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
            cv2.putText(
                out, label, (x, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @classmethod
    def convert_bbox_to_yolo(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> tuple[float, float, float, float]:
        """Convert one COCO ``[x, y, w, h]`` box to YOLO normalised format.

        COCO format: top-left pixel coordinates, width and height in pixels.
        YOLO format: centre_x, centre_y, width, height — all divided by image
        dimensions so values lie in ``[0, 1]``.

        Parameters
        ----------
        x, y:
            Top-left corner of the bounding box in pixels.
        w, h:
            Width and height of the bounding box in pixels.
        img_w, img_h:
            Image dimensions in pixels.

        Returns
        -------
        tuple[float, float, float, float]
            ``(cx_norm, cy_norm, w_norm, h_norm)`` in ``[0, 1]``.

        Raises
        ------
        ValueError
            When ``w`` or ``h`` are zero or negative, or image dimensions are zero.
        """
        if w <= 0:
            raise ValueError(f"Bounding-box width must be positive, got w={w}")
        if h <= 0:
            raise ValueError(f"Bounding-box height must be positive, got h={h}")
        if img_w <= 0 or img_h <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got img_w={img_w}, img_h={img_h}"
            )

        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        # Clamp to [0, 1] to guard against 1-pixel rounding overflows
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        return cx, cy, nw, nh

    # ------------------------------------------------------------------
    # Split conversion
    # ------------------------------------------------------------------

    @classmethod
    def convert_split(
        cls,
        coco: dict[str, Any],
        image_dir: str | Path,
        out_dir: str | Path,
        category_map: dict[int, int],
        class_names: list[str],
    ) -> None:
        """Convert one COCO split to YOLO directory layout.

        Output structure::

            out_dir/
            ├── images/     (symlinks or copies of source images)
            └── labels/     (one .txt per image, empty if no annotations)

        A round-trip assertion is run internally: each written normalised value
        is back-converted to pixel coordinates and compared with the original
        within a tolerance of 1e-6.

        Parameters
        ----------
        coco:
            Loaded COCO dict for this split.
        image_dir:
            Source directory containing the images.
        out_dir:
            Destination root.  Created if absent.
        category_map:
            ``{coco_category_id: yolo_class_id}``.  Annotations whose
            ``category_id`` is not in this map are skipped with a warning.
        class_names:
            List of class name strings in YOLO class-id order (used only for
            ``data.yaml``; not written here).
        """
        image_dir = Path(image_dir)
        out_dir = Path(out_dir)
        images_out = out_dir / "images"
        labels_out = out_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Build per-image annotation index
        image_map: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        ann_index: dict[int, list[dict]] = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            iid = ann.get("image_id")
            if iid in ann_index:
                ann_index[iid].append(ann)

        for img_info in coco["images"]:
            iid = img_info["id"]
            fname = img_info["file_name"]
            img_w: int = img_info.get("width", 0)
            img_h: int = img_info.get("height", 0)

            # If dimensions missing, try to read from file
            if img_w == 0 or img_h == 0:
                src = image_dir / fname
                if src.exists():
                    tmp = cv2.imread(str(src))
                    if tmp is not None:
                        img_h, img_w = tmp.shape[:2]

            # Copy image
            src_img = image_dir / fname
            dst_img = images_out / fname
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))

            # Write label file
            stem = Path(fname).stem
            label_file = labels_out / f"{stem}.txt"
            lines: list[str] = []

            for ann in ann_index.get(iid, []):
                cat_id = ann.get("category_id", -1)
                if cat_id not in category_map:
                    continue  # skip categories not in this task
                yolo_cls = category_map[cat_id]
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                    continue  # skip invalid boxes
                cx, cy, nw, nh = cls.convert_bbox_to_yolo(x, y, w, h, img_w, img_h)

                # Round-trip assertion (internal)
                cx_px = cx * img_w - nw * img_w / 2
                cy_px = cy * img_h - nh * img_h / 2
                w_px = nw * img_w
                h_px = nh * img_h
                assert abs(cx_px - x) < 1e-6 + 1e-6 * abs(x), (
                    f"Round-trip x failed: original={x}, recovered={cx_px}"
                )
                assert abs(cy_px - y) < 1e-6 + 1e-6 * abs(y), (
                    f"Round-trip y failed: original={y}, recovered={cy_px}"
                )
                assert abs(w_px - w) < 1e-6 + 1e-6 * abs(w), (
                    f"Round-trip w failed: original={w}, recovered={w_px}"
                )
                assert abs(h_px - h) < 1e-6 + 1e-6 * abs(h), (
                    f"Round-trip h failed: original={h}, recovered={h_px}"
                )

                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_file.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # data.yaml
    # ------------------------------------------------------------------

    @classmethod
    def write_data_yaml(
        cls,
        out_dir: str | Path,
        nc: int,
        names: list[str],
        train_path: str = "train/images",
        val_path: str = "valid/images",
        test_path: str = "test/images",
    ) -> None:
        """Write a ``data.yaml`` file compatible with Ultralytics YOLO.

        Parameters
        ----------
        out_dir:
            Root directory where ``data.yaml`` will be written.
        nc:
            Number of classes.
        names:
            Class name list in YOLO class-id order.
        train_path, val_path, test_path:
            Relative paths (from ``out_dir``) to the image directories for each
            split.  Defaults match the structure produced by ``convert_split``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "path": str(out_dir.resolve()),
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "nc": nc,
            "names": names,
        }
        yaml_path = out_dir / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    @classmethod
    def run_conversion_assertions(cls, yolo_dir: str | Path) -> None:
        """Verify all label files in a converted YOLO directory are well-formed.

        Checks
        ------
        - ``labels/`` subdirectory exists.
        - Every ``.txt`` file is parseable (each non-empty line has exactly 5
          space-separated tokens: class_id cx cy w h).
        - All normalised values ``cx``, ``cy``, ``w``, ``h`` lie in ``[0, 1]``.
        - ``class_id`` is a non-negative integer.

        Prints ``PASS`` to stdout on success.

        Raises
        ------
        AssertionError
            On the first malformed label file or value found.
        """
        yolo_dir = Path(yolo_dir)
        labels_dir = yolo_dir / "labels"
        assert labels_dir.exists(), f"labels/ directory not found in {yolo_dir}"

        txt_files = list(labels_dir.rglob("*.txt"))
        assert len(txt_files) > 0, f"No .txt label files found in {labels_dir}"

        for txt_file in sorted(txt_files):
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                continue  # empty label file is valid (no annotations)
            for lineno, raw_line in enumerate(content.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                assert len(parts) == 5, (
                    f"{txt_file.name}:{lineno} — expected 5 tokens, got {len(parts)}: '{line}'"
                )
                cls_id_str, cx_str, cy_str, w_str, h_str = parts
                assert cls_id_str.isdigit(), (
                    f"{txt_file.name}:{lineno} — class_id must be a non-negative integer, got '{cls_id_str}'"
                )
                for name, val_str in [("cx", cx_str), ("cy", cy_str), ("w", w_str), ("h", h_str)]:
                    val = float(val_str)
                    assert 0.0 <= val <= 1.0, (
                        f"{txt_file.name}:{lineno} — {name}={val} is outside [0, 1]"
                    )

        print(f"PASS — {len(txt_files)} label file(s) in {yolo_dir} are well-formed.")
