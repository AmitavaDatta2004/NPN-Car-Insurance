"""Strict severity image resolution shared by training and inference datasets."""

from pathlib import Path

from PIL import Image


def resolve_image(raw_path: str, roots=()) -> Path:
    """Preserve class directories; never guess using a repeated basename."""
    path = Path(str(raw_path).replace("\\", "/"))
    if path.is_file():
        return path
    repo = Path(__file__).resolve().parents[4]
    candidates = []
    for root in [*roots, Path.cwd(), repo]:
        root = Path(root)
        candidates.append(root / path)
        marker = "data/raw/car_damage_severity/"
        if marker in path.as_posix():
            suffix = path.as_posix().split(marker, 1)[1]
            candidates.append(root / suffix)
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) > 1:
        raise ValueError(f"Ambiguous severity image path: {raw_path}")
    if found:
        return found.pop()
    raise FileNotFoundError(
        f"Severity image missing: {raw_path}. Run Notebook 00 and use its repository root."
    )


def read_rgb(path: Path) -> Image.Image:
    """Decode the complete image or fail; never substitute synthetic evidence."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise ValueError(f"Cannot decode severity image: {path}") from exc
