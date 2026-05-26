"""Image preprocessing utilities for OCR-focused question extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


SUPPORTED_PREPROCESSING_VARIANTS = (
    "raw",
    "resize_only",
    "contrast",
    "binarized",
    "deskewed",
)


def to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of a PIL image."""

    if image.mode == "RGB":
        return image.copy()
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def resize_for_ocr(image: Image.Image, min_width: int = 1000) -> Image.Image:
    """Upscale small images while preserving aspect ratio."""

    rgb = to_rgb(image)
    if min_width <= 0 or rgb.width >= min_width:
        return rgb

    scale = min_width / rgb.width
    new_size = (min_width, max(1, int(round(rgb.height * scale))))
    return rgb.resize(new_size, Image.Resampling.LANCZOS)


def deskew(image: Image.Image) -> Image.Image:
    """Estimate and correct small text skew angles using foreground pixels."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return to_rgb(image)

    rgb = to_rgb(image)
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(threshold > 0))
    if coords.size == 0:
        return rgb

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1 or abs(angle) > 15:
        return rgb

    height, width = threshold.shape
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        np.array(rgb),
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rotated).convert("RGB")


def denoise(image: Image.Image) -> Image.Image:
    """Reduce small speckles while preserving text edges."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return to_rgb(image).filter(ImageFilter.MedianFilter(size=3))

    rgb = to_rgb(image)
    array = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(array, None, 7, 7, 7, 21)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)).convert("RGB")


def increase_contrast(image: Image.Image) -> Image.Image:
    """Convert to grayscale and increase contrast for OCR."""

    grayscale = ImageOps.grayscale(to_rgb(image))
    autocontrasted = ImageOps.autocontrast(grayscale)
    enhanced = ImageEnhance.Contrast(autocontrasted).enhance(1.6)
    return enhanced.convert("RGB")


def binarize(image: Image.Image) -> Image.Image:
    """Convert an image to high-contrast black/white text."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        grayscale = ImageOps.grayscale(to_rgb(image))
        return grayscale.point(lambda pixel: 255 if pixel > 180 else 0).convert("RGB")

    grayscale = cv2.cvtColor(np.array(to_rgb(image)), cv2.COLOR_RGB2GRAY)
    threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return Image.fromarray(threshold).convert("RGB")


def pad_image(image: Image.Image, padding: int = 20) -> Image.Image:
    """Add a white margin around an image to help OCR near boundaries."""

    rgb = to_rgb(image)
    if padding <= 0:
        return rgb
    return ImageOps.expand(rgb, border=padding, fill="white")


def preprocess_for_ocr(image: Image.Image, config: dict[str, Any] | None = None) -> Image.Image:
    """Preprocess a PIL image according to a configurable OCR variant."""

    options = _preprocessing_options(config)
    variant = str(options["variant"])
    min_width = int(options["min_width"])
    padding = int(options["padding"])

    if variant == "ensemble":
        variant = "binarized"

    if variant not in SUPPORTED_PREPROCESSING_VARIANTS:
        raise ValueError(
            f"Unsupported preprocessing variant '{variant}'. "
            f"Expected one of: {', '.join(SUPPORTED_PREPROCESSING_VARIANTS)}, ensemble."
        )

    processed = to_rgb(image)
    if variant == "raw":
        return processed

    processed = resize_for_ocr(processed, min_width=min_width)

    if variant == "resize_only":
        return pad_image(processed, padding=padding)
    if variant == "contrast":
        return pad_image(increase_contrast(processed), padding=padding)
    if variant == "binarized":
        return pad_image(binarize(increase_contrast(denoise(processed))), padding=padding)
    if variant == "deskewed":
        return pad_image(binarize(increase_contrast(deskew(denoise(processed)))), padding=padding)

    return processed


def preprocess_variants_for_ocr(
    image: Image.Image,
    config: dict[str, Any] | None = None,
) -> dict[str, Image.Image]:
    """Return configured OCR preprocessing variants for ensemble/debug runs."""

    options = _preprocessing_options(config)
    variants = options["variants"]
    if not variants or "ensemble" in variants:
        variants = list(SUPPORTED_PREPROCESSING_VARIANTS)

    outputs: dict[str, Image.Image] = {}
    for variant in variants:
        if variant == "ensemble":
            continue
        outputs[variant] = preprocess_for_ocr(
            image,
            {
                "variant": variant,
                "min_width": options["min_width"],
                "padding": options["padding"],
            },
        )
    return outputs


def save_preprocessed_debug_outputs(
    input_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    n: int = 30,
) -> list[dict[str, Any]]:
    """Preprocess PNG/JPEG images from a directory and save debug variants."""

    source_dir = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path
        for path in sorted(source_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ][: max(n, 0)]

    manifest: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                variants = preprocess_variants_for_ocr(image, config=config)
            for variant, processed in variants.items():
                variant_dir = destination / variant
                variant_dir.mkdir(parents=True, exist_ok=True)
                output_path = variant_dir / f"{image_path.stem}.png"
                processed.save(output_path, format="PNG")
                manifest.append(
                    {
                        "source": str(image_path),
                        "variant": variant,
                        "path": str(output_path),
                        "width": processed.width,
                        "height": processed.height,
                        "mode": processed.mode,
                        "saved": True,
                    }
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            manifest.append({"source": str(image_path), "saved": False, "error": str(exc)})
    return manifest


def _preprocessing_options(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    preprocessing = config.get("preprocessing", config)
    if not isinstance(preprocessing, dict):
        preprocessing = {}

    variant = (
        preprocessing.get("variant")
        or config.get("default_preprocess_variant")
        or config.get("variant")
        or "resize_only"
    )
    variants = preprocessing.get("variants") or config.get("variants") or [variant]

    return {
        "variant": variant,
        "variants": list(variants),
        "min_width": preprocessing.get("min_width", config.get("min_width", 1000)),
        "padding": preprocessing.get("padding", config.get("padding", 20)),
    }
