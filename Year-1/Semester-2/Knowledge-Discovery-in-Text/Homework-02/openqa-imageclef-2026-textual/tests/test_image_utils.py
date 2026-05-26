from PIL import Image

from openqa_textual.image_utils import (
    binarize,
    increase_contrast,
    pad_image,
    preprocess_for_ocr,
    preprocess_variants_for_ocr,
    resize_for_ocr,
    save_preprocessed_debug_outputs,
    to_rgb,
)


def test_to_rgb_flattens_alpha_on_white() -> None:
    image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    converted = to_rgb(image)
    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (255, 255, 255)


def test_resize_for_ocr_upscales_small_images() -> None:
    image = Image.new("RGB", (100, 50), "white")
    resized = resize_for_ocr(image, min_width=1000)
    assert resized.size == (1000, 500)


def test_resize_for_ocr_does_not_downscale() -> None:
    image = Image.new("RGB", (1200, 50), "white")
    resized = resize_for_ocr(image, min_width=1000)
    assert resized.size == image.size


def test_contrast_and_binarize_return_rgb_images() -> None:
    image = Image.new("RGB", (20, 10), "white")
    assert increase_contrast(image).mode == "RGB"
    assert binarize(image).mode == "RGB"


def test_pad_image_adds_white_border() -> None:
    image = Image.new("RGB", (10, 10), "black")
    padded = pad_image(image, padding=5)
    assert padded.size == (20, 20)
    assert padded.getpixel((0, 0)) == (255, 255, 255)


def test_preprocess_for_ocr_uses_config_variant() -> None:
    image = Image.new("RGB", (100, 50), "white")
    processed = preprocess_for_ocr(
        image,
        {"preprocessing": {"variant": "resize_only", "min_width": 200, "padding": 10}},
    )
    assert processed.size == (220, 120)
    assert processed.mode == "RGB"


def test_preprocess_variants_for_ocr_returns_requested_variants() -> None:
    image = Image.new("RGB", (100, 50), "white")
    variants = preprocess_variants_for_ocr(
        image,
        {"preprocessing": {"variants": ["raw", "contrast"], "min_width": 100, "padding": 0}},
    )
    assert list(variants) == ["raw", "contrast"]
    assert all(processed.mode == "RGB" for processed in variants.values())


def test_save_preprocessed_debug_outputs_from_directory(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (20, 10), "white").save(input_dir / "sample.png")

    manifest = save_preprocessed_debug_outputs(
        input_dir,
        output_dir,
        {"preprocessing": {"variants": ["raw", "resize_only"], "min_width": 40, "padding": 0}},
    )

    assert len(manifest) == 2
    assert all(row["saved"] for row in manifest)
    assert (output_dir / "raw" / "sample.png").exists()
    assert (output_dir / "resize_only" / "sample.png").exists()
