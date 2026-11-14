
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "sky_cutouts"

FILES = [
    RAW_DIR / (
        "level2_2025W19_1B_0181_3D5_"
        "spx_l2b-v27-2026-197.fits"
    ),
    RAW_DIR / (
        "level2_2026W20_1A_1166_1D5_"
        "spx_l2b-v27-2026-246.fits"
    ),
]


def load_observation(path):
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(
            hdul["IMAGE"].data, dtype=float
        ).copy()

        header = hdul["IMAGE"].header.copy()
        wcs = WCS(header).celestial

        date_obs = header.get("DATE-OBS", "Unknown")

    return {
        "image": image,
        "wcs": wcs,
        "date": date_obs,
        "filename": path.name,
    }


def create_target_wcs(ra, dec, size, pixel_scale_deg):
    from astropy.wcs import WCS

    target = WCS(naxis=2)

    target.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    target.wcs.crval = [ra, dec]

    target.wcs.crpix = [
        (size + 1) / 2,
        (size + 1) / 2,
    ]

    target.wcs.cdelt = [
        -pixel_scale_deg,
        pixel_scale_deg,
    ]

    target.wcs.radesys = "ICRS"

    return target


def reproject_cutout(observation, target_wcs, size):
    yy, xx = np.indices((size, size), dtype=float)

    ra_grid, dec_grid = target_wcs.pixel_to_world_values(
        xx, yy
    )

    source_x, source_y = (
        observation["wcs"].world_to_pixel_values(
            ra_grid, dec_grid
        )
    )

    image = observation["image"]

    inside = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 1)
        & (source_y >= 1)
        & (source_x < image.shape[1] - 2)
        & (source_y < image.shape[0] - 2)
    )

    output = np.full((size, size), np.nan)

    sampled = map_coordinates(
        image,
        [source_y, source_x],
        order=1,
        mode="constant",
        cval=np.nan,
    )

    output[inside] = sampled[inside]

    return output


def normalize_frames(frames):
    valid_values = np.concatenate([
        frame[np.isfinite(frame)]
        for frame in frames
        if np.any(np.isfinite(frame))
    ])

    vmin, vmax = np.percentile(
        valid_values, [2, 99.5]
    )

    if vmax <= vmin:
        vmax = vmin + 1

    normalized = []

    for frame in frames:
        scaled = (frame - vmin) / (vmax - vmin)

        scaled = np.clip(scaled, 0, 1)
        scaled[~np.isfinite(scaled)] = 0

        normalized.append(
            np.uint8(np.round(scaled * 255))
        )

    return normalized, float(vmin), float(vmax)


def create_sky_preview(output_path, max_size=1200):
    """Create a lightweight preview of the latest local FITS observation."""
    source = FILES[-1]
    if not source.is_file():
        raise FileNotFoundError(f"Missing required FITS file: {source.name}")

    with fits.open(source, memmap=True) as hdul:
        image = np.asarray(hdul["IMAGE"].data, dtype=float)
        header = hdul["IMAGE"].header.copy()
        height, width = image.shape
        step = max(1, int(np.ceil(max(height, width) / max_size)))
        preview = image[::step, ::step].copy()

    finite = preview[np.isfinite(preview)]
    if finite.size == 0:
        raise RuntimeError("The preview observation contains no valid pixels.")

    vmin, vmax = np.percentile(finite, [2, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1
    preview = np.clip((preview - vmin) / (vmax - vmin), 0, 1)
    preview[~np.isfinite(preview)] = 0
    pixels = np.uint8(np.round(preview * 255))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(output_path)

    return {
        "preview_width": int(pixels.shape[1]),
        "preview_height": int(pixels.shape[0]),
        "source_width": int(width),
        "source_height": int(height),
        "sample_step": step,
        "date": header.get("DATE-OBS", "Unknown"),
    }


def preview_pixel_to_sky(x, y, preview_width, preview_height):
    """Convert a preview-image pixel to ICRS RA/Dec using the FITS WCS."""
    with fits.open(FILES[-1], memmap=True) as hdul:
        height, width = hdul["IMAGE"].data.shape
        wcs = WCS(hdul["IMAGE"].header).celestial

    x = float(x)
    y = float(y)
    preview_width = int(preview_width)
    preview_height = int(preview_height)
    if preview_width < 2 or preview_height < 2:
        raise ValueError("Invalid sky preview dimensions.")
    if not (0 <= x < preview_width and 0 <= y < preview_height):
        raise ValueError("The selected point is outside the sky preview.")

    source_x = x * (width - 1) / (preview_width - 1)
    source_y = y * (height - 1) / (preview_height - 1)
    ra, dec = wcs.pixel_to_world_values(source_x, source_y)
    if not (np.isfinite(ra) and np.isfinite(dec)):
        raise RuntimeError("The selected point has no valid sky coordinates.")
    return float(ra % 360), float(dec)


def validate_inputs(ra, dec, size):
    if not (0 <= ra < 360):
        raise ValueError("RA must be between 0 (inclusive) and 360 degrees.")
    if not (-90 <= dec <= 90):
        raise ValueError("Dec must be between -90 and 90 degrees.")
    if size < 16 or size > 1024:
        raise ValueError("Size must be between 16 and 1024 pixels.")


def generate_cutouts(ra, dec, size=128, output_dir=OUTPUT_DIR, files=None, archive_metadata=None):
    """Generate aligned frames and return their metadata.

    This function is shared by the command-line tool and the Flask viewer.
    A caller-specific output directory prevents simultaneous web requests from
    overwriting one another.
    """
    ra = float(ra)
    dec = float(dec)
    size = int(size)
    output_dir = Path(output_dir)
    validate_inputs(ra, dec, size)

    source_files = [Path(path) for path in (files or FILES)]
    missing = [path.name for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required FITS file(s): " + ", ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSPHEREx Sky Cutout Generator")
    print("===========================")

    print(f"RA: {ra:.6f} deg")
    print(f"Dec: {dec:.6f} deg")
    print(f"Cutout size: {size} pixels")

    observations = [
        load_observation(path)
        for path in source_files
    ]

    # Approximate scale from the first image WCS.
    from astropy.wcs.utils import (
        proj_plane_pixel_scales
    )

    scales = proj_plane_pixel_scales(
        observations[0]["wcs"]
    )

    pixel_scale_deg = float(np.mean(scales))

    target_wcs = create_target_wcs(
        ra,
        dec,
        size,
        pixel_scale_deg,
    )

    frames = []

    for observation in observations:
        print("\nProcessing:", observation["filename"])

        x, y = observation["wcs"].world_to_pixel_values(
            ra, dec
        )

        print(f"Target pixel: x={x:.2f}, y={y:.2f}")

        frame = reproject_cutout(
            observation, target_wcs, size
        )

        valid_fraction = float(
            np.mean(np.isfinite(frame))
        )

        print(
            f"Valid coverage: {valid_fraction:.2%}"
        )

        frames.append(frame)

    if not any(np.any(np.isfinite(f)) for f in frames):
        raise RuntimeError(
            "No observation covers the requested coordinates."
        )

    normalized, vmin, vmax = normalize_frames(frames)

    png_paths = []

    for index, (observation, frame) in enumerate(
        zip(observations, normalized)
    ):
        path = output_dir / f"frame_{index + 1:03d}.png"

        Image.fromarray(frame, mode="L").save(path)
        png_paths.append(path)

        print("Saved:", path)

    fig, axes = plt.subplots(
        1, len(normalized),
        figsize=(10, 5),
    )

    if len(normalized) == 1:
        axes = [axes]

    for ax, observation, frame in zip(
        axes, observations, normalized
    ):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(observation["date"][:10])
        ax.axis("off")

    plt.tight_layout()

    comparison_path = output_dir / "comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close(fig)

    gif_frames = [
        Image.fromarray(frame, mode="L")
        for frame in normalized
    ]

    gif_path = output_dir / "animation.gif"

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=900,
        loop=0,
    )

    metadata = {
        "ra": ra,
        "dec": dec,
        "size_pixels": size,
        "pixel_scale_deg": pixel_scale_deg,
        "display_vmin": vmin,
        "display_vmax": vmax,
        "observations": [
            {
                "filename": observation["filename"],
                "date": observation["date"],
                "frame": png_path.name,
                "valid_fraction": float(
                    np.mean(np.isfinite(frame))
                ),
                **((archive_metadata or [{}] * len(observations))[index]),
            }
            for index, (observation, frame, png_path) in enumerate(zip(
                observations, frames, png_paths
            ))
        ],
        "gif": gif_path.name,
        "comparison": comparison_path.name,
    }

    metadata_path = output_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", comparison_path)
    print("Saved:", gif_path)
    print("Saved:", metadata_path)

    print("\nSky cutout generation completed.")

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    try:
        generate_cutouts(args.ra, args.dec, args.size)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
