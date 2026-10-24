"""Décodage et export des photos, sans modifier les fichiers sources."""

import io
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


MAX_PHOTO_BYTES = 32 * 1024 * 1024
MAX_PHOTO_PIXELS = 16_000_000


def decode_photo(raw):
    if not raw or len(raw) > MAX_PHOTO_BYTES:
        raise ValueError("Choisir une photo JPEG/PNG de moins de 32 Mio.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Formats acceptés : JPEG et PNG.")
                if image.width * image.height > MAX_PHOTO_PIXELS:
                    raise ValueError("La photo dépasse 16 mégapixels ; réduire sa taille pour ce PC.")
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    raise ValueError("Choisir une photo opaque, sans transparence.")
                corrected = ImageOps.exif_transpose(image)
                profile = corrected.info.get("icc_profile")
                if profile:
                    corrected = ImageCms.profileToProfile(
                        corrected, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                        ImageCms.createProfile("sRGB"), outputMode="RGB",
                    )
                else:
                    corrected = corrected.convert("RGB")
                return np.array(corrected, dtype=np.uint8, copy=True)
    except ImageCms.PyCMSError as exc:
        raise ValueError("Le profil couleur de cette photo est invalide. Exporter une copie JPEG/PNG en sRGB.") from exc
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("La photo ne peut pas être décodée ou dépasse la limite mémoire.") from exc


def png_bytes(rgb):
    output = io.BytesIO()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(output, format="PNG", icc_profile=profile)
    return output.getvalue()


def preview_bytes(rgb, size=1200):
    image = Image.fromarray(rgb)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
