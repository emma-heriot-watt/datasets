import hashlib

from PIL import Image


def get_image_md5(image_path: str) -> str:
    """Get MD5 of image."""
    image_as_bytes = Image.open(image_path).tobytes()
    md5_hash = hashlib.md5(image_as_bytes)  # noqa: S303
    return md5_hash.hexdigest()
