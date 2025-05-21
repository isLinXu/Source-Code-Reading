import io
import os
import urllib
from hashlib import sha256

from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image


M4_IMAGES_CACHE = os.getenv(
    "M4_IMAGES_CACHE", os.path.expanduser(os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "m4"))
)
os.makedirs(M4_IMAGES_CACHE, exist_ok=True)


def loading_existing_images():
    files = [f for f in os.listdir(M4_IMAGES_CACHE)]
    return files


def convert_to_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def hash_url_to_filename(url):
    """
    Adapted from datasets.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()
    return filename


# Kind of dirty evil trick, passing `existing_files` as a global variable.
# TODO: Ultimately, should handle that with a real database such as Sqlitedict or Redis, but these require opening and closing a connection to the database,
# and the closing part requires quite non trivial changes (creating a fetcher class, that closes the connection once all downloads are done),
# and it doesn't seem worth it as of now imo. It would also properly handle these None cases (i.e. the download failed).
existing_files = loading_existing_images()


def fetch_single_image(image_url, timeout=None, retries=0, retry_none=False):
    global existing_files
    image_url_hashed = hash_url_to_filename(image_url)

    if image_url_hashed in existing_files:
        image = Image.open(os.path.join(M4_IMAGES_CACHE, image_url_hashed))
    elif f"{image_url_hashed}.blank" in existing_files and not retry_none:
        image = None
    else:
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": get_datasets_user_agent()},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                image = None
        if image is not None:
            image.save(os.path.join(M4_IMAGES_CACHE, image_url_hashed), format=image.format)
            existing_files.append(image_url_hashed)
        else:
            with open(os.path.join(M4_IMAGES_CACHE, f"{image_url_hashed}.blank"), "w"):
                pass
            existing_files.append(f"{image_url_hashed}.blank")

    if image is not None:
        image = convert_to_rgb(image)

    return image
