"""
Code from datasets in order to help with the downloading
"""
import copy
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TypeVar
from urllib.parse import urljoin, urlparse

import requests
from datasets import config
from datasets.utils import logging
from datasets.utils.extract import ExtractManager
from datasets.utils.file_utils import (
    DownloadConfig,
    _raise_if_offline_mode_is_enabled,
    _request_with_retry,
    ftp_get,
    ftp_head,
    get_authentication_headers_for_url,
    hash_url_to_filename,
    is_local_path,
    is_remote_url,
)
from datasets.utils.filelock import FileLock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INCOMPLETE_SUFFIX = ".incomplete"

T = TypeVar("T", str, Path)


def cached_path(
    url_or_filename,
    compute_cache_path,
    download_config=None,
    **download_kwargs,
) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
        ValueError: if it couldn't parse the url or filename correctly
        requests.exceptions.ConnectionError: in case of internet connection issue
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)

    cache_dir = download_config.cache_dir or config.DOWNLOADED_DATASETS_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            compute_cache_path=compute_cache_path,
            cache_dir=cache_dir,
            force_download=download_config.force_download,
            proxies=download_config.proxies,
            resume_download=download_config.resume_download,
            user_agent=download_config.user_agent,
            local_files_only=download_config.local_files_only,
            use_etag=download_config.use_etag,
            max_retries=download_config.max_retries,
            use_auth_token=download_config.use_auth_token,
            ignore_url_params=download_config.ignore_url_params,
            download_desc=download_config.download_desc,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif is_local_path(url_or_filename):
        # File, but it doesn't exist.
        raise FileNotFoundError(f"Local file {url_or_filename} doesn't exist")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if output_path is None:
        return output_path

    if download_config.extract_compressed_file:
        output_path = ExtractManager(cache_dir=download_config.cache_dir).extract(
            output_path, force_extract=download_config.force_extract
        )

    return output_path


# Fix because we need the user-agent to be exactly what we wanted.
def http_head(
    url, proxies=None, headers=None, cookies=None, allow_redirects=True, timeout=10, max_retries=0
) -> requests.Response:
    headers = copy.deepcopy(headers) or {}
    # headers["user-agent"] = get_datasets_user_agent(user_agent=headers.get("user-agent"))
    response = _request_with_retry(
        method="HEAD",
        url=url,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        allow_redirects=allow_redirects,
        timeout=timeout,
        max_retries=max_retries,
        max_wait_time=0,
    )
    return response


# Fix because we need the user-agent to be exactly what we wanted.
def http_get(
    url, temp_file, proxies=None, resume_size=0, headers=None, cookies=None, timeout=100, max_retries=0, desc=None
):
    headers = copy.deepcopy(headers) or {}
    # headers["user-agent"] = get_datasets_user_agent(user_agent=headers.get("user-agent"))
    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size:d}-"
    response = _request_with_retry(
        method="GET",
        url=url,
        stream=False,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        max_retries=max_retries,
        timeout=timeout,
        max_wait_time=0,
    )
    if response.status_code == 416:  # Range not satisfiable
        return

    # Who are we kidding ... it's an image ...
    temp_file.write(response.content)

    # content_length = response.headers.get("Content-Length")
    # total = resume_size + int(content_length) if content_length is not None else None
    # with logging.tqdm(
    #     unit="B",
    #     unit_scale=True,
    #     total=total,
    #     initial=resume_size,
    #     desc=desc or "Downloading",
    #     disable=True,  # not logging.is_progress_bar_enabled(),
    # ) as progress:
    #     for chunk in response.iter_content(chunk_size=1024):
    #         progress.update(len(chunk))
    #         temp_file.write(chunk)


# Fix because we expose `compute_cache_path` callback
def get_from_cache(
    url,
    compute_cache_path,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10.0,  # reduce timeout
    resume_download=False,
    user_agent=None,
    local_files_only=False,
    use_etag=True,
    max_retries=0,
    use_auth_token=None,
    ignore_url_params=False,
    download_desc=None,
) -> str:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
    """
    if cache_dir is None:
        cache_dir = config.HF_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    if ignore_url_params:
        # strip all query parameters and #fragments from the URL
        cached_url = urljoin(url, urlparse(url).path)
    else:
        cached_url = url  # additional parameters may be added to the given URL

    connected = False
    response = None
    cookies = None
    etag = None
    head_error = None

    # Try a first time to file the file on the local file system without eTag (None)
    # if we don't ask for 'force_download' then we spare a request
    filename = hash_url_to_filename(cached_url, etag=None)
    cache_path = compute_cache_path(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download and not use_etag:
        return cache_path

    # Prepare headers for authentication
    headers = get_authentication_headers_for_url(url, use_auth_token=use_auth_token)
    if user_agent is not None:
        headers["user-agent"] = user_agent

    # We don't have the file locally or we need an eTag
    if not local_files_only:
        if url.startswith("ftp://"):
            connected = ftp_head(url)
        try:
            response = http_head(
                url,
                allow_redirects=True,
                proxies=proxies,
                timeout=etag_timeout,
                max_retries=max_retries,
                headers=headers,
            )
            if response.status_code == 200:  # ok
                etag = response.headers.get("ETag") if use_etag else None
                for k, v in response.cookies.items():
                    # In some edge cases, we need to get a confirmation token
                    if k.startswith("download_warning") and "drive.google.com" in url:
                        url += "&confirm=" + v
                        cookies = response.cookies
                connected = True
                # Fix Google Drive URL to avoid Virus scan warning
                if "drive.google.com" in url and "confirm=" not in url:
                    url += "&confirm=t"
            # In some edge cases, head request returns 400 but the connection is actually ok
            elif (
                (response.status_code == 400 and "firebasestorage.googleapis.com" in url)
                or (response.status_code == 405 and "drive.google.com" in url)
                or (
                    response.status_code == 403
                    and (
                        re.match(r"^https?://github.com/.*?/.*?/releases/download/.*?/.*?$", url)
                        or re.match(r"^https://.*?s3.*?amazonaws.com/.*?$", response.url)
                    )
                )
                or (response.status_code == 403 and "ndownloader.figstatic.com" in url)
            ):
                connected = True
                logger.info(f"Couldn't get ETag version for url {url}")
            elif response.status_code == 401 and config.HF_ENDPOINT in url and use_auth_token is None:
                raise ConnectionError(
                    f"Unauthorized for URL {url}. Please use the parameter ``use_auth_token=True`` after logging in"
                    " with ``huggingface-cli login``"
                )
        except (OSError, requests.exceptions.Timeout) as e:
            # not connected
            head_error = e
            pass

    # connected == False = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if not connected:
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if local_files_only:
            raise FileNotFoundError(
                f"Cannot find the requested files in the cached path at {cache_path} and outgoing traffic has been"
                " disabled. To enable file online look-ups, set 'local_files_only' to False."
            )
        elif response is not None and response.status_code == 404:
            raise FileNotFoundError(f"Couldn't find file at {url}")
        _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
        if head_error is not None:
            raise ConnectionError(f"Couldn't reach {url} ({repr(head_error)})")
        elif response is not None:
            raise ConnectionError(f"Couldn't reach {url} (error {response.status_code})")
        else:
            raise ConnectionError(f"Couldn't reach {url}")

    # Try a second time
    filename = hash_url_to_filename(cached_url, etag)
    cache_path = compute_cache_path(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # From now on, connected is True.
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            # GET file object
            if url.startswith("ftp://"):
                ftp_get(url, temp_file)
            else:
                http_get(
                    url,
                    temp_file,
                    proxies=proxies,
                    resume_size=resume_size,
                    headers=headers,
                    cookies=cookies,
                    max_retries=max_retries,
                    desc=download_desc,
                )

        logger.info(f"storing {url} in cache at {cache_path}")
        shutil.move(temp_file.name, cache_path)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file)

    return cache_path
