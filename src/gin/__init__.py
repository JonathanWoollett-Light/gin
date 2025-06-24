import tempfile
import requests
from io import BytesIO
from tqdm import tqdm
import os
import zipfile

SUFFIX = "afrOfUbwOs"
CACHE = os.path.join(tempfile.gettempdir(), SUFFIX)
if not os.path.exists(CACHE):
    os.mkdir(CACHE)


def download(url: str, chunk_size: int = 1024 * 64, track: bool = True) -> BytesIO:
    """
    Downloads a file from a URL into an in-memory bytes buffer with a progress bar.

    Args:
        url (str): The URL to download from.
        chunk_size (int): Size of chunks to read at a time (in bytes).
        track (bool): Whether to track download progress.

    Returns:
        io.BytesIO: Buffer containing the downloaded data.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    buffer = BytesIO()

    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading",
        disable=not track,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                buffer.write(chunk)
                bar.update(len(chunk))
    buffer.seek(0)
    return buffer


def unzip(buffer: BytesIO, track: bool = True) -> dict[str, bytes]:
    """
    Unzips the contents of an in-memory ZIP file stored in a BytesIO buffer.

    Args:
        buffer (io.BytesIO): Buffer containing ZIP file data

    Returns:
        dict: {filename: bytes} mapping of unzipped files
    """
    # Reset buffer position to start
    buffer.seek(0)

    # Dictionary to store results {filename: bytes}
    unzipped_files: dict[str, bytes] = {}

    with zipfile.ZipFile(buffer, "r") as zip_ref:
        # Get list of files in archive
        members = zip_ref.infolist()
        total_bytes = sum(getattr(m, "file_size", 0) for m in members)
        total_files = len(members)

        file_bar = tqdm(
            desc="Files", total=total_files, unit="file", position=0, disable=not track
        )
        byte_bar = tqdm(
            desc="Bytes",
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            position=1,
            disable=not track,
        )

        for member in members:
            # Skip directories
            if member.is_dir():
                file_bar.update(1)
                continue

            with zip_ref.open(member) as file_in_zip:
                content = file_in_zip.read()
                unzipped_files[os.path.basename(member.filename)] = content
                # Update progress bars
                byte_bar.update(getattr(member, "file_size", 0))
            file_bar.update(1)

        # Close progress bars
        file_bar.close()
        byte_bar.close()

    return unzipped_files
