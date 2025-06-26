from . import CACHE as C, download as dtm, unzip
import os
from io import BytesIO
import numpy as np
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Callable
from tempfile import NamedTemporaryFile

import numpy.typing as npt
import sparse
from tqdm import tqdm
from numpy import int8, int32, int64

SOURCE = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/468j46mzdv-1.zip"
TAG = "nmnist"
CACHE = os.path.join(C, TAG)
TEST = "Test.zip"
TRAIN = "Train.zip"
FEATURE_DIMENSION = 34
FEATURES = FEATURE_DIMENSION * FEATURE_DIMENSION
MS_TIMESTEPS = 337  # Maximum timestep of datatset in milliseconds


@dataclass
class SparseFrameData:
    test: sparse.COO
    train: sparse.COO


@dataclass
class FrameData:
    test: npt.NDArray[int8]
    train: npt.NDArray[int8]


@dataclass
class Event:
    x: int8
    y: int8
    polarity: bool
    time: int64


@dataclass
class EventData:
    test: list[list[Event]]
    train: list[list[Event]]


def to_event(data: bytes) -> Event:
    """Parses 40-bit event data into structured components.

    Args:
        data: 5-byte input (40 bits) containing:
            bits 39-32: Xaddress (8 bits)
            bits 31-24: Yaddress (8 bits)
            bit 23:     Polarity (1 bit)
            bits 22-0:  Timestamp (23 bits)

    Returns:
        tuple[int, int, bool, int]: (Xaddress, Yaddress, Polarity, Timestamp)
    """
    # Assert is 5 bytes (40 bits).
    assert len(data) == 5

    # Convert bytes to integer (big-endian)
    value = int.from_bytes(data, byteorder="big")

    # Extract components using bitwise operations
    x = int8((value >> 32) & 0xFF)  # Shift 32 bits right, mask 8 bits
    y = int8((value >> 24) & 0xFF)  # Shift 24 bits right, mask 8 bits

    assert (
        0 <= x < FEATURE_DIMENSION and 0 <= y < FEATURE_DIMENSION
    ), f"x: {x}, y: {y}, value: {value}, data: {[data[0],data[1],data[2],data[4]]}, all 8 bit samples: {[value >> i & 0xFF for i in range(0,40,8)]}"
    polarity = bool((value >> 23) & 1)  # Shift 23 bits right, mask 1 bit
    time = int64(value & 0x7FFFFF)  # Mask lower 23 bits
    return Event(x, y, polarity, time)


def to_events(data: dict[str, bytes], track: bool = True) -> list[list[Event]]:
    examples: list[list[Event]] = []
    with tqdm(total=len(data), desc="Listing", disable=not track) as bar:
        for value in data.values():
            # Assert each example is a multiple of 5 bytes (40 bits).
            assert len(value) % 5 == 0

            # Get events.
            events = [to_event(value[i : i + 5]) for i in range(0, len(value), 5)]

            # Sort by timestep.
            events.sort(key=lambda event: int(event.time))

            # Add example.
            examples.append(events)
            bar.update(1)
    return examples


# When using frames we discretize timesteps over milliseconds instead of microseconds to be able to handle the dataset given memory limitations.
def to_frames(
    data: dict[str, bytes], track: bool = True, memory_bytes: int = 1024**3
) -> npt.NDArray[int8]:
    # [time x sample x features..]
    size: Callable[[int], tuple[int, int, int, int]] = lambda n: (
        MS_TIMESTEPS,
        n,
        FEATURE_DIMENSION,
        FEATURE_DIMENSION,
    )
    # The number of samples to store in memory to remain under `memory_bytes`
    memory_samples = memory_bytes // (
        MS_TIMESTEPS * FEATURE_DIMENSION * FEATURE_DIMENSION
    )
    print(f"memory_samples: {memory_samples}")
    mem_events: npt.NDArray[int8] = np.zeros(
        shape=size(memory_samples), dtype=int8
    )  # TODO Why can't I use `np.empty` here?
    dsk_events: npt.NDArray[int8] = np.memmap(
        filename=NamedTemporaryFile(), dtype=int8, mode="w+", shape=size(len(data))
    )
    data_list: list[bytes] = list(data.values())

    max_ts = 0
    with tqdm(total=len(data), desc="Framing", disable=not track) as bar:
        for chunk in range(0, len(data) // memory_samples):
            mem_events.fill(0)
            start, stop = chunk * memory_samples, (chunk + 1) * memory_samples
            for sample_idx, value in enumerate(data_list[start:stop]):
                # Assert each example is a multiple of 5 bytes (40 bits).
                assert len(value) % 5 == 0

                for i in range(0, len(value), 5):
                    # Get event.
                    event = to_event(value[i : i + 5])

                    # Set event.
                    assert event.time // 1000 < MS_TIMESTEPS, f"{event.time // 1000}"
                    if event.time // 1000 > max_ts:
                        max_ts = event.time // 1000

                    mem_events[event.time // 1000, sample_idx, event.x, event.y] = (
                        1 if event.polarity else -1
                    )
                bar.update(1)
            dsk_events[:, start:stop, :, :] = mem_events
            dsk_events.flush()
    print(f"found max: {max_ts}")
    return dsk_events


# When using frames we discretize timesteps over milliseconds instead of microseconds to be able to handle the dataset given memory limitations.
def to_sparse_frames(data: dict[str, bytes], track: bool = True) -> sparse.COO:
    # [time x sample x feature]
    n_samples = len(data)
    events_dict: dict[tuple[int64, int32, int8, int8], int] = (
        {}
    )  # Stores coordinates and values

    with tqdm(total=len(data), desc="Framing", disable=not track) as bar:
        for sample_idx, value in enumerate(data.values()):
            # Assert each example is a multiple of 5 bytes (40 bits)
            assert len(value) % 5 == 0

            for i in range(0, len(value), 5):
                # Decode event
                event = to_event(value[i : i + 5])

                # Calculate feature index
                assert 0 <= event.x < FEATURE_DIMENSION, f"{event.x}"
                assert 0 <= event.y < FEATURE_DIMENSION, f"{event.y}"
                coord = (event.time // 1000, int32(sample_idx), event.x, event.y)

                # Store only non-zero values (polarity 1)
                events_dict[coord] = 1 if event.polarity else -1

            bar.update(1)

    # Convert dictionary to sparse array
    coords = np.array(list(events_dict.keys())).T
    assert coords.shape == (
        4,
        len(events_dict),
    ), f"{coords.shape} != {(4, len(events_dict))}"
    data_vals = np.array(list(events_dict.values()))
    assert data_vals.shape == (
        len(events_dict),
    ), f"{data_vals.shape} != {(len(events_dict),)}"

    return sparse.COO(
        coords=coords,
        data=data_vals,
        shape=(MS_TIMESTEPS, n_samples, FEATURE_DIMENSION, FEATURE_DIMENSION),
    )


@dataclass
class NMNIST:
    test: dict[str, bytes]
    train: dict[str, bytes]

    def sparse_frames(self, track: bool = True) -> SparseFrameData:
        test = to_sparse_frames(self.test, track)
        train = to_sparse_frames(self.train, track)
        return SparseFrameData(test, train)

    def frames(self, track: bool = True) -> FrameData:
        test = to_frames(self.test, track)
        train = to_frames(self.train, track)
        return FrameData(test, train)

    def events(self, track: bool = True) -> EventData:
        test = to_events(self.test, track)
        train = to_events(self.train, track)
        return EventData(test, train)


def download(track: bool = True):
    """
    Downloads the data into memory

    Returns:
        io.BytesIO: Buffer containing the downloaded data.
    """
    return dtm(SOURCE, track=track)


def nmnist(cache: bool = True, track: bool = True) -> NMNIST:
    # Get raw data.
    if cache:
        if os.path.isfile(CACHE):
            with open(CACHE, "rb") as file:
                bytes = file.read()
                buffer = BytesIO(bytes)
        else:
            buffer = download(track=track)
            with open(CACHE, "wb") as file:
                buffer.seek(0)
                file.write(buffer.read())
        buffer.seek(0)
    else:
        buffer = download(track=track)

    # Unzip data.
    files = unzip(buffer, track=track)

    # Unzip test and train data.
    test, train = files.get(TEST), files.get(TRAIN)
    assert test and train
    test, train = unzip(BytesIO(test)), unzip(BytesIO(train))
    return NMNIST(test, train)
