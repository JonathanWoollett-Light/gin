from . import CACHE as C, download as dtm, unzip
import os
from io import BytesIO
import numpy as np
from tqdm import tqdm
import numpy as np
import sparse
from tqdm import tqdm

SOURCE = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/468j46mzdv-1.zip"
TAG = "nmnist"
CACHE = os.path.join(C, TAG)
TEST = "Test.zip"
TRAIN = "Train.zip"
FEATURES = 28 * 28


class FrameData:
    test: np.ndarray
    train: np.ndarray

    def __init__(self, test: np.ndarray, train: np.ndarray):
        self.test = test
        self.train = train


class Event:
    x: int
    y: int
    polarity: bool
    time: int

    def __init__(self, x: int, y: int, polarity: bool, time: int):
        self.x = x
        self.y = y
        self.polarity = polarity
        self.time = time


class EventData:
    test: list[list[Event]]
    train: list[list[Event]]

    def __init__(self, test: list[list[Event]], train: list[list[Event]]):
        self.test = test
        self.train = train


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
    x = (value >> 32) & 0xFF  # Shift 32 bits right, mask 8 bits
    y = (value >> 24) & 0xFF  # Shift 24 bits right, mask 8 bits

    assert (
        0 <= x < 28 and 0 <= y < 28
    ), f"x: {x}, y: {y}, value: {value}, data: {[data[0],data[1],data[2],data[4]]}, all 8 bit samples: {[value >> i & 0xFF for i in range(0,40,8)]}"
    polarity = bool((value >> 23) & 1)  # Shift 23 bits right, mask 1 bit
    time = value & 0x7FFFFF  # Mask lower 23 bits
    return Event(x, y, polarity, time)


def to_events(data: dict[str, bytes], track: bool = True) -> list[list[Event]]:
    examples: list[list[Event]] = []
    with tqdm(total=len(data), desc="Framing", disable=not track) as bar:
        for value in data.values():
            # Assert each example is a multiple of 5 bytes (40 bits).
            assert len(value) % 5 == 0

            # Get events.
            events = [to_event(value[i : i + 5]) for i in range(0, len(value), 5)]

            # Sort by timestep.
            events.sort(key=lambda event: event.time)

            # Add example.
            examples.append(events)
            bar.update(1)
    return examples


def to_frames(data: dict[str, bytes], track: bool = True) -> sparse.COO:  # type: ignore
    # [time x sample x feature]
    n_samples = len(data)
    events_dict: dict[tuple[int, int, int], int] = {}  # Stores coordinates and values
    max_time = 0  # Tracks maximum time dimension

    with tqdm(total=len(data), desc="Framing", disable=not track) as bar:
        for sample_idx, value in enumerate(data.values()):
            # Assert each example is a multiple of 5 bytes (40 bits)
            assert len(value) % 5 == 0

            for i in range(0, len(value), 5):
                # Decode event
                event = to_event(value[i : i + 5])

                # Update max time dimension
                max_time = max(event.time + 1, max_time)

                # Calculate feature index
                assert 0 <= event.x < 28, f"{event.x}"
                assert 0 <= event.y < 28, f"{event.y}"
                feature_idx = event.x + event.y * 28
                assert feature_idx < FEATURES
                coord = (event.time, sample_idx, feature_idx)

                # Store only non-zero values (polarity 1)
                events_dict[coord] = 1 if event.polarity else -1

            bar.update(1)

    # Convert dictionary to sparse array
    coords = np.array(list(events_dict.keys())).T
    assert coords.shape == (
        3,
        len(events_dict),
    ), f"{coords.shape} != {(3, len(events_dict))}"
    data_vals = np.array(list(events_dict.values()))
    assert data_vals.shape == (
        len(events_dict),
    ), f"{data_vals.shape} != {(len(events_dict),)}"

    return sparse.COO(  # type: ignore
        coords=coords, data=data_vals, shape=(max_time, n_samples, FEATURES)
    )


class NMNIST:
    test: dict[str, bytes]
    train: dict[str, bytes]

    def __init__(self, test: dict[str, bytes], train: dict[str, bytes]):
        self.test = test
        self.train = train

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
