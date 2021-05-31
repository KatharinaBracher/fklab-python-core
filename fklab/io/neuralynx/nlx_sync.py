from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import h5py
import numpy as np

from .nlx_config import *
from .nlx_config import _ensure_nlx_event_file
from .nlx_config import _parse_ttl_input
from .nlx_config import CheetahConfig
from .nlx_config import EventInfo
from .nlx_config import SystemBoard

__all__ = [
    "nlx_start_recording_time",
    "nlx_stop_recording_time",
    "nlx_retrieve_event",
    "nlx_guess_sync_signal",
    "NlxEventInfo",
]


def nlx_start_recording_time(file: Union[str, Path, "NlxFileEvent"]) -> float:
    """Get the time stamp for system event "Starting Recording".

    Parameters
    ----------
    file
        Neuralynx Event file or instance

    Returns
    -------
    float
        time

    """
    return float(
        _nlx_retrieve_sys_event(_ensure_nlx_event_file(file), "Starting Recording")[
            0, 0
        ]
    )


def nlx_stop_recording_time(file: Union[str, Path, "NlxFileEvent"]) -> float:
    """Get the time stamp for system event "Stopping Recording".

    Parameters
    ----------
    file
        Neuralynx Event file or instance

    Returns
    -------
    float
        time

    """
    return float(
        _nlx_retrieve_sys_event(_ensure_nlx_event_file(file), "Stopping Recording")[
            0, 0
        ]
    )


def nlx_retrieve_event(
    file: Union[str, Path, "NlxFileEvent"],
    ttl_event: Union[int, Tuple[int, int], EventInfo, Tuple[CheetahConfig, str]] = None,
) -> np.ndarray:
    """Get event time from Neuralynx Events.nev file.

    **Example**

    Retrieve TTL event. It only work when this event file has a simple setup (only one TTL event in file).

    >>> nlx_retrieve_event(evt_file)

    Retrieve specific TTL event if user knows its bit mask.

    >>> nlx_retrieve_event(evt_file, 8) # port: ignore, bit: 3 -> mask: 8
    >>> nlx_retrieve_event(evt_file, (0, 3)) # port: 0, bit: 3

    Retrieve specific System event

    >>> nlx_retrieve_event(evt_file, 'My Event')

    Retrieve specific event by its name

    >>> nlx_retrieve_event(evt_file, (config, 'event_name'))
    >>> nlx_retrieve_event(evt_file, get_nlx_event_info(config, 'event_name')[0]) # same as above.

    Parameters
    ----------
    file
        Neuralynx Event file or instance
    ttl_event
        ttl event mask (value of 2**bit), tuple of (port, bit), the event information or the
        tuple of (CheetahConfig, event_name).

    Returns
    -------
    np.ndarray
        event time data, 2D matrix with shape (N, 2) contains (time_start, time_stop).
        For system event, time_stop equals to time_start.

    Raises
    ------
    RuntimeError
        When ttl_event is omitted and the Events.nev came from a complex setup that it is not
        able to infer a single TTL input's information.

    """
    nlx_event = _ensure_nlx_event_file(file)

    if ttl_event is None:
        return _nlx_retrieve_ttl_event(
            nlx_event, None, nlx_guess_sync_signal(nlx_event)
        )

    elif isinstance(ttl_event, int):  # ttl_event = bit_mask
        return _nlx_retrieve_ttl_event(nlx_event, None, ttl_event)

    elif isinstance(ttl_event, str):  # ttl_event = event_name
        return _nlx_retrieve_sys_event(nlx_event, ttl_event)

    elif isinstance(ttl_event, tuple):  # ttl_event = (port, bit) | (config, event_name)
        if len(ttl_event) != 2:
            raise ValueError("tuple length != 2")

        if isinstance(ttl_event[0], int) and isinstance(ttl_event[1], int):
            return _nlx_retrieve_ttl_event(
                nlx_event, ttl_event[0], int(2 ** ttl_event[1])
            )

        elif isinstance(ttl_event[0], dict) and isinstance(ttl_event[1], str):
            # ignore board name.
            ttl_event, _, _ = get_nlx_event_info(ttl_event[0], ttl_event[1])
            ttl_mask = ttl_event["bits"]
            if ttl_mask >= 0:
                return _nlx_retrieve_ttl_event(
                    nlx_event, ttl_event["port"], int(2 ** ttl_mask)
                )
            else:
                return _nlx_retrieve_sys_event(nlx_event, ttl_event["_original_name"])
        else:
            raise TypeError(f"cannot understand what inside it is : {ttl_event}")

    elif isinstance(ttl_event, dict):  # ttl_event = event_info = {bits, port}
        if "bits" in ttl_event and "port" in ttl_event:
            ttl_mask = ttl_event["bits"]
            if ttl_mask >= 0:
                return _nlx_retrieve_ttl_event(
                    nlx_event, ttl_event["port"], int(2 ** ttl_mask)
                )
            else:
                return _nlx_retrieve_sys_event(nlx_event, ttl_event["_original_name"])

        else:
            raise ValueError('not a EventInfo. lost one of key "port" or "bits"')

    raise TypeError()


def _nlx_retrieve_ttl_event(
    nlx_event, ttl_port: Optional[int], ttl_mask: int
) -> np.ndarray:
    """Retrieve TTL event from nlx_event

    Parameters
    ----------
    nlx_event : NlxFileEvent
    ttl_port
    ttl_mask

    Returns
    -------
    np.ndarray

    """
    event_time = nlx_event.data.time[:]
    event_ttl = nlx_event.data.nttl[:]
    event_str = nlx_event.data.eventstring[:]

    # go through each event.
    ret = []
    pulse = 0
    raising_time = -1
    for ti, (ttl_value, evt_str) in enumerate(
        zip(event_ttl, event_str)
    ):  # type: int, (int, bytes)
        if evt_str.startswith(b"TTL Input "):
            _, port, _ = _parse_ttl_input(evt_str.decode(), on_error="ignore")

            if ttl_port is not None and port is not None and ttl_port != port:
                continue

        ttl_value = ttl_value & ttl_mask
        if ttl_value > 0 and ttl_value != pulse:
            raising_time = event_time[ti]
            pulse = ttl_mask
        elif ttl_value == 0 and pulse != 0:
            ret.append((raising_time, event_time[ti]))
            pulse = 0

    return np.array(ret)


def _nlx_retrieve_sys_event(nlx_event, ttl_name: str) -> np.ndarray:
    """Retrieve System event from nlx_event

    Parameters
    ----------
    nlx_event : NlxFileEvent
    ttl_name

    Returns
    -------

    """
    ttl_name = ttl_name.encode()
    event_time = nlx_event.data.time[:]
    event_str = nlx_event.data.eventstring[:]

    ret = []
    for ti, evt_str in enumerate(event_str):  # type: int, (int, bytes)
        if evt_str == ttl_name:
            ttl_time = event_time[ti]
            ret.append((ttl_time, ttl_time))

    return np.array(ret)


def nlx_guess_sync_signal(file: Union[str, Path, "NlxFileEvent"]) -> int:
    """Infer a single TTL input's information from Events.nev.

    Parameters
    ----------
    file
        Neuralynx Event file or instance

    Returns
    -------
    int
        ttl event mask.

    Raises
    ------
    RuntimeError
        If the Events.nev came from a complex setup that it is not
        able to infer a single TTL input's information.

    """
    nlx_event = _ensure_nlx_event_file(file)
    event_str = nlx_event.data.eventstring[:]

    cnt_board = None
    cnt_port = None
    cnt_bits = None
    error = False

    for i, e in enumerate(event_str):  # type: int, bytes
        if e.startswith(b"TTL Input "):
            board, port, bits = _parse_ttl_input(e.decode())

            if cnt_board is None:
                cnt_board = board
                cnt_port = port
            else:
                if cnt_board != board or cnt_port != port:
                    error = True
                    break

            if bits is not None:
                if cnt_bits is None:
                    cnt_bits = bits
                elif cnt_bits != bits:
                    error = True
                    break

    if cnt_board is None or cnt_bits is None:
        raise RuntimeError(
            "There is no TTL input event in Events.nev, so it is not possible "
            "to infer correct TTL input source. Please provider a config file."
        )
    if error:
        raise RuntimeError(
            "Events.nev not came from a simple setup, so it is not possible "
            "to infer correct TTL input source. Please provider a config file."
        )

    return int(2 ** cnt_bits)


class NlxEventInfo:
    """This class handle event config, retrieve all of the events data from Events.nev and package them together.

    User can save it into disk and use it in latter data processing.

    Attributes
    ----------
    config : CheetahConfig
        Event config
    events : dict
        Event data, with shape {board_name:str -> {event_name: str -> np.ndarray}}.
    """

    def __init__(self, config, event=None):
        """Create NlxEventInfo.

        Parameters
        ----------
        config: Union[str, Path, CheetahConfig]
            Neuralynx config path (*.cfg), log file (*.log) or event config file (*.yaml) or CheetahConfig.
        event:  Union[str, Path, 'NlxFileEvent']
            Neuralynx event file (Events.nev) or NlxFileEvent.
        """
        if isinstance(config, str):
            config = Path(config)

        if isinstance(config, Path):
            if config.suffix == ".cfg":
                config = read_nlx_config(config)
                if event is not None:
                    config = read_nlx_event(event, config)
            elif config.suffix == ".txt":
                config = read_nlx_log(config)
                if event is not None:
                    config = read_nlx_event(event, config)
            elif config.suffix == '.yaml':
                config = load_nlx_config(config)
            else:
                raise RuntimeError(f"unknown config file type : {config.name}")

        if not isinstance(config, dict):
            raise TypeError(f"not a config dictionary : {config}")

        self.config: CheetahConfig = clone_config(config)
        self.events: Dict[str, Dict[str, np.ndarray]] = {}

        if event is not None:
            self.add_nlx_event(event)

    @property
    def start_recording_time(self) -> float:
        """Get the timestamp for event "Starting Recording".

        Returns
        -------
        float
            time
        """
        for events in self.events.values():
            if "Starting Recording" in events:
                return float(events["Starting Recording"][0, 0])
        return None

    @property
    def stop_recording_time(self) -> float:
        """Get the timestamp for event "Stopping Recording".

        Returns
        -------
        float
            time
        """
        for events in self.events.values():
            if "Stopping Recording" in events:
                return float(events["Stopping Recording"][0, 0])
        return None

    @property
    def recording_duration(self) -> float:
        """Duration of this session.

        It equals to `stop_recording_time - start_recording_time`.

        Returns
        -------
        float
            time

        """
        return self.stop_recording_time - self.start_recording_time

    @property
    def event_names(self) -> List[str]:
        """List all events.

        Returns
        -------
        list
            labels name list

        """
        ret = set()
        for events in self.events.values():
            ret.update(events.keys())

        return list(sorted(ret))

    def add_nlx_event(self, file):
        """Reset event data with file.

        Parameters
        ----------
        file:  Union[str, Path, 'NlxFileEvent']
            Neuralynx Event file or instance

        """
        self.events.clear()

        nlx_event = _ensure_nlx_event_file(file)

        for board_name, board_info in self.config.items():
            self.events[board_name] = d = {}
            for event_name, event_info in board_info["Events"].items():
                d[event_name] = nlx_retrieve_event(nlx_event, event_info)

    def rename_event(self, old_name: str, new_name: str):
        """Rename the event name.

        Parameters
        ----------
        old_name
            old event name or event's _original_name
        new_name
            new event name.

        Raises
        ------
        KeyError
            event name not found

        """
        for board_name, board_info in self.config.items():
            for name, event_info in board_info["Events"].items():
                if name == old_name or event_info["_original_name"] == old_name:
                    data = self.events[board_name][name]

                    del board_info["Events"][name]
                    board_info["Events"][new_name] = event_info

                    del self.events[board_name][name]
                    self.events[board_name][new_name] = data
                    return

        raise KeyError()

    def nlx_retrieve_event(self, event_name: str) -> np.ndarray:
        """Get event time from Neuralynx Events.nev file.

        Parameters
        ----------
        event_name
            event name.

        Returns
        -------
        np.ndarray
            event time data, 2D matrix with shape (N, 2) contains (time_start, time_stop).

        Raises
        ------
        KeyError
            event name not found

        """
        for board_name, board_info in self.config.items():
            for name, event_info in board_info["Events"].items():
                if name == event_name or event_info["_original_name"] == event_name:
                    return self.events[board_name][name]
        raise KeyError()

    def save(self, path: Union[str, Path]):
        """Save this object into '.h5' file.

        Parameters
        ----------
        path
            '.h5' file path.

        """
        with h5py.File(path, "w") as f:
            for board_name, board_info in self.config.items():
                g = f.create_group(board_name)
                self._save_board(g, board_info, self.events[board_name])

    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load data from '.h5' file.

        Parameters
        ----------
        path
            '.h5' file path.

        Returns
        -------
        NlxEventInfo

        """
        config = {}
        events = {}

        with h5py.File(path, "r") as f:
            for board_name in f:
                cls._load_board(
                    f[board_name],
                    config.setdefault(board_name, {}),
                    events.setdefault(board_name, {}),
                )

        ret = NlxEventInfo(config)
        ret.events = events

        return ret

    @classmethod
    def _save_board(
        cls, f: h5py.Group, config: SystemBoard, data: Dict[str, np.ndarray]
    ):
        f.attrs["DigitalIOInputScanDelay"] = config["DigitalIOInputScanDelay"]
        f.attrs["DigitalIOBitsPerPort"] = config["DigitalIOBitsPerPort"]

        ports = config["Ports"]
        f.create_dataset(
            "DigitalIOEventsEnabled",
            data=[
                1 if ports[p]["DigitalIOEventsEnabled"] else 0
                for p in range(len(ports))
            ],
        )
        f.create_dataset(
            "DigitalIOPortDirection",
            data=[
                1 if ports[p]["DigitalIOPortDirection"] == "output" else 0
                for p in range(len(ports))
            ],
        )
        f.create_dataset(
            "DigitalIOPulseDuration",
            data=[ports[p]["DigitalIOPulseDuration"] for p in range(len(ports))],
        )
        f.create_dataset(
            "DigitalIOUseStrobeBit",
            data=[
                1 if ports[p]["DigitalIOUseStrobeBit"] else 0 for p in range(len(ports))
            ],
        )

        g = f.create_group("Events")
        for name, event_info in config["Events"].items():
            if name in data:
                cls._save_event(g.create_group(name), event_info, data[name])

    @classmethod
    def _load_board(
        cls, f: h5py.Group, config: SystemBoard, data: Dict[str, np.ndarray]
    ):
        config["DigitalIOInputScanDelay"] = f.attrs["DigitalIOInputScanDelay"]
        config["DigitalIOBitsPerPort"] = f.attrs["DigitalIOBitsPerPort"]

        dee = list(f["DigitalIOEventsEnabled"])
        dpd = list(f["DigitalIOPortDirection"])
        dpt = list(f["DigitalIOPulseDuration"])
        dus = list(f["DigitalIOUseStrobeBit"])

        if not (len(dee) == len(dpd) == len(dpt) == len(dus)):
            raise RuntimeError("ports length not equals")

        ports = config.setdefault("Ports", {})
        for p in range(len(dee)):
            ports[p] = {
                "DigitalIOEventsEnabled": dee[p] > 0,
                "DigitalIOPortDirection": "output" if dpd[p] > 0 else "input",
                "DigitalIOPulseDuration": dpt[p],
                "DigitalIOUseStrobeBit": dus[p] > 0,
            }

        config_events = config.setdefault("Events", {})
        data_events = f["Events"]
        for event_name in data_events:
            data[event_name] = cls._load_event(
                data_events[event_name], config_events.setdefault(event_name, {})
            )

    @classmethod
    def _save_event(cls, f: h5py.Group, config: EventInfo, data: np.ndarray):
        f.attrs["_original_name"] = config["_original_name"]
        f.attrs["type"] = config["type"]
        f.attrs["direct"] = config["direct"]
        f.attrs["port"] = config["port"]
        f.attrs["bits"] = config["bits"]
        f.create_dataset("data", data=data)

    @classmethod
    def _load_event(cls, f: h5py.Group, config: EventInfo) -> np.ndarray:
        config["_original_name"] = f.attrs["_original_name"]
        config["type"] = f.attrs["type"]
        config["direct"] = f.attrs["direct"]
        config["port"] = f.attrs["port"]
        config["bits"] = f.attrs["bits"]
        return np.array(f["data"])
