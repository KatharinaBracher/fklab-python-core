"""
# Neuralynx Event Info

This module is used for parsing `Cheetah.cfg` or `CheetahLogFile.txt` to fetch
the TTL event setting, which is used for fetch the TTL event from `Events.nev`.

## Event Configuration

::

    AcqSystem1_0:                      # System board name
      DigitalIOBitsPerPort: 8               # bits per port
      DigitalIOInputScanDelay: 1            #
      Events:                               # event information
        custom event:                           # event name, use can change its name
          _original_name: custom event              # original event name recorded in file. shouldn't be changed.
          bits: 1                                   # bit/pin number (0~DigitalIOBitsPerPort-1)
          port: 0                                   # port number (0~3)
          direct: input                             # input/output
          type: user                                # event type
        Starting Recording:                     # system event
          _original_name: Starting Recording
          bits: -1                                  # system event doesn't use bits
          port: -1                                  # system event doesn't use port
          direct: input
          type: system
        Stopping Recording:                     # system event
          _original_name: Stopping Recording
          bits: -1
          port: -1
          direct: input
          type: system
        TTL Input on AcqSystem1_0 board 0 port 0 value (0x0040).:   # TTL event
          _original_name: TTL Input on AcqSystem1_0 board 0 port 0 value (0x0040).
          bits: 6
          port: 0
          direct: input
          type: user
      Ports:                                # port information
        0:                                      # port number
          DigitalIOEventsEnabled: true              #
          DigitalIOPortDirection: input             # input/output
          DigitalIOPulseDuration: 15                #
          DigitalIOUseStrobeBit: false              #
        1:
          DigitalIOEventsEnabled: true
          DigitalIOPortDirection: input
          DigitalIOPulseDuration: 15
          DigitalIOUseStrobeBit: false
        2:
          DigitalIOEventsEnabled: true
          DigitalIOPortDirection: input
          DigitalIOPulseDuration: 15
          DigitalIOUseStrobeBit: false
        3:
          DigitalIOEventsEnabled: true
          DigitalIOPortDirection: input
          DigitalIOPulseDuration: 15
          DigitalIOUseStrobeBit: false

User can modify this config by changing the name of the event (the event name
at key posistion, not the value of _original_name) to make the following
work more easily (??).

## Usage Example

### Use command line to get the config file from cheetah.cfg

    python -m fklab.io.neuralynx.nlx_config cheetah.cfg Events.nev

You can `Events.nev` here, but neither `Cheetah.cfg` nor `CheetahLogFile.txt`
contains the 'TTL Input ...' event, so you need to add other `Events.nev` file
to make config file more complete.

### Use in python

>>> from fklab.io.neuralynx.nlx_config import *
>>> cheetah_cfg_file = 'Cheetah.cfg'
>>> cheetah_log_file = 'CheetahLogFile.txt'
>>> cheetah_nev_file = 'Events.nev'

Create config

>>> config = read_nlx_config(cheetah_cfg_file)
>>> # config = read_nlx_log(cheetah_cfg_file) # either cfg or log
>>> config = read_nlx_event(cheetah_nev_file, config)

Save and laod

>>> save_nlx_config(config, 'event_config.yaml')
>>> config = load_nlx_config('event_config.yaml')

menipulate it. `config` is just a dictionary and its data structure
follow the `CheetahConfig` declaration.

>>> config['AcqSystem1_0']['Events'].keys()
dict_keys(['custom event', 'Starting Recording', 'Stopping Recording', 'TTL Input on AcqSystem1_0 board 0 port 0 value (0x0040).'])

### Create Config in python

create a config template with default value.

>>> config = create_default_config()

add an event with name 'My Event'.

>>> config['AcqSystem1_0']['Events']['My Event'] = create_nlx_event(
...     port=0,
...     bits=1,
...     name='My Event',
...     # default input direction and user type.
... )

You can use literal dict to create a EventInfo instead of calling function `create_nlx_event`,
but you need to provide all necessaey fields and value required by EventInfo declaration.

## Event

From Neuralynx Documents, section `4.2.3 TTL I/O Ports`

::

    The Digital Lynx provides 32 TTLs. They are configurable as inputs or outputs in four
    groups of eight. TTL I/O Ports 0 through 3 (Figure 4-9) represent the four groups of
    eight TTLs. When configured as outpus, the TTLs can be set high or low and pulsed high
    or low for pulse durations of 1 millisecond to 10,000 milliseconds. Port direction and
    other TTL settings are configured using the Cheetah DAS.

    The TTLs are accessible from two 34-pin connectors on the SX Motherboard fron panel.
    Each 34-pin port includes two 8-bit TTL ports (Figure 4-9).

For each TTL events, there are three value definie its source: board, port and bits. For
the 'custom event' in previous config as example, its source board is 'AcqSystem1_0',
port is 0 and bits is 1. You can see the similar things in un-named event, like
'TTL Input on AcqSystem1_0 board 0 port 0 value (0x0040).'.

All those information will become one bit in the TTL event in the `Events.nev` files. You
can use the bit mask (2 ** bits) to filter out the event in nttl field from the NlxFileEvent.

>>> import numpy as np
>>> from fklab.io.neuralynx import NlxOpen
>>> nlx_event = NlxOpen('Events.nev')
>>> np.unique(nlx_event.data.nttl[:])
[ 0  2  64 66]

There is some issue here, according to Neuralynx Documents of file Format[1]. The type of
`nttl` field is *int16* (16-bits), which is impossible to contains 4 ports 8-bits event value
(it needs at least 32 bits). ...

## Command Line Interface.

This module provide a simple cli let read a config or a log file and print the parsed config.

::

    $ python -m fklab.io.neuralynx,nlx_config Cheetah.cfg Events.nev > config.yaml

[1]: https://neuralynx.com/software/category/development

"""

import math
import sys
from pathlib import Path
from typing import Union, Tuple, Dict, TypedDict, Literal, IO, List

import yaml

__all__ = [
    # type
    'EventInfo',
    'PortInfo',
    'SystemBoard',
    'CheetahConfig',
    # IO functions
    'read_nlx_config',
    'read_nlx_log',
    'read_nlx_event',
    'save_nlx_config',
    'load_nlx_config',
    # config function
    'create_default_config',
    'clone_config',
    'create_nlx_event',
    'get_nlx_event_info',
    'get_event_mask',
]

EventInfo = TypedDict('EventInfo', {
    '_original_name': str,
    'type': Literal['system', 'user'],
    'direct': Literal['input', 'output'],
    'port': int,
    'bits': int,
})

PortInfo = TypedDict('PortInfo', {
    'DigitalIOPortDirection': Literal['input', 'output'],
    'DigitalIOEventsEnabled': bool,
    'DigitalIOPulseDuration': int,
    'DigitalIOUseStrobeBit': bool,
})

SystemBoard = TypedDict('SystemBoard', {
    'DigitalIOBitsPerPort': int,
    'DigitalIOInputScanDelay': int,
    'Ports': Dict[int, PortInfo],
    'Events': Dict[str, EventInfo]
})

CheetahConfig = Dict[str, SystemBoard]


def create_default_config(total_ports=4, *,
                          board_name: str = 'AcqSystem1_0',
                          bits_per_port=8,
                          scan_delay=1,
                          pulse_duration=15) -> CheetahConfig:
    """
    Create a default config data structure with default fields and value.

    Parameters
    ----------
    total_ports
        total ports number, default 4.
    board_name
        system board name, default 'AcqSystem1_0'
    bits_per_port
        default 8
    scan_delay
        default 1
    pulse_duration
        default 15

    Returns
    -------
    CheetahConfig

    """
    return {
        board_name: {
            'DigitalIOInputScanDelay': scan_delay,
            'DigitalIOBitsPerPort': bits_per_port,
            'Events': {},
            'Ports': {
                port: {
                    'DigitalIOEventsEnabled': True,
                    'DigitalIOPortDirection': 'input',
                    'DigitalIOPulseDuration': pulse_duration,
                    'DigitalIOUseStrobeBit': False
                }
                for port in range(total_ports)
            }
        }
    }


def clone_config(config: CheetahConfig) -> CheetahConfig:
    return {
        board_name: {
            'DigitalIOInputScanDelay': board['DigitalIOInputScanDelay'],
            'DigitalIOBitsPerPort': board['DigitalIOBitsPerPort'],
            'Events': {
                event_name: {
                    '_original_name': event['_original_name'],
                    'type': event['type'],
                    'direct': event['direct'],
                    'port': event['port'],
                    'bits': event['bits']
                } for event_name, event in board['Events'].items()
            },
            'Ports': {
                port_i: {
                    'DigitalIOEventsEnabled': port['DigitalIOEventsEnabled'],
                    'DigitalIOPortDirection': port['DigitalIOPortDirection'],
                    'DigitalIOPulseDuration': port['DigitalIOPulseDuration'],
                    'DigitalIOUseStrobeBit': port['DigitalIOUseStrobeBit']
                } for port_i, port in board['Ports'].items()
            }
        }
        for board_name, board in config.items()
    }


def create_nlx_event(port: int, bits: int, name: str,
                     type: Literal['user', 'system'] = 'user',
                     direct: Literal['input', 'output'] = 'input') -> EventInfo:
    """
    constructor-like function to create EventInfo.

    Parameters
    ----------
    port
    bits
    name
    type
        default 'user'
    direct
        default 'input'

    Returns
    -------
    EventInfo

    """
    return EventInfo(
        port=port,
        bits=bits,
        type=type,
        direct=direct,
        _original_name=name,
    )


def read_nlx_config(file: Union[str, Path], config: CheetahConfig = None) -> CheetahConfig:
    """
    read event information from Neuralynx/Cheetah config (*.cfg) file.

    Parameters
    ----------
    file
        config file path
    config
        overwrite/update to config.

    Returns
    -------
    CheetahConfig

    """
    if isinstance(file, str):
        file = Path(file)

    if config is None:
        config = {}

    with file.open('r') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or line.startswith('#'):
                continue

            _config_set_command(config, line)

    return config


def read_nlx_log(file: Union[str, Path], config: CheetahConfig = None) -> CheetahConfig:
    """
    read event information from Neuralynx/Cheetah log (CheetahLogFile.txt) file.

    Parameters
    ----------
    file
        log file path
    config
        overwrite/update to config.

    Returns
    -------
    CheetahConfig

    """
    if isinstance(file, str):
        file = Path(file)

    if config is None:
        config = {}

    # line pattern we care
    # -* NOTICE  *-  13:28:35.505 - 22799408 - FormatCmdLine::GetNextLine() - Processing line: ...
    # -* NOTICE  *-  13:28:35.505 - 22799408 - FormatCmdLine::GetNextLine() - Processing line NUMBER: ...
    N = '-* NOTICE  *-'
    F = 'FormatCmdLine::GetNextLine() - Processing line'

    with file.open('r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0 or line.startswith('#'):
                continue

            if not line.startswith(N) or F not in line:
                continue

            line = line[line.index(':', 80) + 1:].strip()
            _config_set_command(config, line)

    return config


def _config_set_command(config: CheetahConfig, line: str):
    """
    parse event relate command to config.

    Parameters
    ----------
    config
        update to config.
    line
        Cheetah command line.

    """

    def get_dict(s: str, *keys) -> Dict:
        d = config.setdefault(_unquote(s), {})
        for k in keys:
            d = d.setdefault(k, {})
        return d

    # take the command we care
    if line.startswith("-SetDigitalIOBitsPerPort"):
        # -SetDigitalIOBitsPerPort "AcqSystem1_0" VALUE
        p = line.split(' ', 2)
        get_dict(p[1])['DigitalIOBitsPerPort'] = int(p[2])

    elif line.startswith('-SetDigitalIOInputScanDelay'):
        # -SetDigitalIOInputScanDelay "AcqSystem1_0" VALUE
        p = line.split(' ', 2)
        get_dict(p[1])['DigitalIOInputScanDelay'] = int(p[2])

    elif line.startswith('-SetDigitalIOPortDirection'):
        # -SetDigitalIOPortDirection "AcqSystem1_0" PORT Input/Output
        p = line.split(' ', 3)
        get_dict(p[1], 'Ports', int(p[2]))['DigitalIOPortDirection'] = p[3].lower()

    elif line.startswith('-SetDigitalIOUseStrobeBit'):
        # -SetDigitalIOUseStrobeBit "AcqSystem1_0" PORT True/False
        p = line.split(' ', 3)
        get_dict(p[1], 'Ports', int(p[2]))['DigitalIOUseStrobeBit'] = p[3] != 'False'

    elif line.startswith('-SetDigitalIOEventsEnabled'):
        # -SetDigitalIOEventsEnabled "AcqSystem1_0" PORT True/False
        p = line.split(' ', 3)
        get_dict(p[1], 'Ports', int(p[2]))['DigitalIOEventsEnabled'] = p[3] != 'False'

    elif line.startswith('-SetDigitalIOPulseDuration'):
        # -SetDigitalIOPulseDuration "AcqSystem1_0" PORT VALUE
        p = line.split(' ', 3)
        get_dict(p[1], 'Ports', int(p[2]))['DigitalIOPulseDuration'] = int(p[3])

    elif line.startswith('-SetNamedTTLEvent'):
        # -SetNamedTTLEvent "AcqSystem1_0" PORT BITS "NAME"
        p = line.split(' ', 4)

        port = int(p[2])
        bits = int(p[3])
        e = _unquote(p[4])
        direct = get_dict(p[1], 'Ports', port).get('DigitalIOPortDirection', 'input')
        get_dict(p[1], 'Events')[e] = create_nlx_event(port, bits, e, 'user', direct)


def read_nlx_event(file: Union[str, Path, 'NlxFileEvent'], config: CheetahConfig = None) -> CheetahConfig:
    """
    read event information from Neuralynx Event file (Events.nev).

    This method will read and set the lost information from event file,
    includes system event (Starting/Stopping Recording) and TTL Input
    events.

    Parameters
    ----------
    file
        Neuralynx event file or instance.
    config
        overwrite/update to config.

    Returns
    -------
    CheetahConfig

    """
    nlx_event = _ensure_nlx_event_file(file)

    if config is None:
        config = create_default_config()

    event_str = nlx_event.data.eventstring[:]

    for e_str in event_str:  # type: bytes
        e_str: str = e_str.decode()
        try:
            get_nlx_event_info(config, e_str)
            continue
        except ValueError:
            pass

        if e_str.startswith('TTL Input '):
            board, port, bits = _parse_ttl_input(e_str, on_error='warning')
            if board is not None:
                try:
                    board = config[board]
                except KeyError as e:
                    raise RuntimeError(f'board {board} not found in config, '
                                       f'make sure this Events.nev is generated under this config') from e
                else:
                    if e_str not in board['Events'] and port is not None and bits is not None:
                        board['Events'][e_str] = create_nlx_event(port, bits, e_str, 'user')
        else:
            for board in config.values():
                board['Events'][e_str] = create_nlx_event(-1, -1, e_str, 'system')

    return config


def _parse_ttl_input(e_str: str,
                     on_error: Literal['error', 'warning', 'ignore'] = 'error') -> Tuple[str, int, int]:
    """
    parse TTL Input event string.

    There is one TTL Input event string example::

         TTL Input on AcqSystem1_0 board 0 port 0 value (0x0040).

    Parameters
    ----------
    e_str
        event string
    on_error : {'error', 'warning', 'ignore'}
        print warning for bad event string instead of raising error.
        If 'ignore', the return values would be all None.

    Returns
    -------
    str
        system board name
    int
        port number
    int
        bits number. -1 for failing edge event.

    Raises
    ------
    RuntimeError
        bad event string or mask value.

    """
    p = e_str.split(' ')

    board = None
    port = None
    bits = None

    try:
        board = p[3]

        port = int(p[7])
        mask = int(p[9][1:-2], 16)

        if mask > 0:
            bits = math.log2(mask)
            if bits - int(bits) != 0:
                if on_error == 'ignore':
                    pass
                elif on_error == 'warning':
                    print(f'bad mask value : {mask} for event : {e_str}')
                else:
                    raise RuntimeError(f'bad mask value : {mask} for event : {e_str}')

            bits = int(bits)
        else:
            bits = None

    except ValueError as e:
        if on_error == 'ignore':
            pass
        elif on_error == 'warning':
            print(f'bad value : {e}')
        else:
            raise e

    except IndexError as e:
        if on_error == 'ignore':
            pass
        elif on_error == 'warning':
            print(f'bad format : {e}')
        else:
            raise e

    return board, port, bits


def save_nlx_config(config: CheetahConfig, file: Union[str, Path, IO] = None):
    """
    save config in yaml-format file.

    Parameters
    ----------
    config
    file
        file path, stdout if omitted.

    """
    if file is None:
        file = sys.stdout
    elif isinstance(file, str):
        with open(file, 'w') as f:
            return save_nlx_config(config, f)
    elif isinstance(file, Path):
        with file.open('w') as f:
            return save_nlx_config(config, f)

    yaml.dump(config, file)


def load_nlx_config(file: Union[str, Path, IO]) -> CheetahConfig:
    """
    load config file from disk.

    Parameters
    ----------
    file
        file path

    Returns
    -------
    CheetahConfig

    """
    if isinstance(file, str):
        with open(file, 'r') as f:
            return load_nlx_config(f)
    elif isinstance(file, Path):
        with file.open('r') as f:
            return load_nlx_config(f)

    return yaml.safe_load(file)


def _ensure_nlx_event_file(file: Union[str, Path, 'NlxFileEvent']) -> 'NlxFileEvent':
    from fklab.io.neuralynx import NlxOpen, NlxFileEvent

    if isinstance(file, NlxFileEvent):
        return file

    elif isinstance(file, (str, Path)):
        # noinspection PyProtectedMember
        return NlxOpen(str(file))
    else:
        raise TypeError(f"not an event file : {file}")


def get_nlx_event_info(config: CheetahConfig, event_name: str) -> Tuple[EventInfo, str, str]:
    """
    get event info from config by name.

    This method will try to find the first name matched (include name and _original_name) event.

    Parameters
    ----------
    config
    event_name
        event name.

    Returns
    -------
    EventInfo
        event info dictionary.
    str
        event name
    str
        system board name

    Raises
    ------
    ValueError
        event not found.

    """
    for board, info in config.items():
        events = info['Events']

        if event_name in events:
            return events[event_name], event_name, board

        for name, event in events.items():
            if event['_original_name'] == event_name:
                return event, name, board

    raise ValueError(f'event {event_name} not found in config')


def get_event_mask(board: SystemBoard) -> Dict[int, str]:
    """
    get mask values for all events in config.

    Parameters
    ----------
    board
        system board information.

    Returns
    -------
    dict
        dictionary of mask value to event name

    """
    return {
        int(2 ** info['bits']): name
        for name, info in board['Events'].items()
        if info['bits'] >= 0
    }


def _unquote(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    else:
        return s


def _main(args: List[str]):
    f = args[0]
    if f.endswith('.cfg'):
        r = read_nlx_config(f)
    elif f.endswith('.txt'):
        r = read_nlx_log(f)
    elif f.endswith('.nev'):
        r = read_nlx_event(f)
    else:
        raise ValueError(f'unknown file type : {f}')

    for f in args[1:]:
        r = read_nlx_event(f, r)

    save_nlx_config(r)


if __name__ == '__main__':
    _main(sys.argv[1:])
