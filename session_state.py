"""Session acquisition scaffolding."""

PACKETS_PER_SESSION = 8

session_signals = {
    "X": [],
    "Y": [],
    "Z": [],
}

sessions = []
stop_requested = False
