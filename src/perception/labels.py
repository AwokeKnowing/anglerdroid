"""Per-cell ego label codes (uint8)."""

UNKNOWN = 0
SELF = 1
CLEAR = 2
OBSTACLE = 3

LABEL_NAMES = {
    UNKNOWN: "unknown",
    SELF: "self",
    CLEAR: "clear",
    OBSTACLE: "obstacle",
}
