"""Time formatting helpers."""


def format_elapsed_time(seconds):
    """Format elapsed time in hours, minutes, and seconds for time measurements.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
