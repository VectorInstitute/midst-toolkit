import platform


def is_apple_silicon():
    """Check if running on macOS with Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
