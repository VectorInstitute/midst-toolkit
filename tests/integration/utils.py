import os


def is_running_on_ci_environment() -> bool:
    """
    Check if running on a CI environment, particularly GitHub Actions.

    Returns:
        bool: True if running on a CI environment (Github Actions), False otherwise.
    """
    return os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
