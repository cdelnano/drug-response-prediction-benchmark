import re


def sanitize_treatment_id(treatment_id: str) -> str:
    """Return a treatment ID that is safe to use as a directory name."""

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        treatment_id,
    )
