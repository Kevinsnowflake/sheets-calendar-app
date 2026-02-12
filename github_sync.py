"""Push file changes to GitHub so they persist across Streamlit Cloud redeployments.

Uses the GitHub Contents API.  Requires two Streamlit secrets:
    GITHUB_TOKEN  – a fine-grained PAT with Contents read/write permission
    GITHUB_REPO   – owner/repo, e.g. "Kevinsnowflake/sheets-calendar-app"

If either secret is missing the helper functions silently do nothing, so local
development is completely unaffected.
"""

import base64
import logging
from pathlib import Path

import requests
import streamlit as st

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_credentials() -> tuple[str, str] | None:
    """Return (token, repo) from Streamlit secrets, or None if not configured."""
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
    except Exception:
        return None
    if not token or not repo:
        return None
    return token, repo


def _api_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_file_sha(token: str, repo: str, repo_path: str) -> str | None:
    """Fetch the current SHA of a file in the repo (needed for updates).

    Returns None if the file doesn't exist yet (i.e. it will be created).
    """
    url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    resp = requests.get(url, headers=_api_headers(token), timeout=15)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def push_file_to_github(
    file_path: str | Path,
    commit_message: str = "Update file via app",
) -> bool:
    """Push a local file to the GitHub repo.

    *file_path* is relative to the app directory (e.g. ``config.json`` or
    ``data/events.csv``).  The file must already exist on disk.

    Returns True on success, False on failure (logged, never raises).
    """
    creds = _get_credentials()
    if creds is None:
        return False  # secrets not configured – silent skip
    token, repo = creds

    local_path = APP_DIR / file_path
    if not local_path.exists():
        logger.warning("push_file_to_github: file not found: %s", local_path)
        return False

    repo_path = str(Path(file_path))  # normalise separators
    content_b64 = base64.b64encode(local_path.read_bytes()).decode()

    sha = _get_file_sha(token, repo, repo_path)

    url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    body: dict = {
        "message": commit_message,
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha  # required when updating an existing file

    resp = requests.put(url, json=body, headers=_api_headers(token), timeout=30)
    if resp.status_code in (200, 201):
        logger.info("Pushed %s to GitHub (%s)", repo_path, resp.status_code)
        return True

    logger.error(
        "GitHub push failed for %s: %s %s",
        repo_path, resp.status_code, resp.text[:300],
    )
    return False


def delete_file_from_github(
    file_path: str | Path,
    commit_message: str = "Delete file via app",
) -> bool:
    """Delete a file from the GitHub repo.

    Returns True on success, False on failure (logged, never raises).
    """
    creds = _get_credentials()
    if creds is None:
        return False
    token, repo = creds

    repo_path = str(Path(file_path))
    sha = _get_file_sha(token, repo, repo_path)
    if sha is None:
        # File doesn't exist in the repo – nothing to delete
        return True

    url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    body = {
        "message": commit_message,
        "sha": sha,
    }
    resp = requests.delete(url, json=body, headers=_api_headers(token), timeout=30)
    if resp.status_code == 200:
        logger.info("Deleted %s from GitHub", repo_path)
        return True

    logger.error(
        "GitHub delete failed for %s: %s %s",
        repo_path, resp.status_code, resp.text[:300],
    )
    return False
