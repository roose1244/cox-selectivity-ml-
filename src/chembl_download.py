"""Robust retrieval of ChEMBL target and activity data.

This module is the only place in the project that talks to the ChEMBL REST API.
It is deliberately conservative: every request has a timeout, every response is
checked with ``raise_for_status``, failures are retried with exponential
backoff, and pagination continues until ChEMBL says there is no next page.

Nothing here filters or transforms the data. The tables written to
``data/raw/`` are the unmodified records as ChEMBL returned them, so that all
curation decisions live in ``src/curate.py`` and stay reproducible.

Run as a script to download both COX targets:

    python src/chembl_download.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "COX_Selectivity_ML/0.1 (educational cheminformatics project)"

# Human cyclooxygenase targets. Verified against the live API by
# ``fetch_target`` before use rather than trusted blindly.
COX1_TARGET_ID = "CHEMBL221"  # PTGS1, prostaglandin G/H synthase 1
COX2_TARGET_ID = "CHEMBL230"  # PTGS2, prostaglandin G/H synthase 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_TIMEOUT = 60
DEFAULT_PAGE_SIZE = 200
DEFAULT_MAX_ATTEMPTS = 5


def make_session() -> requests.Session:
    """Return a requests session that identifies this project to ChEMBL."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict:
    """GET a JSON document, retrying transient failures with backoff.

    Raises the final exception if every attempt fails, so a partial download
    can never be mistaken for a complete one.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            wait = 2 ** (attempt - 1)
            print(
                f"  request failed ({type(error).__name__}), "
                f"attempt {attempt}/{max_attempts}; retrying in {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(f"Giving up on {url} after {max_attempts} attempts") from last_error


def fetch_status(session: requests.Session) -> dict:
    """Return the ChEMBL service status, including the database release."""
    return get_json(session, f"{BASE_URL}/status.json")


def fetch_target(session: requests.Session, target_chembl_id: str) -> dict:
    """Return the live target record for a ChEMBL target ID."""
    return get_json(session, f"{BASE_URL}/target/{target_chembl_id}.json")


def describe_target(target: dict) -> str:
    """One-line human-readable summary of a target record."""
    accessions = ", ".join(
        component.get("accession") or "?"
        for component in target.get("target_components", [])
    )
    return (
        f"{target['target_chembl_id']}: {target['pref_name']} "
        f"[{target['organism']}, tax_id {target['tax_id']}, "
        f"{target['target_type']}, UniProt {accessions}]"
    )


def fetch_activities(
    session: requests.Session,
    target_chembl_id: str,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[dict], dict]:
    """Retrieve every activity record for one target.

    Returns the list of records and the query parameters used, so the exact
    query can be recorded alongside the saved data. The expected total comes
    from ChEMBL's own ``page_meta`` on the first page; it is never hard-coded.
    """
    url = f"{BASE_URL}/activity.json"
    params = {"target_chembl_id": target_chembl_id, "limit": page_size, "offset": 0}

    records: list[dict] = []
    offset = 0
    total_count: int | None = None

    while True:
        page_params = dict(params, offset=offset)
        payload = get_json(session, url, params=page_params)

        if total_count is None:
            total_count = payload["page_meta"]["total_count"]
            print(
                f"ChEMBL reports {total_count:,} activity records for "
                f"{target_chembl_id}; downloading in pages of {page_size}"
            )

        batch = payload["activities"]
        records.extend(batch)
        print(f"  retrieved {len(records):,} / {total_count:,}")

        if not payload["page_meta"].get("next") or not batch:
            break
        offset += page_size

    if total_count is not None and len(records) != total_count:
        raise RuntimeError(
            f"Downloaded {len(records):,} records for {target_chembl_id} but "
            f"ChEMBL reported {total_count:,}; refusing to save a partial table"
        )

    query = {
        "endpoint": url,
        "params": {k: v for k, v in params.items() if k != "offset"},
        "pagination": f"offset stepped by limit={page_size} until page_meta.next was null",
        "reported_total_count": total_count,
    }
    return records, query


def download_target(
    session: requests.Session,
    target_chembl_id: str,
    label: str,
    raw_dir: Path = RAW_DIR,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """Download one target's activities and write the raw table to disk.

    Returns a provenance dictionary describing exactly what was fetched.
    """
    target = fetch_target(session, target_chembl_id)
    print(describe_target(target))

    records, query = fetch_activities(session, target_chembl_id, page_size=page_size)

    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    activities_path = raw_dir / f"chembl_activities_{label}_{target_chembl_id}.csv.gz"
    frame.to_csv(activities_path, index=False, compression="gzip")

    target_path = raw_dir / f"chembl_target_{label}_{target_chembl_id}.json"
    target_path.write_text(json.dumps(target, indent=2, sort_keys=True))

    print(f"  saved {len(frame):,} rows x {frame.shape[1]} columns -> {activities_path.name}")

    return {
        "label": label,
        "target_chembl_id": target_chembl_id,
        "target_pref_name": target["pref_name"],
        "target_organism": target["organism"],
        "target_tax_id": target["tax_id"],
        "target_type": target["target_type"],
        "uniprot_accessions": [
            c.get("accession") for c in target.get("target_components", [])
        ],
        "query": query,
        "rows_saved": int(len(frame)),
        "columns_saved": sorted(frame.columns),
        "activities_file": activities_path.name,
        "target_file": target_path.name,
    }


def main(raw_dir: Path = RAW_DIR, page_size: int = DEFAULT_PAGE_SIZE) -> dict:
    """Download both COX targets and write a provenance record."""
    session = make_session()

    status = fetch_status(session)
    print(
        f"ChEMBL release {status['chembl_db_version']} "
        f"(release date {status['chembl_release_date']}), service {status['status']}"
    )

    targets = [
        download_target(session, COX1_TARGET_ID, "cox1", raw_dir, page_size),
        download_target(session, COX2_TARGET_ID, "cox2", raw_dir, page_size),
    ]

    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chembl_db_version": status["chembl_db_version"],
        "chembl_release_date": status["chembl_release_date"],
        "chembl_total_activities_in_release": status["activities"],
        "base_url": BASE_URL,
        "user_agent": USER_AGENT,
        "downloader": "src/chembl_download.py",
        "notes": (
            "Raw ChEMBL activity records, unmodified apart from being written as "
            "a gzipped CSV table. No filtering on activity type, relation, units "
            "or organism happens here; see src/curate.py for curation."
        ),
        "targets": targets,
    }

    provenance_path = raw_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(f"\nProvenance written to {provenance_path}")
    return provenance


if __name__ == "__main__":
    main()
