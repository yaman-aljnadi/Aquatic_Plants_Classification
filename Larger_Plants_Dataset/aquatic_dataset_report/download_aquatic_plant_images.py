#!/usr/bin/env python3
"""
download_aquatic_plant_images.py

Download images listed in:
    1) gbif_samples.csv
    2) inaturalist_samples.csv

The script:
- accepts either or both CSVs
- detects the relevant URL/species/license columns
- keeps GBIF and iNaturalist source information
- optionally downloads only records whose CSV says the license looks open
- avoids duplicate downloads by URL and, where possible, iNaturalist photo ID
- uses retries + timeouts
- writes a detailed download_manifest.csv
- stores images in a species-based folder structure

Recommended folder layout:

project/
├── gbif_samples.csv
├── inaturalist_samples.csv
└── download_aquatic_plant_images.py

Run:
    python download_aquatic_plant_images.py

Safer licensing-focused run:
    python download_aquatic_plant_images.py --open-only

Use a small test first:
    python download_aquatic_plant_images.py --limit 20

GBIF only:
    python download_aquatic_plant_images.py --gbif-only

iNaturalist only:
    python download_aquatic_plant_images.py --inat-only
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_GBIF = "gbif_samples.csv"
DEFAULT_INAT = "inaturalist_samples.csv"
DEFAULT_OUTPUT = "aquatic_plant_images"

REQUEST_TIMEOUT = (15, 60)  # connect timeout, read timeout
CHUNK_SIZE = 1024 * 256     # 256 KB
SLEEP_BETWEEN_DOWNLOADS = 0.10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_name(value: object) -> str:
    """Make a safe Windows-friendly folder/file component."""
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:180] or "unknown"


def make_session() -> requests.Session:
    """Create a requests session with retry handling."""
    session = requests.Session()

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": (
            "AquaticPlantDatasetDownloader/1.0 "
            "(research dataset collection; Python requests)"
        )
    })

    return session


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash for a downloaded file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_inat_photo_id(url: str) -> Optional[str]:
    """
    Extract a likely iNaturalist photo ID from URLs such as:
      .../photos/743562794/square.jpg
      .../photos/604593364/original.jpg
    """
    if not url:
        return None

    match = re.search(r"/photos/(\d+)/", url)
    return match.group(1) if match else None


def extension_from_url(url: str, content_type: str = "") -> str:
    """Choose a reasonable image extension."""
    path = urlparse(url).path.lower()

    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext

    # Fall back to the server Content-Type.
    content_type = content_type.lower().split(";")[0].strip()
    guessed = mimetypes.guess_extension(content_type)
    if guessed:
        return ".jpg" if guessed == ".jpe" else guessed

    return ".jpg"


def choose_unique_filename(
    species_dir: Path,
    species: str,
    row_number: int,
    source: str,
    url: str,
    photo_id: Optional[str],
) -> Path:
    """
    Build deterministic filenames.
    Photo ID is preferred because it is stable for iNaturalist images.
    """
    identifier = photo_id or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return species_dir / f"{clean_name(species)}_{source}_{identifier}.jpg"


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------

def load_gbif(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"species", "image_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"GBIF file is missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame({
        "source": "GBIF",
        "species": df["species"].astype(str).str.strip(),
        "image_url": df["image_url"].astype(str).str.strip(),
        "license": df["image_license"] if "image_license" in df else "",
        "license_looks_open": (
            df["license_looks_open"]
            if "license_looks_open" in df
            else pd.NA
        ),
        "rights_holder": (
            df["rights_holder"] if "rights_holder" in df else ""
        ),
        "creator": df["creator"] if "creator" in df else "",
        "country": df["country"] if "country" in df else "",
        "state_province": (
            df["state_province"] if "state_province" in df else ""
        ),
        "lat": df["lat"] if "lat" in df else pd.NA,
        "lon": df["lon"] if "lon" in df else pd.NA,
        "event_date": (
            df["event_date"] if "event_date" in df else ""
        ),
        "gbif_occurrence_key": (
            df["gbif_occurrence_key"]
            if "gbif_occurrence_key" in df
            else ""
        ),
        "observation_id": "",
        "photo_id": "",
    })

    return out


def load_inat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"species", "photo_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"iNaturalist file is missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame({
        "source": "iNaturalist",
        "species": df["species"].astype(str).str.strip(),
        "image_url": df["photo_url"].astype(str).str.strip(),
        "license": (
            df["photo_license_code"]
            if "photo_license_code" in df else ""
        ),
        "license_looks_open": (
            df["license_looks_open"]
            if "license_looks_open" in df else pd.NA
        ),
        "rights_holder": "",
        "creator": "",
        "country": "",
        "state_province": "",
        "lat": df["lat"] if "lat" in df else pd.NA,
        "lon": df["lon"] if "lon" in df else pd.NA,
        "event_date": (
            df["observed_on"] if "observed_on" in df else ""
        ),
        "gbif_occurrence_key": "",
        "observation_id": (
            df["observation_id"]
            if "observation_id" in df else ""
        ),
        "photo_id": (
            df["photo_id"] if "photo_id" in df
            else df["photo_url"].astype(str).map(extract_inat_photo_id)
        ),
    })

    return out


def build_record_table(
    gbif_path: Optional[Path],
    inat_path: Optional[Path],
) -> pd.DataFrame:
    frames = []

    if gbif_path:
        if not gbif_path.exists():
            raise FileNotFoundError(f"GBIF file not found: {gbif_path}")
        frames.append(load_gbif(gbif_path))

    if inat_path:
        if not inat_path.exists():
            raise FileNotFoundError(
                f"iNaturalist file not found: {inat_path}"
            )
        frames.append(load_inat(inat_path))

    if not frames:
        raise ValueError("No CSV files were selected.")

    records = pd.concat(frames, ignore_index=True)

    # Remove rows without useful URLs.
    records = records[
        records["image_url"].notna()
        & (records["image_url"].astype(str).str.strip() != "")
        & (records["image_url"].astype(str).str.lower() != "nan")
    ].copy()

    # Normalize booleans stored as strings.
    records["license_looks_open"] = (
        records["license_looks_open"]
        .astype("string")
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
    )

    # Determine a stable photo identity when possible.
    records["photo_id_normalized"] = records["photo_id"].astype("string")
    records.loc[
        records["photo_id_normalized"].isna()
        | (records["photo_id_normalized"].str.strip() == ""),
        "photo_id_normalized"
    ] = records["image_url"].map(extract_inat_photo_id)

    return records.reset_index(drop=True)


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

def download_one(
    session: requests.Session,
    url: str,
    output_path: Path,
) -> tuple[str, str, Optional[int], str]:
    """
    Return:
        status, message, byte_count, sha256
    """
    try:
        with session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True,
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                return (
                    "failed",
                    f"Server returned HTML instead of image ({content_type})",
                    None,
                    "",
                )

            output_path = output_path.with_suffix(
                extension_from_url(url, content_type)
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("wb") as handle:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)

            size = output_path.stat().st_size

            if size == 0:
                output_path.unlink(missing_ok=True)
                return "failed", "Downloaded file was empty", 0, ""

            digest = sha256_file(output_path)

            return "downloaded", "OK", size, digest

    except requests.RequestException as exc:
        return "failed", f"Request error: {exc}", None, ""

    except OSError as exc:
        return "failed", f"File-system error: {exc}", None, ""

    except Exception as exc:
        return "failed", f"Unexpected error: {exc}", None, ""


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download aquatic plant images from GBIF/iNaturalist CSV files."
    )

    parser.add_argument("--gbif", default=DEFAULT_GBIF)
    parser.add_argument("--inat", default=DEFAULT_INAT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)

    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Download only rows where license_looks_open is True.",
    )

    parser.add_argument(
        "--gbif-only",
        action="store_true",
        help="Use only the GBIF CSV.",
    )

    parser.add_argument(
        "--inat-only",
        action="store_true",
        help="Use only the iNaturalist CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most N selected records (useful for testing).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if the output already exists.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.gbif_only and args.inat_only:
        print("ERROR: --gbif-only and --inat-only cannot be used together.")
        return 1

    gbif_path = None if args.inat_only else Path(args.gbif)
    inat_path = None if args.gbif_only else Path(args.inat)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Aquatic Plant Image Downloader")
    print("=" * 70)

    try:
        records = build_record_table(gbif_path, inat_path)
    except Exception as exc:
        print(f"\nERROR while reading CSV files:\n{exc}")
        return 1

    print(f"\nTotal rows read: {len(records):,}")

    # Filter by licensing flag if requested.
    if args.open_only:
        before = len(records)
        records = records[records["license_looks_open"] == True].copy()
        print(
            f"Rows after --open-only filter: {len(records):,} "
            f"(removed {before - len(records):,})"
        )

    # Cross-source duplicate detection.
    # Prefer URL-based deduplication, then use iNaturalist photo ID.
    records["url_key"] = records["image_url"].str.strip().str.lower()

    # Keep first URL occurrence. GBIF generally has original-size URLs,
    # while iNaturalist sample CSV may contain square.jpg for same photos.
    records["duplicate_url"] = records["url_key"].duplicated(keep="first")

    photo_mask = records["photo_id_normalized"].notna()
    records["duplicate_photo_id"] = False
    records.loc[photo_mask, "duplicate_photo_id"] = (
        records.loc[photo_mask, "photo_id_normalized"]
        .duplicated(keep="first")
    )

    print(
        f"Exact duplicate URLs found: "
        f"{int(records['duplicate_url'].sum()):,}"
    )
    print(
        f"Repeated iNaturalist photo IDs found: "
        f"{int(records['duplicate_photo_id'].sum()):,}"
    )

    # Don't download duplicates.
    records["skip_duplicate"] = (
        records["duplicate_url"] | records["duplicate_photo_id"]
    )

    # Optional limit applies to actual download candidates.
    candidates = records[~records["skip_duplicate"]].copy()

    if args.limit is not None:
        candidates = candidates.head(max(args.limit, 0))

    print(f"Download candidates: {len(candidates):,}")

    # Summary by species.
    if len(candidates):
        print("\nDownload candidates by species:")
        counts = candidates["species"].value_counts()
        for species, count in counts.items():
            print(f"  {species}: {count}")

    session = make_session()
    manifest_rows = []

    total = len(records)
    candidate_index = set(candidates.index)

    for row_index, row in records.iterrows():
        url = str(row["image_url"]).strip()
        species = str(row["species"]).strip()
        source = str(row["source"]).strip()
        photo_id = (
            None if pd.isna(row["photo_id_normalized"])
            else str(row["photo_id_normalized"]).strip()
        )

        base_species_dir = output_dir / clean_name(species)

        base_record = {
            "source": source,
            "species": species,
            "image_url": url,
            "license": row.get("license", ""),
            "license_looks_open": row.get("license_looks_open", pd.NA),
            "rights_holder": row.get("rights_holder", ""),
            "creator": row.get("creator", ""),
            "country": row.get("country", ""),
            "state_province": row.get("state_province", ""),
            "lat": row.get("lat", pd.NA),
            "lon": row.get("lon", pd.NA),
            "event_date": row.get("event_date", ""),
            "gbif_occurrence_key": row.get("gbif_occurrence_key", ""),
            "observation_id": row.get("observation_id", ""),
            "photo_id": photo_id or "",
            "status": "",
            "message": "",
            "bytes": pd.NA,
            "sha256": "",
            "file_path": "",
        }

        if row["skip_duplicate"]:
            base_record["status"] = "skipped_duplicate"
            base_record["message"] = "Duplicate URL/photo ID"
            manifest_rows.append(base_record)
            continue

        if row_index not in candidate_index:
            base_record["status"] = "skipped_limit"
            base_record["message"] = (
                "Not downloaded because --limit was reached."
            )
            manifest_rows.append(base_record)
            continue

        filename = choose_unique_filename(
            base_species_dir,
            species,
            row_index,
            source,
            url,
            photo_id,
        )

        # If the URL ends in an extension different from jpg, download_one()
        # will return the actual path after inspecting Content-Type. We first
        # use a provisional path and then infer the final path below.
        provisional = filename

        if provisional.exists() and not args.force:
            base_record["status"] = "skipped_existing"
            base_record["message"] = "File already exists"
            base_record["file_path"] = str(provisional)
            try:
                base_record["bytes"] = provisional.stat().st_size
                base_record["sha256"] = sha256_file(provisional)
            except OSError:
                pass

            manifest_rows.append(base_record)
            continue

        print(
            f"[{len(manifest_rows)+1:>5}/{total:<5}] "
            f"{species:<36} {source:<12}",
            end=" ",
            flush=True,
        )

        status, message, byte_count, digest = download_one(
            session,
            url,
            provisional,
        )

        # Find the actual downloaded extension.
        downloaded_candidates = list(
            base_species_dir.glob(
                f"{provisional.stem}.*"
            )
        )
        actual_path = (
            downloaded_candidates[0]
            if downloaded_candidates
            and status == "downloaded"
            else provisional
        )

        base_record["status"] = status
        base_record["message"] = message
        base_record["bytes"] = byte_count if byte_count is not None else pd.NA
        base_record["sha256"] = digest
        base_record["file_path"] = (
            str(actual_path) if status == "downloaded" else ""
        )

        if status == "downloaded":
            print(f"OK ({byte_count:,} bytes)")
        else:
            print(f"FAILED: {message}")

        manifest_rows.append(base_record)

        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

        # Save progress periodically so an interrupted run does not lose data.
        if len(manifest_rows) % 25 == 0:
            progress = pd.DataFrame(manifest_rows)
            progress.to_csv(
                output_dir / "download_manifest_progress.csv",
                index=False,
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Source/species summary.
    downloaded = manifest[manifest["status"] == "downloaded"].copy()

    if len(downloaded):
        summary = (
            downloaded.groupby(["source", "species"])
            .size()
            .reset_index(name="downloaded_images")
            .sort_values(["source", "species"])
        )
    else:
        summary = pd.DataFrame(
            columns=["source", "species", "downloaded_images"]
        )

    summary.to_csv(output_dir / "download_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(
        f"Downloaded: {int((manifest['status'] == 'downloaded').sum()):,}"
    )
    print(
        f"Duplicates skipped: "
        f"{int((manifest['status'] == 'skipped_duplicate').sum()):,}"
    )
    print(
        f"Existing skipped: "
        f"{int((manifest['status'] == 'skipped_existing').sum()):,}"
    )
    print(
        f"Failed: "
        f"{int((manifest['status'] == 'failed').sum()):,}"
    )
    print(
        f"Output directory: {output_dir.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
