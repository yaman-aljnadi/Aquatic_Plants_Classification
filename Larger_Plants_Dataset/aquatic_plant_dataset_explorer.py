#!/usr/bin/env python3
"""
Aquatic Plant Dataset Explorer
==============================

Purpose:
    Audit several public biodiversity / aquatic-vegetation sources for a target
    list of aquatic plants before committing to a computer-vision dataset.

Sources checked:
    1. GBIF occurrence API (species matching + image-bearing occurrences)
    2. iNaturalist API (taxon matching + photo-bearing observations)
    3. Pl@ntNet-300K-v2 on Zenodo (metadata coverage; does NOT download 41.8 GB images)
    4. AqUavplant on Figshare (dataset metadata / files)
    5. Michigan EGLE Aquatic Nuisance Control vegetation survey sites
    6. USGS Lake Michigan SAV AUV dataset (catalog availability)
    7. NOAA Great Lakes SAV hyperspectral dataset (catalog availability)

Outputs:
    aquatic_dataset_report/
        source_catalog.csv
        gbif_report.csv
        gbif_samples.csv
        inaturalist_report.csv
        inaturalist_samples.csv
        plantnet_report.csv
        aquavplant_files.csv
        egle_sites.geojson
        run_summary.txt

Notes:
    - This script is an AUDIT / DISCOVERY tool first. It intentionally does not
      download thousands of images automatically.
    - Keep image licenses / attribution metadata with every image you later use.
    - "Great Lakes" below is a broad geographic screening box, NOT an exact basin
      polygon. Refine it for final research-quality filtering.

Dependencies:
    pip install requests pandas

Examples:
    python aquatic_plant_dataset_explorer.py
    python aquatic_plant_dataset_explorer.py --sources gbif inat
    python aquatic_plant_dataset_explorer.py --sources all --plantnet-metadata
    python aquatic_plant_dataset_explorer.py --sources egle
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SPECIES_LIST = [
    "Brasenia schreberi",
    "Cabomba",
    "Ceratophyllum demersum",
    "Elodea canadensis",
    "Heteranthera dubia",
    "Hydrocharis morsus-ranae",
    "Myriophyllum sibiricum",
    "Myriophyllum spicatum",
    "Najas flexilis",
    "Nitellopsis obtusa",
    "Nuphar variegata",
    "Nymphaea odorata",
    "Potamogeton crispus",
    "Potamogeton gramineus",
    "Potamogeton illinoensis",
    "Potamogeton natans",
    "Potamogeton praelongus",
    "Potamogeton richardsonii",
    "Potamogeton robbinsii",
    "Ranunculus aquatilis",
    "Vallisneria americana",
]

OUTPUT_DIR = Path("aquatic_dataset_report")
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = 0.15

# Broad screening area covering the Great Lakes and adjacent land.
# Longitude: -93 to -75, latitude: 40 to 49.
# This is deliberately broad and should NOT be interpreted as an exact Great
# Lakes watershed/lake polygon.
GREAT_LAKES_WKT = (
    "POLYGON((-93 40,-75 40,-75 49,-93 49,-93 40))"
)

GBIF_API = "https://api.gbif.org/v1"
INAT_API = "https://api.inaturalist.org/v1"
PLANTNET_ZENODO_RECORD = "https://zenodo.org/api/records/10419064"
FIGSHARE_AQUAVPLANT = "https://api.figshare.com/v2/articles/27019894"
EGLE_LAYER = (
    "https://gisagoegle.state.mi.us/arcgis/rest/services/"
    "EGLE/WrdOpenData/FeatureServer/14"
)
USGS_CATALOG = (
    "https://data.usgs.gov/datacatalog/data/USGS%3A66df378ad34eef5af66da455"
)
USGS_DOI = "https://doi.org/10.5066/P1GPDYMM"
NOAA_SAV_METADATA = (
    "https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/"
    "gov.noaa.ncdc%3AC01632/html"
)

SOURCE_INFO = [
    {
        "source": "GBIF",
        "type": "occurrence + image metadata",
        "url": "https://www.gbif.org/",
        "notes": "Taxonomically indexed occurrence records; supports mediaType=StillImage filtering.",
    },
    {
        "source": "iNaturalist",
        "type": "observation + photo metadata",
        "url": "https://www.inaturalist.org/",
        "notes": "Citizen-science observations; use photo license fields when building a redistributable image set.",
    },
    {
        "source": "Pl@ntNet-300K-v2",
        "type": "plant image benchmark",
        "url": "https://zenodo.org/records/10419064",
        "notes": "306,087 images / 1,000 species; metadata can be inspected before downloading the 41.8 GB image archive.",
    },
    {
        "source": "AqUavplant",
        "type": "aquatic plant UAV segmentation dataset",
        "url": "https://figshare.com/articles/dataset/AqUavplant_Dataset_A_High-Resolution_Aquatic_Plant_Classification_and_Segmentation_Image_Dataset_Using_UAV/27019894",
        "notes": "197 high-resolution images covering 31 aquatic species; includes segmentation annotations.",
    },
    {
        "source": "Michigan EGLE ANC",
        "type": "Michigan aquatic vegetation survey sites",
        "url": "https://www.michigan.gov/egle/about/organization/water-resources/aquatic-nuisance-control",
        "notes": "Historical survey-site coverage from 1986 to present; the public layer is primarily site/survey metadata rather than an image dataset.",
    },
    {
        "source": "USGS Lake Michigan SAV AUV",
        "type": "underwater RGB + masks + polygons",
        "url": USGS_DOI,
        "notes": "2020 Lake Michigan AUV imagery; 4096x2176 color images with masks/polygons and metadata.",
    },
    {
        "source": "NOAA Great Lakes SAV hyperspectral",
        "type": "airborne hyperspectral imagery",
        "url": NOAA_SAV_METADATA,
        "notes": "Great Lakes SAV hyperspectral flights; metadata indicates public access by request rather than normal direct download.",
    },
]


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "aquatic-plant-dataset-explorer/1.0 "
                "(research dataset audit; contact user-agent not provided)"
            ),
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {response.url}")
    return data


def safe_get_text(
    session: requests.Session,
    url: str,
) -> Tuple[bool, str, Optional[str]]:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return True, response.url, response.text[:10000]
    except Exception as exc:  # intentionally surfaced to caller
        return False, url, repr(exc)


def sleep_brief() -> None:
    time.sleep(SLEEP_BETWEEN_REQUESTS)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def normalize_name(value: Optional[str]) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value.strip())
    return value.lower()


def requested_rank(name: str) -> str:
    # "Cabomba" in the supplied list is a genus rather than a species.
    return "GENUS" if len(name.split()) == 1 else "SPECIES"


def is_acceptable_image_license(value: Optional[str]) -> bool:
    if not value:
        return False
    text = value.lower()
    return (
        "creativecommons.org/licenses/by/" in text
        or "creativecommons.org/publicdomain" in text
        or "cc-by" in text
        or "cc0" in text
        or "public domain" in text
    )


def write_dataframe(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"status": "no rows returned"}])
    df.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# GBIF
# -----------------------------------------------------------------------------


def gbif_match_taxon(
    session: requests.Session,
    name: str,
) -> Dict[str, Any]:
    data = get_json(
        session,
        f"{GBIF_API}/species/match",
        params={"scientificName": name},
    )
    sleep_brief()

    usage_key = data.get("usageKey")
    matched_name = data.get("scientificName") or data.get("canonicalName")
    rank = data.get("rank")
    match_type = data.get("matchType")
    status = data.get("status")

    # This prevents the common failure where a fuzzy match silently lands on a
    # broader taxon than intended.
    expected_rank = requested_rank(name)
    rank_warning = ""
    if usage_key and rank and rank.upper() != expected_rank:
        rank_warning = f"Requested {expected_rank}, matched {rank}"

    return {
        "requested_name": name,
        "requested_rank": expected_rank,
        "gbif_key": usage_key,
        "matched_name": matched_name,
        "matched_rank": rank,
        "match_type": match_type,
        "taxonomic_status": status,
        "confidence": data.get("confidence"),
        "rank_warning": rank_warning,
        "match_response": data,
    }


def gbif_occurrence_page(
    session: requests.Session,
    taxon_key: int,
    *,
    geometry: Optional[str] = None,
    limit: int = 1,
    offset: int = 0,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "taxonKey": taxon_key,
        "mediaType": "StillImage",
        "hasCoordinate": "true",
        "limit": min(limit, 300),
        "offset": offset,
    }
    if geometry:
        params["geometry"] = geometry
    data = get_json(session, f"{GBIF_API}/occurrence/search", params=params)
    sleep_brief()
    return data


def extract_gbif_media(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    media_rows: List[Dict[str, Any]] = []
    for media_item in record.get("media", []) or []:
        if media_item.get("type") != "StillImage":
            continue
        media_rows.append(
            {
                "identifier": media_item.get("identifier"),
                "format": media_item.get("format"),
                "license": media_item.get("license"),
                "rights_holder": media_item.get("rightsHolder"),
                "creator": media_item.get("creator"),
                "title": media_item.get("title"),
            }
        )
    return media_rows


def run_gbif(
    session: requests.Session,
    species_list: Sequence[str],
    out_dir: Path,
    sample_per_scope: int = 25,
) -> None:
    report_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    print("\n[GBIF] Taxonomy + image occurrence audit")

    for name in species_list:
        print(f"  - {name}")
        base: Dict[str, Any]
        try:
            base = gbif_match_taxon(session, name)
        except Exception as exc:
            report_rows.append(
                {
                    "species": name,
                    "requested_rank": requested_rank(name),
                    "status": "ERROR",
                    "error": repr(exc),
                }
            )
            continue

        key = base.get("gbif_key")
        if not key:
            report_rows.append(
                {
                    "species": name,
                    "requested_rank": requested_rank(name),
                    "matched_name": base.get("matched_name"),
                    "matched_rank": base.get("matched_rank"),
                    "match_type": base.get("match_type"),
                    "status": "NO_MATCH",
                    "rank_warning": base.get("rank_warning"),
                }
            )
            continue

        row = {
            "species": name,
            "requested_rank": base.get("requested_rank"),
            "gbif_key": key,
            "matched_name": base.get("matched_name"),
            "matched_rank": base.get("matched_rank"),
            "match_type": base.get("match_type"),
            "taxonomic_status": base.get("taxonomic_status"),
            "confidence": base.get("confidence"),
            "rank_warning": base.get("rank_warning"),
        }

        # Global count.
        try:
            global_data = gbif_occurrence_page(session, int(key), limit=1, offset=0)
            row["global_image_occurrences"] = global_data.get("count", 0)
            global_records = global_data.get("results", [])
        except Exception as exc:
            row["global_image_occurrences"] = None
            row["global_error"] = repr(exc)
            global_records = []

        # Great Lakes screening count.
        try:
            gl_data = gbif_occurrence_page(
                session,
                int(key),
                geometry=GREAT_LAKES_WKT,
                limit=1,
                offset=0,
            )
            row["great_lakes_bbox_image_occurrences"] = gl_data.get("count", 0)
            gl_records = gl_data.get("results", [])
        except Exception as exc:
            row["great_lakes_bbox_image_occurrences"] = None
            row["great_lakes_bbox_error"] = repr(exc)
            gl_records = []

        # Pull a small global sample for inspection, but never assume 100 is the
        # full dataset. GBIF occurrence searches are paginated.
        try:
            sample_data = gbif_occurrence_page(
                session,
                int(key),
                limit=min(sample_per_scope, 300),
                offset=0,
            )
            sample_records = sample_data.get("results", [])
        except Exception as exc:
            sample_records = []
            row["sample_error"] = repr(exc)

        for scope, records in (("global", sample_records), ("great_lakes_bbox", gl_records)):
            for record in records:
                for media in extract_gbif_media(record):
                    if not media.get("identifier"):
                        continue
                    sample_rows.append(
                        {
                            "species": name,
                            "matched_name": base.get("matched_name"),
                            "scope": scope,
                            "gbif_key": key,
                            "gbif_occurrence_key": record.get("key"),
                            "scientific_name": record.get("scientificName"),
                            "event_date": record.get("eventDate"),
                            "year": record.get("year"),
                            "dataset_name": record.get("datasetName"),
                            "publisher": record.get("publishingOrgKey"),
                            "country": record.get("country"),
                            "state_province": record.get("stateProvince"),
                            "lat": record.get("decimalLatitude"),
                            "lon": record.get("decimalLongitude"),
                            "basis_of_record": record.get("basisOfRecord"),
                            "image_url": media.get("identifier"),
                            "image_license": media.get("license"),
                            "rights_holder": media.get("rights_holder"),
                            "creator": media.get("creator"),
                            "license_looks_open": is_acceptable_image_license(media.get("license")),
                        }
                    )

        report_rows.append(row)

    write_dataframe(report_rows, out_dir / "gbif_report.csv")
    write_dataframe(sample_rows, out_dir / "gbif_samples.csv")


# -----------------------------------------------------------------------------
# iNaturalist
# -----------------------------------------------------------------------------


def inat_find_taxon(
    session: requests.Session,
    name: str,
) -> Optional[Dict[str, Any]]:
    params = {"q": name, "per_page": 20}
    data = get_json(session, f"{INAT_API}/taxa", params=params)
    sleep_brief()
    candidates = data.get("results", []) or []
    if not candidates:
        return None

    target = normalize_name(name)

    # Prefer an exact scientific-name hit.
    exact = [
        c for c in candidates
        if normalize_name(c.get("name")) == target
    ]
    if exact:
        return exact[0]

    # Otherwise prefer a candidate with the expected rank.
    exp_rank = requested_rank(name).lower()
    same_rank = [
        c for c in candidates
        if str(c.get("rank", "")).lower() == exp_rank
    ]
    return same_rank[0] if same_rank else candidates[0]


def run_inaturalist(
    session: requests.Session,
    species_list: Sequence[str],
    out_dir: Path,
    sample_per_species: int = 25,
) -> None:
    report_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    print("\n[iNaturalist] Taxon + photo observation audit")

    for name in species_list:
        print(f"  - {name}")
        try:
            taxon = inat_find_taxon(session, name)
            if not taxon:
                report_rows.append(
                    {
                        "species": name,
                        "status": "NO_MATCH",
                    }
                )
                continue

            taxon_id = taxon.get("id")
            params = {
                "taxon_id": taxon_id,
                "photos": "true",
                "quality_grade": "research",
                "per_page": 1,
                "page": 1,
            }
            data = get_json(session, f"{INAT_API}/observations", params=params)
            sleep_brief()

            total = data.get("total_results")
            results = data.get("results", []) or []

            report_rows.append(
                {
                    "species": name,
                    "inat_taxon_id": taxon_id,
                    "matched_name": taxon.get("name"),
                    "preferred_common_name": taxon.get("preferred_common_name"),
                    "rank": taxon.get("rank"),
                    "ancestry": taxon.get("ancestry"),
                    "research_grade_photo_observations": total,
                    "status": "OK",
                }
            )

            # Request a small sample page so we can inspect photo URLs/licensing.
            params["per_page"] = min(sample_per_species, 200)
            sample_data = get_json(session, f"{INAT_API}/observations", params=params)
            sleep_brief()

            for obs in sample_data.get("results", []) or []:
                for photo in obs.get("photos", []) or []:
                    sample_rows.append(
                        {
                            "species": name,
                            "inat_taxon_id": taxon_id,
                            "observation_id": obs.get("id"),
                            "observed_on": obs.get("observed_on"),
                            "quality_grade": obs.get("quality_grade"),
                            "lat": ((obs.get("geojson") or {}).get("coordinates") or [None, None])[1],
                            "lon": ((obs.get("geojson") or {}).get("coordinates") or [None, None])[0],
                            "photo_url": photo.get("url"),
                            "photo_id": photo.get("id"),
                            "photo_license_code": photo.get("license_code"),
                            "license_looks_open": is_acceptable_image_license(photo.get("license_code")),
                        }
                    )

        except Exception as exc:
            report_rows.append(
                {
                    "species": name,
                    "status": "ERROR",
                    "error": repr(exc),
                }
            )

    write_dataframe(report_rows, out_dir / "inaturalist_report.csv")
    write_dataframe(sample_rows, out_dir / "inaturalist_samples.csv")


# -----------------------------------------------------------------------------
# Pl@ntNet-300K-v2
# -----------------------------------------------------------------------------


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def zenodo_file_map(
    session: requests.Session,
) -> Dict[str, Dict[str, Any]]:
    data = get_json(session, PLANTNET_ZENODO_RECORD)
    return {
        str(file_info.get("key")): file_info
        for file_info in data.get("files", [])
        if file_info.get("key")
    }


def file_download_url(file_info: Dict[str, Any]) -> Optional[str]:
    links = file_info.get("links") or {}
    return (
        links.get("content")
        or links.get("self")
    )


def run_plantnet(
    session: requests.Session,
    species_list: Sequence[str],
    out_dir: Path,
    download_metadata: bool = False,
) -> None:
    print("\n[Pl@ntNet-300K-v2] Metadata coverage audit")

    try:
        files = zenodo_file_map(session)
        rows: List[Dict[str, Any]] = []
        for filename, file_info in files.items():
            rows.append(
                {
                    "file": filename,
                    "size_bytes": file_info.get("size"),
                    "checksum": file_info.get("checksum"),
                    "download_url": file_download_url(file_info),
                }
            )
        write_dataframe(rows, out_dir / "plantnet_files.csv")

        metadata_dir = out_dir / "plantnet_metadata"
        metadata_dir.mkdir(exist_ok=True)

        if download_metadata:
            for target in ("species_metadata.csv", "plantnet300K_metadata.csv"):
                info = files.get(target)
                if not info:
                    print(f"  Could not find {target} in Zenodo record")
                    continue
                url = file_download_url(info)
                if not url:
                    print(f"  No download URL available for {target}")
                    continue
                path = metadata_dir / target
                if not path.exists():
                    print(f"  Downloading {target} ...")
                    download_file(session, url, path)
                    print(f"  Saved: {path}")
                else:
                    print(f"  Already exists: {path}")

        species_meta_path = metadata_dir / "species_metadata.csv"
        image_meta_path = metadata_dir / "plantnet300K_metadata.csv"

        report_rows: List[Dict[str, Any]] = []

        if species_meta_path.exists():
            species_df = pd.read_csv(species_meta_path)
            species_cols = {c.lower(): c for c in species_df.columns}
            species_name_col = species_cols.get("species") or species_cols.get("full_species")
            species_id_col = species_cols.get("species_id")

            counts: Dict[Any, int] = {}
            if image_meta_path.exists() and species_id_col:
                image_df = pd.read_csv(image_meta_path, usecols=["species_id"])
                counts = image_df["species_id"].value_counts().to_dict()

            for requested in species_list:
                target = normalize_name(requested)
                matches = pd.DataFrame()
                if species_name_col:
                    matches = species_df[
                        species_df[species_name_col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq(target)
                    ]
                match = matches.iloc[0].to_dict() if not matches.empty else None
                species_id = match.get(species_id_col) if match else None

                report_rows.append(
                    {
                        "species": requested,
                        "found_exactly": match is not None,
                        "plantnet_species_id": species_id,
                        "full_species": match.get("full_species") if match else None,
                        "genus": match.get("genus") if match else None,
                        "family": match.get("family") if match else None,
                        "image_count_if_metadata_downloaded": counts.get(species_id, 0) if species_id is not None else 0,
                    }
                )
        else:
            for requested in species_list:
                report_rows.append(
                    {
                        "species": requested,
                        "found_exactly": None,
                        "status": "Download metadata with --plantnet-metadata",
                    }
                )

        write_dataframe(report_rows, out_dir / "plantnet_report.csv")

    except Exception as exc:
        write_dataframe(
            [{"status": "ERROR", "error": repr(exc)}],
            out_dir / "plantnet_report.csv",
        )
        print(f"  Pl@ntNet error: {exc}")


# -----------------------------------------------------------------------------
# AqUavplant / Figshare
# -----------------------------------------------------------------------------


def run_aquavplant(
    session: requests.Session,
    out_dir: Path,
) -> None:
    print("\n[AqUavplant] Figshare dataset metadata")
    try:
        data = get_json(session, FIGSHARE_AQUAVPLANT)
        rows = []
        for f in data.get("files", []) or []:
            rows.append(
                {
                    "article_id": data.get("id"),
                    "article_title": data.get("title"),
                    "version": data.get("version"),
                    "published_date": data.get("published_date"),
                    "license_name": (data.get("license") or {}).get("name") if isinstance(data.get("license"), dict) else data.get("license"),
                    "file_name": f.get("name"),
                    "size_bytes": f.get("size"),
                    "download_url": f.get("download_url"),
                }
            )
        if not rows:
            rows = [{"status": "Article found but no file records returned"}]
        write_dataframe(rows, out_dir / "aquavplant_files.csv")
        print(f"  Title: {data.get('title')}")
        print(f"  Version: {data.get('version')}")
        print(f"  Files returned: {len(rows)}")
    except Exception as exc:
        write_dataframe(
            [{"status": "ERROR", "error": repr(exc)}],
            out_dir / "aquavplant_files.csv",
        )
        print(f"  AqUavplant error: {exc}")


# -----------------------------------------------------------------------------
# Michigan EGLE
# -----------------------------------------------------------------------------


def run_egle(
    session: requests.Session,
    out_dir: Path,
) -> None:
    print("\n[EGLE] Aquatic Nuisance Control survey sites")
    try:
        # First get layer metadata / fields.
        layer_info = get_json(session, f"{EGLE_LAYER}?f=json")
        fields = [field.get("name") for field in layer_info.get("fields", [])]

        # Query all records. The layer advertises a max record count of 1000, so
        # use pagination.
        all_features: List[Dict[str, Any]] = []
        offset = 0
        page_size = 1000

        while True:
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "geojson",
            }
            response = session.get(
                f"{EGLE_LAYER}/query",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            geojson = response.json()
            features = geojson.get("features", []) or []
            all_features.extend(features)
            sleep_brief()

            exceeded = bool(geojson.get("exceededTransferLimit"))
            if not exceeded and len(features) < page_size:
                break
            if not features:
                break
            offset += len(features)

        output = {
            "type": "FeatureCollection",
            "features": all_features,
        }
        path = out_dir / "egle_sites.geojson"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")

        # Simple tabular copy for easy inspection.
        table_rows = [feature.get("properties", {}) for feature in all_features]
        write_dataframe(table_rows, out_dir / "egle_sites.csv")

        summary = pd.DataFrame(table_rows)
        print(f"  Layer fields: {', '.join([f for f in fields if f])}")
        print(f"  Survey-site records downloaded: {len(all_features)}")
        if "Year" in summary.columns and not summary.empty:
            years = pd.to_numeric(summary["Year"], errors="coerce").dropna()
            if not years.empty:
                print(f"  Year range in layer: {int(years.min())}-{int(years.max())}")

    except Exception as exc:
        write_dataframe(
            [{"status": "ERROR", "error": repr(exc)}],
            out_dir / "egle_sites.csv",
        )
        print(f"  EGLE error: {exc}")


# -----------------------------------------------------------------------------
# USGS and NOAA catalog probes
# -----------------------------------------------------------------------------


def run_catalog_probe(
    session: requests.Session,
    source_name: str,
    url: str,
    out_dir: Path,
) -> None:
    print(f"\n[{source_name}] Catalog availability probe")
    ok, final_url, text_or_error = safe_get_text(session, url)
    row = {
        "source": source_name,
        "requested_url": url,
        "reachable": ok,
        "final_url": final_url,
        "note_or_error": text_or_error if not ok else "HTTP request succeeded",
    }
    path = out_dir / f"{source_name.lower().replace(' ', '_').replace('@', 'at')}_probe.csv"
    write_dataframe([row], path)
    print(f"  Reachable: {ok}")
    if ok:
        print(f"  Final URL: {final_url}")


# -----------------------------------------------------------------------------
# Report scaffolding
# -----------------------------------------------------------------------------


def write_source_catalog(out_dir: Path) -> None:
    write_dataframe(SOURCE_INFO, out_dir / "source_catalog.csv")


def write_run_summary(out_dir: Path, selected_sources: Sequence[str]) -> None:
    summary_path = out_dir / "run_summary.txt"
    lines = [
        "Aquatic Plant Dataset Explorer",
        "=" * 30,
        f"Selected sources: {', '.join(selected_sources)}",
        f"Species/classes requested: {len(SPECIES_LIST)}",
        "",
        "Important interpretation notes:",
        "- GBIF/iNaturalist counts are occurrence/observation counts, not guaranteed unique plant individuals.",
        "- Photo counts can differ from image counts because one observation can contain multiple photographs.",
        "- The Great Lakes GBIF screen uses a broad bounding box and is not a watershed polygon.",
        "- EGLE's public layer is survey-site metadata; it should not be assumed to be a species-image dataset.",
        "- Pl@ntNet-300K-v2 metadata is useful for class coverage before committing to the 41.8 GB image archive.",
        "- Preserve original licenses/rights-holder metadata for every image used in research.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit aquatic plant datasets across several public sources.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["all", "gbif", "inat", "plantnet", "aquavplant", "egle", "usgs", "noaa"],
        default=["all"],
        help="Sources to inspect (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory (default: aquatic_dataset_report)",
    )
    parser.add_argument(
        "--plantnet-metadata",
        action="store_true",
        help="Download the two small Pl@ntNet metadata CSVs (NOT the 41.8 GB image archive).",
    )
    parser.add_argument(
        "--gbif-samples",
        type=int,
        default=25,
        help="Maximum GBIF sample occurrences inspected per species (default: 25).",
    )
    parser.add_argument(
        "--inat-samples",
        type=int,
        default=25,
        help="Maximum iNaturalist observations inspected per species (default: 25).",
    )
    return parser.parse_args()


def expand_sources(values: Sequence[str]) -> List[str]:
    if "all" in values:
        return ["gbif", "inat", "plantnet", "aquavplant", "egle", "usgs", "noaa"]
    # Preserve caller ordering and remove duplicates.
    return list(dict.fromkeys(values))


def main() -> int:
    args = parse_args()
    selected = expand_sources(args.sources)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    write_source_catalog(out_dir)
    write_run_summary(out_dir, selected)

    print("Aquatic Plant Dataset Explorer")
    print("=" * 32)
    print(f"Species/classes: {len(SPECIES_LIST)}")
    print(f"Sources: {', '.join(selected)}")
    print(f"Output: {out_dir.resolve()}")

    failures = 0

    try:
        if "gbif" in selected:
            run_gbif(session, SPECIES_LIST, out_dir, sample_per_scope=max(1, args.gbif_samples))
    except Exception as exc:
        failures += 1
        print(f"[GBIF] FATAL: {exc}")

    try:
        if "inat" in selected:
            run_inaturalist(session, SPECIES_LIST, out_dir, sample_per_species=max(1, args.inat_samples))
    except Exception as exc:
        failures += 1
        print(f"[iNaturalist] FATAL: {exc}")

    try:
        if "plantnet" in selected:
            run_plantnet(session, SPECIES_LIST, out_dir, download_metadata=args.plantnet_metadata)
    except Exception as exc:
        failures += 1
        print(f"[Pl@ntNet-300K-v2] FATAL: {exc}")

    try:
        if "aquavplant" in selected:
            run_aquavplant(session, out_dir)
    except Exception as exc:
        failures += 1
        print(f"[AqUavplant] FATAL: {exc}")

    try:
        if "egle" in selected:
            run_egle(session, out_dir)
    except Exception as exc:
        failures += 1
        print(f"[EGLE] FATAL: {exc}")

    try:
        if "usgs" in selected:
            run_catalog_probe(session, "USGS Lake Michigan SAV AUV", USGS_CATALOG, out_dir)
    except Exception as exc:
        failures += 1
        print(f"[USGS] FATAL: {exc}")

    try:
        if "noaa" in selected:
            run_catalog_probe(session, "NOAA Great Lakes SAV", NOAA_SAV_METADATA, out_dir)
    except Exception as exc:
        failures += 1
        print(f"[NOAA] FATAL: {exc}")

    print("\nDone.")
    print(f"Reports written to: {out_dir.resolve()}")
    if failures:
        print(f"Completed with {failures} source-level fatal error(s). See the report files for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
