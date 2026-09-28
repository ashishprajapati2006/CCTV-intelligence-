"""Step 6 ANPR Ingestion Pipeline for Observed Vehicle Database (Step 7).

Consumes Step 6 outputs:
  - data/anpr/<camera_id>/consensus/anpr_consensus.json
  - and/or data/anpr/<camera_id>/raw_results/raw_observations.jsonl

Enforces strict timestamp semantics:
  - Preserves Step 6 first_seen_pts_ms, last_seen_pts_ms.
  - Determines recognition_pts_ms from the strongest supporting evidence frame.
  - Keeps source_time as None with source_time_status="NOT_RESOLVED" (no fabricated times).
  - Only uses application time for ingested_at_utc.
  - Audits filename PTS vs evidence PTS for provenance verification.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ai.anpr.normalizer import normalize_raw_text
from src.common.logging import get_logger
from src.observed.aggregator import VehicleAggregator
from src.observed.models import (
    SupportingFrameRef,
    UncertainObservation,
    VehicleObservation,
    VehicleTrackRecord,
)
from src.observed.repository import ObservedVehicleRepository

logger = get_logger("observed_importer")


def check_evidence_filename_pts(
    image_path: Optional[str],
    evidence_pts_ms: Optional[float],
) -> Tuple[str, Optional[float]]:
    """Inspect evidence image filename for embedded PTS and audit against evidence PTS.

    Returns:
        (status, filename_pts_ms) where status is 'MATCH', 'DISCREPANCY', or 'UNKNOWN'.
    """
    if not image_path:
        return "UNKNOWN", None

    match = re.search(r"_pts(\d+)", image_path)
    if not match:
        return "UNKNOWN", None

    fn_pts = float(match.group(1))
    if evidence_pts_ms is None:
        return "UNKNOWN", fn_pts

    if abs(fn_pts - evidence_pts_ms) < 1.0:
        return "MATCH", fn_pts
    return "DISCREPANCY", fn_pts


class ANPRStep6Importer:
    """Imports Step 6 ANPR results into the Observed Vehicle Database."""

    def __init__(
        self,
        repository: ObservedVehicleRepository,
        aggregator: Optional[VehicleAggregator] = None,
        promoted_statuses: Optional[List[str]] = None,
    ):
        self.repo = repository
        self.aggregator = aggregator or VehicleAggregator()
        self.promoted_statuses = promoted_statuses or ["CONFIRMED", "PROBABLE"]

    def import_consensus_file(self, file_path: str) -> Dict[str, Any]:
        """Import a Step 6 anpr_consensus.json file into the observed vehicle database.

        Args:
            file_path: Path to anpr_consensus.json.

        Returns:
            Dictionary of import statistics.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Step 6 consensus file not found: {file_path}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        cam_id = data.get("camera_id", "cam_unknown")
        tracks_data = data.get("tracks", [])
        logger.info("Importing Step 6 consensus from %s (camera=%s, tracks=%d)", file_path, cam_id, len(tracks_data))

        imported_obs_count = 0
        skipped_duplicates = 0
        uncertain_count = 0
        tracks_saved = 0
        malformed_count = 0

        # Group valid observations by normalized registration for aggregation
        obs_by_vehicle: Dict[str, List[VehicleObservation]] = defaultdict(list)

        for trk in tracks_data:
            track_id = trk.get("track_id")
            status = trk.get("recognition_status", "UNREADABLE")
            final_reg = trk.get("final_registration_number")
            consensus_score = float(trk.get("consensus_score", 0.0))
            first_pts = trk.get("first_seen_pts_ms")
            last_pts = trk.get("last_seen_pts_ms")
            loop_inst = trk.get("loop_instance")
            evidence_chain = trk.get("evidence_chain", [])

            if not track_id:
                malformed_count += 1
                continue

            # Build supporting frames references and locate strongest evidence frame
            supporting_frames: List[SupportingFrameRef] = []
            best_evidence_img: Optional[str] = None
            best_evidence_pts: Optional[float] = None
            best_evidence_score = -1.0
            best_ocr_conf = 0.0
            best_det_conf = 0.85
            best_qual_score = 0.70

            for ev in evidence_chain:
                fid = int(ev.get("frame_id", 0))
                pts = ev.get("pts_ms")
                img_path = ev.get("image_crop_path")

                fn_status, fn_pts = check_evidence_filename_pts(img_path, pts)
                supporting_frames.append(SupportingFrameRef(
                    frame_id=fid,
                    pts_ms=pts,
                    evidence_image=img_path,
                    filename_pts_ms=fn_pts,
                    filename_pts_status=fn_status,
                ))

                raw_ocr = ev.get("raw_ocr", {})
                ocr_conf = float(raw_ocr.get("confidence", 0.0))
                det_conf = float(ev.get("plate_detection_confidence", 0.85))
                qual_score = float(ev.get("plate_quality_score", 0.70))
                ev_score = float(ev.get("observation_score", ocr_conf * det_conf))

                if ev_score > best_evidence_score or best_evidence_pts is None:
                    best_evidence_score = ev_score
                    best_evidence_pts = pts
                    best_evidence_img = img_path
                    best_ocr_conf = ocr_conf
                    best_det_conf = det_conf
                    best_qual_score = qual_score

            # recognition_pts_ms: PTS of the strongest supporting evidence frame
            rec_pts = best_evidence_pts if best_evidence_pts is not None else first_pts

            # Audit the primary evidence image against its filename PTS
            fn_status, fn_pts = check_evidence_filename_pts(best_evidence_img, rec_pts)

            # 1. Save Track Record
            norm_reg = normalize_raw_text(final_reg) if final_reg else None
            track_record = VehicleTrackRecord(
                track_id=track_id,
                camera_id=cam_id,
                registration_number=final_reg,
                normalized_registration_number=norm_reg,
                first_seen_pts_ms=first_pts,
                recognition_pts_ms=rec_pts,
                last_seen_pts_ms=last_pts,
                source_time=None,
                source_time_status="NOT_RESOLVED",
                supporting_frames=supporting_frames,
                consensus_score=consensus_score,
                status=status,
                observation_count=max(1, len(evidence_chain)),
                loop_instance=loop_inst,
            )
            self.repo.save_track(track_record, flush=False)
            tracks_saved += 1

            # 2. Check if Status is Promoted to Definitive Vehicle Identity
            if status in self.promoted_statuses and final_reg and norm_reg:
                rec_pts_int = int(rec_pts) if rec_pts is not None else 0
                obs_id = f"OBS-{cam_id}-{track_id}-{rec_pts_int}"

                obs = VehicleObservation(
                    observation_id=obs_id,
                    camera_id=cam_id,
                    track_id=track_id,
                    registration_number=final_reg,
                    normalized_registration_number=norm_reg,
                    first_seen_pts_ms=first_pts,
                    recognition_pts_ms=rec_pts,
                    last_seen_pts_ms=last_pts,
                    source_time=None,
                    source_time_status="NOT_RESOLVED",
                    ocr_confidence=best_ocr_conf,
                    plate_detection_confidence=best_det_conf,
                    plate_quality_score=best_qual_score,
                    consensus_score=consensus_score,
                    supporting_frame_count=trk.get("supporting_frame_count", len(supporting_frames)),
                    status=status,
                    evidence_image=best_evidence_img,
                    evidence_filename_pts_status=fn_status,
                    evidence_filename_pts_ms=fn_pts,
                    source="sentinel",
                    loop_instance=loop_inst,
                )

                added = self.repo.add_observation(obs)
                if added:
                    imported_obs_count += 1
                else:
                    skipped_duplicates += 1

                obs_by_vehicle[norm_reg].append(obs)
            else:
                # 3. Route to Uncertain Observations (Audit Record)
                raw_extracted = ""
                if evidence_chain:
                    raw_extracted = evidence_chain[0].get("raw_ocr", {}).get("raw_text", "")

                u_obs = UncertainObservation(
                    observation_id=f"UNC-{cam_id}-{track_id}",
                    camera_id=cam_id,
                    track_id=track_id,
                    raw_text=raw_extracted,
                    normalized_text=normalize_raw_text(raw_extracted),
                    status=status,
                    reason="LOW_CONFIDENCE_OR_UNREADABLE" if status == "UNREADABLE" else "CONFLICTING_OCR_READINGS",
                    evidence_image=best_evidence_img,
                    first_seen_pts_ms=first_pts,
                    recognition_pts_ms=rec_pts,
                    last_seen_pts_ms=last_pts,
                    source_time=None,
                    source_time_status="NOT_RESOLVED",
                    loop_instance=loop_inst,
                )
                self.repo.add_uncertain_observation(u_obs)
                uncertain_count += 1

        # 4. Aggregate Vehicles
        vehicles_updated = 0
        curr_vehicle_count = len(self.repo.list_vehicles(limit=10000))

        for norm_reg in obs_by_vehicle.keys():
            all_v_obs = self.repo.get_vehicle_observations(norm_reg)
            existing = self.repo.get_vehicle(norm_reg)
            idx = curr_vehicle_count + 1 if not existing else 1

            aggregated = self.aggregator.aggregate_vehicle_observations(
                normalized_reg=norm_reg,
                observations=all_v_obs,
                existing_vehicle=existing,
                vehicle_index=idx,
            )
            if aggregated:
                self.repo.save_vehicle(aggregated, flush=False)
                vehicles_updated += 1
                if not existing:
                    curr_vehicle_count += 1

        if hasattr(self.repo, "flush"):
            self.repo.flush()

        summary = {
            "source_file": str(file_path),
            "camera_id": cam_id,
            "tracks_processed": len(tracks_data),
            "tracks_saved": tracks_saved,
            "observations_imported": imported_obs_count,
            "observations_skipped_duplicate": skipped_duplicates,
            "uncertain_observations_recorded": uncertain_count,
            "vehicles_updated": vehicles_updated,
            "malformed_records_skipped": malformed_count,
        }
        logger.info("Import complete for %s: %s", cam_id, summary)
        return summary
