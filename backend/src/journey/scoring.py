"""Journey Confidence Scoring for Step 11 Vehicle Journey Reconstruction.

Provides an engineering confidence indicator assessing data completeness and plausibility.
Keeps individual component scores visible without hiding contributing factors.
"""
from __future__ import annotations

from typing import List

from src.journey.models import (
    CameraObservationSegment,
    JourneyConfidenceScore,
    JourneyLeg,
    PlausibilityState,
)


class JourneyScorer:
    """Computes transparent confidence score for a reconstructed journey."""

    def score_journey(
        self,
        segments: List[CameraObservationSegment],
        legs: List[JourneyLeg],
    ) -> JourneyConfidenceScore:
        if not segments:
            return JourneyConfidenceScore(
                overall_score=0.0,
                recognition_quality=0.0,
                temporal_resolution=0.0,
                spatial_resolution=0.0,
                plausibility_consistency=0.0,
                notes=["No observation segments available"],
                explanations=["No observation segments — all component scores are 0.0"],
            )

        notes = []
        explanations = []

        # 1. Recognition Quality (Average consensus & OCR)
        avg_consensus = sum(s.consensus_score for s in segments) / len(segments)
        avg_ocr = sum(s.ocr_confidence for s in segments) / len(segments)
        rec_quality = round((avg_consensus * 0.6) + (avg_ocr * 0.4), 4)
        explanations.append(
            f"Recognition quality: avg_consensus={avg_consensus:.3f}, avg_ocr={avg_ocr:.3f} "
            f"→ score={rec_quality} (weight 35%)"
        )

        # 2. Temporal Resolution
        # GUARD: ingestion_time (ingested_at_utc) does NOT contribute to this score.
        # Only source_time_status == "RESOLVED" counts — meaning a verified calendar anchor.
        resolved_count = sum(1 for s in segments if s.source_time_status == "RESOLVED")
        if resolved_count == len(segments):
            temp_res = 1.0
            notes.append("Common clock source time fully resolved across all cameras")
            explanations.append(
                f"Temporal resolution: all {len(segments)} segments have RESOLVED source time "
                f"→ score=1.0 (weight 30%)"
            )
        elif resolved_count > 0:
            temp_res = round(resolved_count / len(segments), 4)
            notes.append(f"Source time partially resolved ({resolved_count}/{len(segments)} cameras)")
            explanations.append(
                f"Temporal resolution: {resolved_count}/{len(segments)} segments RESOLVED "
                f"→ score={temp_res} (weight 30%). "
                "NOTE: ingested_at_utc is NOT used for this score."
            )
        else:
            temp_res = 0.2
            notes.append("Source time unresolved; camera-local media PTS only")
            explanations.append(
                "Temporal resolution: 0 segments have RESOLVED source time → score=0.2 "
                "(minimum floor; weight 30%). "
                "ingested_at_utc is NOT used — only a verified calendar anchor qualifies."
            )

        # 3. Spatial Resolution (Fraction with verified coordinates)
        # GUARD: camera_name text does NOT count as geographic coordinates.
        # Only camera_latitude and camera_longitude numeric values count.
        coord_count = sum(
            1 for s in segments
            if s.camera_latitude is not None and s.camera_longitude is not None
        )
        spatial_res = round(coord_count / len(segments), 4)
        if coord_count == 0:
            notes.append("No geographic coordinates found for cameras in catalogue")
            explanations.append(
                "Spatial resolution: 0 cameras have verified coordinates → score=0.0 (weight 15%). "
                "NOTE: camera_name text is NOT used as a coordinate substitute."
            )
        else:
            explanations.append(
                f"Spatial resolution: {coord_count}/{len(segments)} cameras have coordinates "
                f"→ score={spatial_res} (weight 15%)"
            )

        # 4. Plausibility Consistency
        if not legs:
            plaus_score = 1.0 if len(segments) == 1 else 0.5
            explanations.append(
                f"Plausibility: no legs (single segment or no sequence) → score={plaus_score} (weight 20%)"
            )
        else:
            anomalous = sum(1 for leg in legs if leg.plausibility == PlausibilityState.ANOMALOUS)
            if anomalous > 0:
                plaus_score = max(0.0, round(1.0 - (anomalous / len(legs)), 4))
                notes.append(f"{anomalous} anomalous leg(s) detected with high/negative implied speed")
                explanations.append(
                    f"Plausibility: {anomalous}/{len(legs)} legs ANOMALOUS → score={plaus_score} (weight 20%)"
                )
            else:
                plaus_score = 1.0
                explanations.append(
                    f"Plausibility: all {len(legs)} leg(s) are not ANOMALOUS → score=1.0 (weight 20%)"
                )

        # Weighted Composite Score
        overall = round(
            (rec_quality * 0.35) +
            (temp_res * 0.30) +
            (spatial_res * 0.15) +
            (plaus_score * 0.20),
            4
        )

        return JourneyConfidenceScore(
            overall_score=overall,
            recognition_quality=rec_quality,
            temporal_resolution=temp_res,
            spatial_resolution=spatial_res,
            plausibility_consistency=plaus_score,
            notes=notes,
            explanations=explanations,
        )
