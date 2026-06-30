"""Application entry point for the REDROB AI Candidate Ranking Engine.

Pipeline order:
    Load Candidates → Feature Extraction → Evidence Verification →
    Confidence → Consistency → Individual Scores → Penalty →
    Honeypot → Composite Score → Ranking → Reason Generation → CSV Export

Fallback #1 fix: JD-specific parameters (target_title, required_skills,
    preferred_skills, target_yoe) are now passed to all scoring engines.
Fallback #2 fix: LocationExtractor instantiated and score_location() wired.
Fallback #12 fix: Reason generation runs only for top-100 candidates.
Fallback #13 fix: Single output path (model_output/submission.csv).
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.candidate import Candidate
from src.parser.loader import load_candidates
from src.utils.banner import print_banner

# JD Config — loaded ONCE at module level (Fallback #1)
from src.jd_config import JD

# Feature Extractors
from src.features.title import TitleExtractor
from src.features.skills import SkillsExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.features.location import LocationExtractor
from src.features.career import CareerExtractor

# Evidence Layer
from src.evidence.verifier import EvidenceVerifier
from src.evidence.consistency import ConsistencyAnalyzer
from src.evidence.confidence import ConfidenceCalculator

# Scoring Engines
from src.scoring.title_score import score_title
from src.scoring.skill_score import score_skills
from src.scoring.experience_score import score_experience
from src.scoring.education_score import score_education
from src.scoring.behavior_score import score_behavior
from src.scoring.location_score import score_location
from src.scoring.penalties import apply_penalties
from src.scoring.honeypot import detect_honeypot
from src.scoring.composite import compose_score

# Ranking & Output
from src.output.ranking import rank_candidates
from src.reasoning.generator import generate_reason
from src.output.csv_writer import write_submission_csv

DATASET_PATH = "data/candidates.jsonl"
OUTPUT_PATH = "model_output/submission.csv"
DEBUG: bool = False

# --- Progress / timing instrumentation only — does not affect scoring ---
HEARTBEAT_EVERY_N = 100
HEARTBEAT_EVERY_SECS = 5.0

# How many top-ranked candidates get reason generation (Fallback #12)
TOP_N_FOR_REASONS = 100
TOP_N_BUFFER = 20  # Small buffer for safety

# ---------------------------------------------------------------------------
# JD Parameters — loaded from config/jd_config.yaml (Fallback #1)
# ---------------------------------------------------------------------------
_TARGET_TITLE: str | None = JD.get("target_title")
_TARGET_YOE: float | None = JD.get("target_yoe")
_REQUIRED_SKILLS: list[str] = list(JD.get("required_skills") or [])
_PREFERRED_SKILLS: list[str] = list(JD.get("preferred_skills") or [])

# ---------------------------------------------------------------------------
# PERF 1: Module-level singletons — instantiated ONCE per process
# ---------------------------------------------------------------------------
_TITLE_EXTRACTOR = TitleExtractor()
_SKILLS_EXTRACTOR = SkillsExtractor()
_EXPERIENCE_EXTRACTOR = ExperienceExtractor()
_EDUCATION_EXTRACTOR = EducationExtractor()
_LOCATION_EXTRACTOR = LocationExtractor()
_CAREER_EXTRACTOR = CareerExtractor()
_EVIDENCE_VERIFIER = EvidenceVerifier()
_CONSISTENCY_ANALYZER = ConsistencyAnalyzer()
_CONFIDENCE_CALC = ConfidenceCalculator()


def _format_elapsed(seconds: float) -> str:
    """Render a duration like '1m 23.4s' or '8.2s' for log output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


def _now_str() -> str:
    """Wall-clock timestamp for log lines, e.g. '14:08:32'."""
    return datetime.now().strftime("%H:%M:%S")


def _count_lines(path: Path) -> Optional[int]:
    """Best-effort total record count, used only to show progress/ETA."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _score_one_candidate(candidate) -> tuple:
    """Run the full per-candidate pipeline and return the composite score
    plus per-engine results needed for reason generation.

    Fallback #1: JD-specific parameters passed to all scoring engines.
    Fallback #2: Location extracted and scored.
    Fallback #3: skill_features passed to apply_penalties for domain checks.

    Returns:
        (FinalScore, component_results: dict[str, ScoreResult])
    """
    # Feature Extraction
    extracted = {
        "title":      _TITLE_EXTRACTOR.extract(candidate),
        "skills":     _SKILLS_EXTRACTOR.extract(candidate),
        "experience": _EXPERIENCE_EXTRACTOR.extract(candidate),
        "education":  _EDUCATION_EXTRACTOR.extract(candidate),
        "location":   _LOCATION_EXTRACTOR.extract(candidate),
        "career":     _CAREER_EXTRACTOR.extract(candidate),
    }

    # Evidence Layer
    evidence_result = _EVIDENCE_VERIFIER.verify(
        candidate,
        multi_source_skills=extracted["skills"].get("multi_source_skills"),
    )
    consistency_result = _CONSISTENCY_ANALYZER.analyze(candidate)
    confidence_result = _CONFIDENCE_CALC.calculate(
        candidate=candidate,
        evidence=evidence_result,
        consistency=consistency_result,
    )

    # Individual Scores — JD parameters wired in (Fallback #1)
    title_score = score_title(
        candidate,
        extracted["title"],
        target_title=_TARGET_TITLE,       # Fallback #1 fix
        career_features=extracted["career"],
    )
    skill_score = score_skills(
        candidate,
        extracted["skills"],
        required_skills=_REQUIRED_SKILLS,   # Fallback #1 fix
        preferred_skills=_PREFERRED_SKILLS, # Fallback #1 fix
    )
    exp_score = score_experience(
        candidate,
        extracted["experience"],
        extracted["career"],
        target_years=_TARGET_YOE,          # Fallback #1 fix
    )
    edu_score = score_education(
        candidate,
        extracted["education"],
        candidate_skills=set(extracted["skills"].get("skills", [])),
    )
    beh_score = score_behavior(candidate)

    # Location score (Fallback #2)
    loc_score = score_location(candidate, extracted["location"])

    # Penalties — now includes skill_features for domain mismatch (Fallback #3)
    penalty_result = apply_penalties(
        candidate=candidate,
        title_features=extracted["title"],
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        career_features=extracted["career"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
        skill_features=extracted["skills"],  # Fallback #3 fix
    )
    honeypot_result = detect_honeypot(
        candidate=candidate,
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        title_features=extracted["title"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
    )

    # Composite Score — location_score passed in (Fallback #2)
    final = compose_score(
        candidate_id=candidate.candidate_id,
        title_score=title_score,
        skill_score=skill_score,
        experience_score=exp_score,
        education_score=edu_score,
        behavior_score=beh_score,
        location_score=loc_score,           # Fallback #2 fix
        penalty=penalty_result,
        confidence_result=confidence_result,
        consistency_result=consistency_result,
        honeypot=honeypot_result,
    )

    component_results = {
        "title": title_score,
        "skills": skill_score,
        "experience": exp_score,
        "education": edu_score,
        "behavior": beh_score,
        "location": loc_score,
    }

    return final, component_results, candidate


def main() -> None:
    """Load dataset, score all candidates, rank, and export CSV."""

    print_banner()

    # Validate output path (Fallback #13 — only model_output)
    output_dir = Path("model_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(DATASET_PATH)
    if not dataset.exists():
        print(f"\nDataset not found: {DATASET_PATH}")
        return

    # Print JD configuration in use
    print(f"\n{'=' * 60}")
    print("JD Configuration (Fallback #1 fix)")
    print(f"{'=' * 60}")
    print(f"  Target Title    : {_TARGET_TITLE or 'not set'}")
    print(f"  Target YoE      : {_TARGET_YOE or 'not set'}")
    print(f"  Required Skills : {len(_REQUIRED_SKILLS)} skills")
    print(f"  Preferred Skills: {len(_PREFERRED_SKILLS)} skills")
    print(f"{'=' * 60}")

    run_start = time.time()
    print(f"\n[{_now_str()}] Loading dataset: {DATASET_PATH} ...")

    total_candidates = None
    print(f"[{_now_str()}] Processing candidates...")

    # ------------------------------------------------------------------
    # Stage 1/4 — Score every candidate
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 1/4: feature extraction + scoring — starting...")
    stage1_start = time.time()
    last_heartbeat_time = stage1_start
    last_heartbeat_count = 0

    all_finals: list = []
    all_component_results: dict[str, dict] = {}
    all_candidates_map: dict[str, Candidate] = {}
    count = 0

    for candidate in load_candidates(DATASET_PATH):
        final, component_results, cand_obj = _score_one_candidate(candidate)
        all_finals.append(final)
        all_component_results[candidate.candidate_id] = component_results
        all_candidates_map[candidate.candidate_id] = cand_obj
        count += 1

        if DEBUG and count <= 3:
            print(f"  [DEBUG] {candidate.candidate_id} → score={final.score:.2f}")

        now = time.time()
        if (
            count % HEARTBEAT_EVERY_N == 0
            or (now - last_heartbeat_time) >= HEARTBEAT_EVERY_SECS
        ):
            elapsed = now - stage1_start
            window_count = count - last_heartbeat_count
            window_secs = now - last_heartbeat_time
            rate = window_count / window_secs if window_secs > 0 else 0.0

            if total_candidates:
                pct = (count / total_candidates) * 100
                remaining = max(total_candidates - count, 0)
                eta = remaining / rate if rate > 0 else None
                eta_str = f", ETA {_format_elapsed(eta)}" if eta is not None else ""
                progress_str = f"{count:,}/{total_candidates:,} ({pct:.1f}%)"
            else:
                eta_str = ""
                progress_str = f"{count:,} processed"

            print(
                f"  [{_now_str()}] {progress_str} | "
                f"elapsed {_format_elapsed(elapsed)} | "
                f"{rate:.1f} candidates/s{eta_str}"
            )
            last_heartbeat_time = now
            last_heartbeat_count = count

    if not all_finals:
        print("\nNo valid candidates were found.")
        return

    stage1_time = time.time() - stage1_start
    print(
        f"[{_now_str()}] Stage 1/4 done — scored {count:,} candidates "
        f"in {_format_elapsed(stage1_time)}"
    )

    # ------------------------------------------------------------------
    # Stage 2/4 — Ranking
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 2/4: ranking candidates — starting...")
    stage2_start = time.time()

    ranked = rank_candidates(all_finals)

    stage2_time = time.time() - stage2_start
    print(
        f"[{_now_str()}] Stage 2/4 done — ranked {len(ranked):,} candidates "
        f"in {_format_elapsed(stage2_time)}"
    )

    # ------------------------------------------------------------------
    # Stage 3/4 — Reason Generation (TOP N ONLY — Fallback #12 fix)
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 3/4: generating reasons for top {TOP_N_FOR_REASONS} — starting...")
    stage3_start = time.time()

    # Only generate reasons for top-N candidates + buffer (Fallback #12)
    top_n_limit = TOP_N_FOR_REASONS + TOP_N_BUFFER
    reasons: dict = {}
    for entry in ranked[:top_n_limit]:
        comp = all_component_results.get(entry.candidate_id, {})
        cand = all_candidates_map.get(entry.candidate_id)
        reasons[entry.candidate_id] = generate_reason(entry.final, comp, candidate=cand)

    stage3_time = time.time() - stage3_start
    print(
        f"[{_now_str()}] Stage 3/4 done — generated {len(reasons):,} reasons "
        f"in {_format_elapsed(stage3_time)} (saved time by skipping {len(ranked) - top_n_limit:,})"
    )

    # ------------------------------------------------------------------
    # Stage 4/4 — CSV Export (Fallback #13: single output path)
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 4/4: writing CSV — starting...")
    stage4_start = time.time()

    output = write_submission_csv(OUTPUT_PATH, ranked, reasons)

    stage4_time = time.time() - stage4_start
    print(f"[{_now_str()}] Stage 4/4 done — wrote CSV in {_format_elapsed(stage4_time)}")

    total_time = time.time() - run_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Top 10 Ranked Candidates")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6}{'ID':<16}{'Score':>8}{'Conf':>7}{'Cons':>7}  Reason")
    print("-" * 72)
    for entry in ranked[:10]:
        r = reasons.get(entry.candidate_id)
        reason_preview = (r.reasons[0][:55] if r and r.reasons else "") + "…" if r and r.reasons and len(r.reasons[0]) > 55 else (r.reasons[0] if r and r.reasons else "")
        print(
            f"{entry.rank:<6}{entry.candidate_id:<16}"
            f"{entry.final_score:>8.2f}{entry.confidence:>7.2f}"
            f"{entry.consistency:>7.2f}  {reason_preview}"
        )

    print(f"\n{'=' * 60}")
    print(f"Total candidates : {count:,}")
    print("Stage timing:")
    print(f"  Feature extraction + scoring : {_format_elapsed(stage1_time)}")
    print(f"  Ranking                      : {_format_elapsed(stage2_time)}")
    print(f"  Reason generation (top-{TOP_N_FOR_REASONS})  : {_format_elapsed(stage3_time)}")
    print(f"  CSV export                   : {_format_elapsed(stage4_time)}")
    print(f"Total runtime    : {_format_elapsed(total_time)}")
    print(f"CSV output       : {output}")
    print(f"Finished at      : {_now_str()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()