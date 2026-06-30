"""Application entry point for the REDROB AI Candidate Ranking Engine.

Pipeline order:
    Load Candidates → Feature Extraction → Evidence Verification →
    Confidence → Consistency → Individual Scores → Penalty →
    Honeypot → Composite Score → Ranking → Reason Generation → CSV Export
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.parser.loader import load_candidates
from src.utils.banner import print_banner

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
from src.scoring.penalties import apply_penalties
from src.scoring.honeypot import detect_honeypot
from src.scoring.composite import compose_score

# Ranking & Output
from src.output.ranking import rank_candidates
from src.reasoning.generator import generate_reason
from src.output.csv_writer import write_submission_csv

DATASET_PATH = "data/test.jsonl"
OUTPUT_PATH = "model_output/submission.csv"
DEBUG: bool = False

# --- Progress / timing instrumentation only — does not affect scoring ---
HEARTBEAT_EVERY_N = 100       # print a progress line at least every N candidates
HEARTBEAT_EVERY_SECS = 5.0    # ...or at least every N seconds, whichever first

# ---------------------------------------------------------------------------
# PERF 1: Module-level singletons — instantiated ONCE per process, not once
# per candidate. Previously each call to _score_one_candidate re-loaded all
# YAML config files and re-compiled all regex patterns (500k+ disk reads for
# 100k candidates). These are now shared, stateless objects.
# ---------------------------------------------------------------------------
_TITLE_EXTRACTOR = TitleExtractor()
_SKILLS_EXTRACTOR = SkillsExtractor()
_EXPERIENCE_EXTRACTOR = ExperienceExtractor()
_EDUCATION_EXTRACTOR = EducationExtractor()
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
    """Best-effort total record count, used only to show progress/ETA.

    This is a read-only scan separate from load_candidates() and has no
    effect on the scoring pipeline. Returns None if it can't be counted
    (e.g. unreadable file), in which case progress logging just omits the
    total/percentage/ETA and falls back to a simple running count.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def _score_one_candidate(candidate) -> tuple:
    """Run the full per-candidate pipeline and return the composite score
    plus per-engine results needed for reason generation.

    Uses module-level singleton extractors/calculators (PERF 1) so no
    YAML loading or regex compilation happens at runtime per candidate.

    Returns:
        (FinalScore, component_results: dict[str, ScoreResult])
    """
    # Feature Extraction — uses pre-built singletons (PERF 1)
    extracted = {
        "title":      _TITLE_EXTRACTOR.extract(candidate),
        "skills":     _SKILLS_EXTRACTOR.extract(candidate),
        "experience": _EXPERIENCE_EXTRACTOR.extract(candidate),
        "education":  _EDUCATION_EXTRACTOR.extract(candidate),
        "career":     _CAREER_EXTRACTOR.extract(candidate),
    }

    # Evidence Layer — singletons reused (PERF 1).
    # PERF 2: Pass already-extracted multi_source_skills to the verifier so
    # it does not re-run SkillsExtractor internally.
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

    # Individual Scores
    # FIX 3: Pass career_features to score_title for domain validation bonus.
    title_score = score_title(
        candidate,
        extracted["title"],
        career_features=extracted["career"],
    )
    skill_score = score_skills(candidate, extracted["skills"])
    exp_score = score_experience(
        candidate, extracted["experience"], extracted["career"],
    )
    edu_score = score_education(
        candidate, extracted["education"],
        candidate_skills=set(extracted["skills"].get("skills", [])),
    )
    beh_score = score_behavior(candidate)
    penalty_result = apply_penalties(
        candidate=candidate,
        title_features=extracted["title"],
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        career_features=extracted["career"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
    )
    honeypot_result = detect_honeypot(
        candidate=candidate,
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        title_features=extracted["title"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
    )

    # Composite Score
    final = compose_score(
        candidate_id=candidate.candidate_id,
        title_score=title_score,
        skill_score=skill_score,
        experience_score=exp_score,
        education_score=edu_score,
        behavior_score=beh_score,
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
    }

    return final, component_results


def main() -> None:
    """Load dataset, score all candidates, rank, and export CSV."""

    print_banner()

    dataset = Path(DATASET_PATH)
    if not dataset.exists():
        print(f"\nDataset not found: {DATASET_PATH}")
        return

    run_start = time.time()
    print(f"\n[{_now_str()}] Loading dataset: {DATASET_PATH} ...")

    # Read-only pre-scan, purely for progress/ETA display below.
    total_candidates = _count_lines(dataset)
    if total_candidates is not None:
        print(f"[{_now_str()}] Found ~{total_candidates:,} candidate records.")

    # ------------------------------------------------------------------
    # Stage 1/4 — Score every candidate (single-pass, memory-efficient
    # streaming). This is the original scoring loop, unmodified, with
    # progress heartbeats added around it.
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 1/4: feature extraction + scoring — starting...")
    stage1_start = time.time()
    last_heartbeat_time = stage1_start
    last_heartbeat_count = 0

    all_finals: list = []
    all_component_results: dict[str, dict] = {}
    count = 0

    for candidate in load_candidates(DATASET_PATH):
        final, component_results = _score_one_candidate(candidate)
        all_finals.append(final)
        all_component_results[candidate.candidate_id] = component_results
        count += 1

        if DEBUG and count <= 3:
            print(f"  [DEBUG] {candidate.candidate_id} → score={final.score:.2f}")

        # --- progress heartbeat (display only; does not affect scoring) ---
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
    # Stage 3/4 — Reason Generation
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Stage 3/4: generating reasons — starting...")
    stage3_start = time.time()

    reasons: dict = {}
    for entry in ranked:
        comp = all_component_results.get(entry.candidate_id, {})
        reasons[entry.candidate_id] = generate_reason(entry.final, comp)

    stage3_time = time.time() - stage3_start
    print(
        f"[{_now_str()}] Stage 3/4 done — generated {len(reasons):,} reasons "
        f"in {_format_elapsed(stage3_time)}"
    )

    # ------------------------------------------------------------------
    # Stage 4/4 — CSV Export
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
    print(f"  Reason generation            : {_format_elapsed(stage3_time)}")
    print(f"  CSV export                   : {_format_elapsed(stage4_time)}")
    print(f"Total runtime    : {_format_elapsed(total_time)}")
    print(f"CSV output       : {output}")
    print(f"Finished at      : {_now_str()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()