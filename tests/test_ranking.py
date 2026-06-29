"""Tests for the ranking engine."""

from __future__ import annotations

from src.scoring import FinalScore, PenaltyResult, HoneypotResult
from src.output.ranking import RankedEntry, rank_candidates, RankingEngine


def _make_final(
    cid: str = "CAND_0000001",
    score: float = 50.0,
    confidence: float = 0.8,
    consistency: float = 0.9,
    experience: float = 60.0,
) -> FinalScore:
    return FinalScore(
        candidate_id=cid,
        score=score,
        component_scores={
            "title": 70.0,
            "skills": 65.0,
            "experience": experience,
            "education": 40.0,
            "behavior": 20.0,
            "confidence": confidence,
            "consistency": consistency,
        },
    )


class TestRankCandidates:
    """Tests for rank_candidates()."""

    def test_empty_input(self):
        """Empty list should return empty list."""
        result = rank_candidates([])
        assert result == []

    def test_single_candidate(self):
        """Single candidate should get rank 1."""
        result = rank_candidates([_make_final()])
        assert len(result) == 1
        assert result[0].rank == 1
        assert result[0].candidate_id == "CAND_0000001"

    def test_correct_ordering(self):
        """Candidates should be sorted by descending score."""
        finals = [
            _make_final(cid="CAND_A", score=30.0),
            _make_final(cid="CAND_B", score=90.0),
            _make_final(cid="CAND_C", score=60.0),
        ]
        result = rank_candidates(finals)
        assert [e.candidate_id for e in result] == ["CAND_B", "CAND_C", "CAND_A"]
        assert [e.rank for e in result] == [1, 2, 3]

    def test_sequential_ranks(self):
        """Ranks should be 1, 2, 3, ... with no gaps."""
        finals = [_make_final(cid=f"CAND_{i:07d}", score=float(100 - i)) for i in range(10)]
        result = rank_candidates(finals)
        assert [e.rank for e in result] == list(range(1, 11))

    def test_tie_break_confidence(self):
        """On tied score, higher confidence should rank first."""
        finals = [
            _make_final(cid="CAND_LOW", score=50.0, confidence=0.3),
            _make_final(cid="CAND_HIGH", score=50.0, confidence=0.9),
        ]
        result = rank_candidates(finals)
        assert result[0].candidate_id == "CAND_HIGH"
        assert result[0].rank == 1
        assert result[1].candidate_id == "CAND_LOW"

    def test_tie_break_consistency(self):
        """On tied score+confidence, higher consistency should rank first."""
        finals = [
            _make_final(cid="CAND_LOW", score=50.0, confidence=0.8, consistency=0.4),
            _make_final(cid="CAND_HIGH", score=50.0, confidence=0.8, consistency=0.9),
        ]
        result = rank_candidates(finals)
        assert result[0].candidate_id == "CAND_HIGH"

    def test_tie_break_experience(self):
        """On tied score+confidence+consistency, higher experience ranks first."""
        finals = [
            _make_final(cid="CAND_LOW", score=50.0, confidence=0.8, consistency=0.8, experience=30.0),
            _make_final(cid="CAND_HIGH", score=50.0, confidence=0.8, consistency=0.8, experience=80.0),
        ]
        result = rank_candidates(finals)
        assert result[0].candidate_id == "CAND_HIGH"

    def test_tie_break_candidate_id(self):
        """On fully tied scores, ascending candidate ID should win."""
        finals = [
            _make_final(cid="CAND_Z", score=50.0, confidence=0.5, consistency=0.5, experience=50.0),
            _make_final(cid="CAND_A", score=50.0, confidence=0.5, consistency=0.5, experience=50.0),
            _make_final(cid="CAND_M", score=50.0, confidence=0.5, consistency=0.5, experience=50.0),
        ]
        result = rank_candidates(finals)
        assert [e.candidate_id for e in result] == ["CAND_A", "CAND_M", "CAND_Z"]

    def test_returns_ranked_entry(self):
        """Each result must be a RankedEntry with all expected fields."""
        result = rank_candidates([_make_final()])
        entry = result[0]
        assert isinstance(entry, RankedEntry)
        assert hasattr(entry, "candidate_id")
        assert hasattr(entry, "rank")
        assert hasattr(entry, "final_score")
        assert hasattr(entry, "confidence")
        assert hasattr(entry, "consistency")
        assert hasattr(entry, "final")

    def test_preserves_final_reference(self):
        """RankedEntry.final should reference the original FinalScore."""
        final = _make_final(cid="CAND_X")
        result = rank_candidates([final])
        assert result[0].final is final

    def test_large_dataset(self):
        """Should handle 10,000 candidates without issue."""
        import random
        random.seed(42)
        finals = [
            _make_final(
                cid=f"CAND_{i:07d}",
                score=random.uniform(0, 100),
                confidence=random.uniform(0, 1),
                consistency=random.uniform(0, 1),
                experience=random.uniform(0, 100),
            )
            for i in range(10_000)
        ]
        result = rank_candidates(finals)
        assert len(result) == 10_000
        # Verify descending order.
        for i in range(1, len(result)):
            assert result[i - 1].final_score >= result[i].final_score

    def test_deterministic(self):
        """Same input must produce same output."""
        finals = [
            _make_final(cid=f"CAND_{i:07d}", score=float(i % 10))
            for i in range(100)
        ]
        r1 = rank_candidates(finals)
        r2 = rank_candidates(finals)
        assert [e.candidate_id for e in r1] == [e.candidate_id for e in r2]
        assert [e.rank for e in r1] == [e.rank for e in r2]

    def test_ranking_engine_wrapper(self):
        """RankingEngine.rank() should produce identical results."""
        engine = RankingEngine()
        finals = [_make_final(cid=f"CAND_{i:07d}", score=float(i % 5)) for i in range(20)]
        result = engine.rank(finals)
        assert len(result) == 20
        assert result[0].rank == 1
