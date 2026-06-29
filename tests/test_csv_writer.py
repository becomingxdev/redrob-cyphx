"""Tests for the CSV writer."""

from __future__ import annotations

import csv
from pathlib import Path

from src.scoring import FinalScore
from src.output.ranking import RankedEntry
from src.reasoning.generator import Reason
from src.output.csv_writer import write_submission_csv, CSVWriter


def _make_entry(
    cid: str = "CAND_0000001",
    rank: int = 1,
    score: float = 75.5,
    confidence: float = 0.85,
    consistency: float = 0.90,
) -> RankedEntry:
    return RankedEntry(
        candidate_id=cid,
        rank=rank,
        final_score=score,
        confidence=confidence,
        consistency=consistency,
        final=FinalScore(candidate_id=cid, score=score),
    )


def _make_reason(cid: str, reasons: list[str]) -> Reason:
    return Reason(candidate_id=cid, reasons=reasons)


class TestCSVWriter:
    """Tests for write_submission_csv()."""

    def test_creates_file(self, tmp_path):
        """CSV file should be created at the given path."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        reasons = {e.candidate_id: _make_reason(e.candidate_id, ["Good profile."]) for e in ranked}
        result = write_submission_csv(out, ranked, reasons)
        assert result.exists()

    def test_header_row(self, tmp_path):
        """First row must be the expected header."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        reasons = {e.candidate_id: _make_reason(e.candidate_id, ["test"]) for e in ranked}
        write_submission_csv(out, ranked, reasons)
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["candidate_id", "rank", "final_score", "confidence", "consistency", "reason"]

    def test_data_row_count(self, tmp_path):
        """Number of data rows should match the number of ranked entries."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry(cid=f"CAND_{i:07d}", rank=i + 1) for i in range(5)]
        reasons = {e.candidate_id: _make_reason(e.candidate_id, ["reason"]) for e in ranked}
        write_submission_csv(out, ranked, reasons)
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        assert len(rows) == 5

    def test_rounded_scores(self, tmp_path):
        """final_score, confidence, consistency must be rounded to 2 decimals."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry(score=75.556, confidence=0.8555, consistency=0.9123)]
        reasons = {ranked[0].candidate_id: _make_reason(ranked[0].candidate_id, ["test"])}
        write_submission_csv(out, ranked, reasons)
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[2] == "75.56"
        assert row[3] == "0.86"
        assert row[4] == "0.91"

    def test_reason_in_output(self, tmp_path):
        """Reason text should appear in the last column."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        reasons = {ranked[0].candidate_id: _make_reason(ranked[0].candidate_id, [
            "Strong backend experience.", "Evidence-backed Python."
        ])}
        write_submission_csv(out, ranked, reasons)
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert "Strong backend experience." in row[5]
        assert "Evidence-backed Python." in row[5]

    def test_empty_reason(self, tmp_path):
        """Missing reason should produce an empty reason column."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        write_submission_csv(out, ranked, {})  # no reasons dict
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[5] == ""

    def test_utf8_encoding(self, tmp_path):
        """File must be UTF-8 encoded."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        reasons = {ranked[0].candidate_id: _make_reason(ranked[0].candidate_id, [
            "Café résumé — naïve façade."
        ])}
        write_submission_csv(out, ranked, reasons)
        raw = out.read_bytes()
        # Verify UTF-8 BOM-free by reading back.
        text = raw.decode("utf-8")
        assert "Café résumé" in text

    def test_deterministic_order(self, tmp_path):
        """Rows must follow the order of the ranked list."""
        out = tmp_path / "submission.csv"
        ranked = [
            _make_entry(cid=f"CAND_{i:07d}", rank=i + 1)
            for i in range(3)
        ]
        reasons = {e.candidate_id: _make_reason(e.candidate_id, ["test"]) for e in ranked}
        write_submission_csv(out, ranked, reasons)
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        assert rows[0][0] == "CAND_0000000"
        assert rows[1][0] == "CAND_0000001"
        assert rows[2][0] == "CAND_0000002"

    def test_empty_ranked(self, tmp_path):
        """Empty ranked list should produce a CSV with only the header."""
        out = tmp_path / "submission.csv"
        write_submission_csv(out, [], {})
        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 1  # header only

    def test_csv_writer_wrapper(self, tmp_path):
        """CSVWriter.write() should produce identical output."""
        out = tmp_path / "submission.csv"
        ranked = [_make_entry()]
        reasons = {ranked[0].candidate_id: _make_reason(ranked[0].candidate_id, ["test"])}
        writer = CSVWriter()
        result = writer.write(out, ranked, reasons)
        assert result.exists()
