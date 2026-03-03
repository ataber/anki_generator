import json
import pytest
from pathlib import Path

from make_anki_audio_deck import load_sentences, build_model


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_json(tmp_path):
    """Write a JSON file and return its path."""
    def _write(data):
        p = tmp_path / "sentences.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p
    return _write


@pytest.fixture
def tmp_csv(tmp_path):
    """Write a CSV file and return its path."""
    def _write(text):
        p = tmp_path / "sentences.csv"
        p.write_text(text, encoding="utf-8")
        return p
    return _write


# ── load_sentences: JSON ──────────────────────────────────────────────


class TestLoadSentencesJSON:
    def test_pairs(self, tmp_json):
        path = tmp_json([["Hola", "Hello"], ["Adiós", "Goodbye"]])
        sentences, text_col = load_sentences(path)
        assert text_col is None
        assert sentences == [("UNK", "Hola", "Hello"), ("UNK", "Adiós", "Goodbye")]

    def test_triples(self, tmp_json):
        path = tmp_json([["A1", "Hola", "Hello"], ["B2", "Adiós", "Goodbye"]])
        sentences, _ = load_sentences(path)
        assert sentences == [("A1", "Hola", "Hello"), ("B2", "Adiós", "Goodbye")]

    def test_triples_empty_level_defaults_to_unk(self, tmp_json):
        path = tmp_json([["", "Hola", "Hello"]])
        sentences, _ = load_sentences(path)
        assert sentences[0][0] == "UNK"

    def test_strips_whitespace(self, tmp_json):
        path = tmp_json([["  A1 ", "  Hola  ", " Hello "]])
        sentences, _ = load_sentences(path)
        assert sentences == [("A1", "Hola", "Hello")]

    def test_rejects_non_list_root(self, tmp_json):
        path = tmp_json({"key": "value"})
        with pytest.raises(ValueError, match="must be a list"):
            load_sentences(path)

    def test_rejects_non_list_item(self, tmp_json):
        path = tmp_json(["not a list"])
        with pytest.raises(ValueError, match="expected list"):
            load_sentences(path)

    def test_rejects_wrong_length(self, tmp_json):
        path = tmp_json([["a", "b", "c", "d"]])
        with pytest.raises(ValueError, match="expected .* Text, English"):
            load_sentences(path)

    def test_rejects_empty_text(self, tmp_json):
        path = tmp_json([["", "Hello"]])
        with pytest.raises(ValueError, match="empty text"):
            load_sentences(path)

    def test_rejects_empty_english(self, tmp_json):
        path = tmp_json([["Hola", ""]])
        with pytest.raises(ValueError, match="empty text or English"):
            load_sentences(path)


# ── load_sentences: CSV ───────────────────────────────────────────────


class TestLoadSentencesCSV:
    def test_auto_detects_spanish_and_english(self, tmp_csv):
        path = tmp_csv("Level,Spanish,English\nA1,Hola,Hello\n")
        sentences, text_col = load_sentences(path)
        assert text_col == "Spanish"
        assert sentences == [("A1", "Hola", "Hello")]

    def test_auto_detects_greek(self, tmp_csv):
        path = tmp_csv("Level,Greek,English\nB1,Γεια,Hello\n")
        sentences, text_col = load_sentences(path)
        assert text_col == "Greek"

    def test_missing_level_column_defaults_to_unk(self, tmp_csv):
        path = tmp_csv("Korean,English\n안녕,Hello\n")
        sentences, text_col = load_sentences(path)
        assert text_col == "Korean"
        assert sentences == [("UNK", "안녕", "Hello")]

    def test_explicit_columns(self, tmp_csv):
        path = tmp_csv("difficulty,phrase,meaning\nhard,Hola,Hello\n")
        sentences, text_col = load_sentences(
            path,
            text_column="phrase",
            english_column="meaning",
            level_column="difficulty",
        )
        assert text_col == "phrase"
        assert sentences == [("hard", "Hola", "Hello")]

    def test_fallback_text_column_excludes_known(self, tmp_csv):
        # No preferred language column, should pick "Farsi" (first non-excluded)
        path = tmp_csv("Farsi,English\nسلام,Hello\n")
        sentences, text_col = load_sentences(path)
        assert text_col == "Farsi"

    def test_english_column_fallback_to_translation(self, tmp_csv):
        path = tmp_csv("Spanish,Translation\nHola,Hello\n")
        sentences, text_col = load_sentences(path)
        assert text_col == "Spanish"
        assert sentences == [("UNK", "Hola", "Hello")]

    def test_rejects_missing_english_column(self, tmp_csv):
        path = tmp_csv("Spanish,Other\nHola,Hello\n")
        with pytest.raises(ValueError, match="Could not find an English column"):
            load_sentences(path)

    def test_rejects_empty_text_in_row(self, tmp_csv):
        path = tmp_csv("Spanish,English\n,Hello\n")
        with pytest.raises(ValueError, match="Bad CSV row"):
            load_sentences(path)

    def test_custom_level_column(self, tmp_csv):
        path = tmp_csv("cefr_level,Spanish,English\nC1,Hola,Hello\n")
        sentences, _ = load_sentences(path, level_column="cefr_level")
        assert sentences == [("C1", "Hola", "Hello")]


# ── load_sentences: unsupported format ────────────────────────────────


def test_rejects_unsupported_format(tmp_path):
    path = tmp_path / "data.xml"
    path.write_text("<root/>")
    with pytest.raises(ValueError, match="Unsupported input type"):
        load_sentences(path)


# ── build_model ───────────────────────────────────────────────────────


class TestBuildModel:
    def test_default_has_one_template(self):
        model = build_model(123, "Test")
        assert len(model.templates) == 1
        assert model.templates[0]["name"] == "Recognition"

    def test_reverse_adds_recall_template(self):
        model = build_model(123, "Test", reverse=True)
        assert len(model.templates) == 2
        names = [t["name"] for t in model.templates]
        assert names == ["Recognition", "Recall"]

    def test_recognition_template_content(self):
        model = build_model(123, "Test", text_field_name="Greek", english_field_name="Eng")
        tmpl = model.templates[0]
        assert "{{Greek}}" in tmpl["qfmt"]
        assert "{{Audio}}" in tmpl["qfmt"]
        assert "{{Eng}}" in tmpl["afmt"]

    def test_recall_template_content(self):
        model = build_model(123, "Test", text_field_name="Greek", english_field_name="Eng", reverse=True)
        tmpl = model.templates[1]
        # Recall: English on front, target + audio on back
        assert "{{Eng}}" in tmpl["qfmt"]
        assert "{{Greek}}" in tmpl["afmt"]
        assert "{{Audio}}" in tmpl["afmt"]
        # Audio should NOT be on the question side of recall
        assert "Audio" not in tmpl["qfmt"]

    def test_field_names(self):
        model = build_model(123, "Test", text_field_name="Korean", english_field_name="Eng")
        field_names = [f["name"] for f in model.fields]
        assert field_names == ["Level", "Korean", "Eng", "Audio"]
