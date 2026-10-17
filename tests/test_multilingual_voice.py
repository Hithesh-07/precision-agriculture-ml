"""
tests/test_multilingual_voice.py
================================
Unit tests for multilingual speech recognition, entity extraction,
and localized response generation (Telugu, Hindi, English).
"""

import pytest
from src.multilingual_voice import (
    detect_language,
    normalize_indian_numerals,
    parse_multilingual_voice_text,
    build_multilingual_spoken_answer,
    build_multilingual_result_card,
    CROP_DISPLAY_MAP,
)


class TestLanguageDetection:
    def test_detect_telugu(self):
        assert detect_language("వరి ఆంధ్రప్రదేశ్ ఖరీఫ్") == "te"

    def test_detect_hindi(self):
        assert detect_language("गेहूं पंजाब रबी 2020") == "hi"

    def test_detect_english(self):
        assert detect_language("Rice Punjab Kharif 5000 ha") == "en"

    def test_detect_empty_defaults_to_english(self):
        assert detect_language("") == "en"


class TestNumeralNormalization:
    def test_telugu_numerals(self):
        # ౫౦౦౦ -> 5000
        assert normalize_indian_numerals("౫౦౦౦") == "5000"

    def test_hindi_numerals(self):
        # ५००० -> 5000
        assert normalize_indian_numerals("५०००") == "5000"

    def test_mixed_text(self):
        assert normalize_indian_numerals("విస్తీర్ణం ౫౦౦౦ హెక్టార్లు") == "విస్తీర్ణం 5000 హెక్టార్లు"


class TestMultilingualParsing:
    def test_telugu_voice_parsing(self):
        text = "వరి ఆంధ్రప్రదేశ్ ఖరీఫ్ విస్తీర్ణం 5000 వర్షపాతం 1200"
        parsed = parse_multilingual_voice_text(text)
        assert parsed["detected_lang"] == "te"
        assert parsed["crop"] == "Rice"
        assert parsed["state"] == "Andhra Pradesh"
        assert parsed["season"] == "Kharif"
        assert parsed["area"] == 5000.0
        assert parsed["rainfall"] == 1200.0

    def test_telugu_voice_parsing_with_varsham(self):
        text = "వరి ఆంధ్రప్రదేశ్ 5000 వర్షం 1200"
        parsed = parse_multilingual_voice_text(text)
        assert parsed["detected_lang"] == "te"
        assert parsed["crop"] == "Rice"
        assert parsed["state"] == "Andhra Pradesh"
        assert parsed["area"] == 5000.0
        assert parsed["rainfall"] == 1200.0

    def test_hindi_voice_parsing(self):
        text = "गेहूं पंजाब रबी 4000 हेक्टेयर 800 वर्षा"
        parsed = parse_multilingual_voice_text(text)
        assert parsed["detected_lang"] == "hi"
        assert parsed["crop"] == "Wheat"
        assert parsed["state"] == "Punjab"
        assert parsed["season"] == "Rabi"
        assert parsed["area"] == 4000.0
        assert parsed["rainfall"] == 800.0

    def test_english_voice_parsing(self):
        text = "Cotton in Maharashtra season Kharif area 2500 rainfall 950"
        parsed = parse_multilingual_voice_text(text)
        assert parsed["detected_lang"] == "en"
        assert parsed["crop"] == "Cotton(lint)"
        assert parsed["state"] == "Maharashtra"
        assert parsed["season"] == "Kharif"
        assert parsed["area"] == 2500.0
        assert parsed["rainfall"] == 950.0

    def test_telugu_numerals_parsing(self):
        text = "మొక్కజొన్న తెలంగాణ రబీ ౩౦౦౦ ౧౦౦౦"
        parsed = parse_multilingual_voice_text(text)
        assert parsed["detected_lang"] == "te"
        assert parsed["crop"] == "Maize"
        assert parsed["state"] == "Telangana"
        assert parsed["season"] == "Rabi"
        assert parsed["area"] == 3000.0
        assert parsed["rainfall"] == 1000.0


class TestSpokenAnswerGeneration:
    def test_telugu_spoken_answer(self):
        speech, lang_code = build_multilingual_spoken_answer("Rice", 4.2, "Decision Tree", 5000, "te")
        assert lang_code == "te-IN"
        assert "వరి" in speech
        assert "4.20" in speech
        assert "Decision Tree" in speech
        assert "టన్నులు" in speech

    def test_hindi_spoken_answer(self):
        speech, lang_code = build_multilingual_spoken_answer("Wheat", 3.8, "Random Forest", 2000, "hi")
        assert lang_code == "hi-IN"
        assert "गेहूं" in speech
        assert "3.80" in speech
        assert "Random Forest" in speech
        assert "टन" in speech

    def test_english_spoken_answer(self):
        speech, lang_code = build_multilingual_spoken_answer("Rice", 4.0, "Decision Tree", 1000, "en")
        assert lang_code == "en-IN"
        assert "Rice" in speech
        assert "4.00" in speech

    def test_result_card_generation(self):
        card_te = build_multilingual_result_card("Rice", 3.5, "Decision Tree", 1000, "te")
        assert "వరి" in card_te["headline"]
        assert "టన్నులు" in card_te["yield_text"]


class TestTeluguTranslations:
    def test_telugu_keys_parity_with_english(self):
        from src.translations import TRANSLATIONS
        en_keys = set(TRANSLATIONS["en"].keys())
        te_keys = set(TRANSLATIONS["te"].keys())
        missing_in_telugu = en_keys - te_keys
        assert len(missing_in_telugu) == 0, f"Missing Telugu translations: {missing_in_telugu}"

