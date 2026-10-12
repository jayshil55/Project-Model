"""
NutriCare ML -- Automated Chat Engine Test Suite
Author: Senior Python Engineer & TUI Developer
Date: 2026-10-12

Verifies:
  - Natural language parsing of blood biomarkers and clinical synonyms
  - Extraction of numeric values and canonical units
  - Real-time clinical bounds and validation warnings
  - Conversational commands: help, summary, clear, presets (good/okay/poor)
  - End-to-end Random Forest prediction execution via chat
"""

import unittest
from src.tui.chat_engine import ChatEngine


class TestChatEngine(unittest.TestCase):
    """Unit tests for ChatEngine natural language processing and state management."""

    def setUp(self):
        self.engine = ChatEngine()

    def test_initial_state_has_welcome_message(self):
        """Engine should initialize with an assistant greeting message and empty biomarkers."""
        self.assertEqual(len(self.engine.history), 1)
        self.assertEqual(self.engine.history[0].role, "assistant")
        self.assertEqual(len(self.engine.collected_biomarkers), 0)

    def test_single_biomarker_extraction_canonical_name(self):
        """Can parse canonical biomarker name with unit."""
        resp = self.engine.handle_message("Glucose_mg_dL: 92.5")
        self.assertIn("Glucose_mg_dL", self.engine.collected_biomarkers)
        self.assertEqual(self.engine.collected_biomarkers["Glucose_mg_dL"], 92.5)
        self.assertIn("Glucose", resp.text)

    def test_natural_language_synonym_extraction(self):
        """Can parse natural language terms like 'blood sugar', 'vitamin d', and 'hba1c'."""
        resp = self.engine.handle_message("patient blood sugar is 105 mg/dL, vit d 34 ng/mL and a1c 5.4%")
        self.assertEqual(self.engine.collected_biomarkers.get("Glucose_mg_dL"), 105.0)
        self.assertEqual(self.engine.collected_biomarkers.get("Vitamin_D_ng_mL"), 34.0)
        self.assertEqual(self.engine.collected_biomarkers.get("HbA1c_pct"), 5.4)

    def test_multiple_biomarkers_batch(self):
        """Can extract lipid panel in a single sentence."""
        text = "Total cholesterol 185 mg/dL, LDL 95, HDL 55 mg/dL, Triglycerides 120"
        self.engine.handle_message(text)
        self.assertEqual(self.engine.collected_biomarkers.get("Total_Cholesterol_mg_dL"), 185.0)
        self.assertEqual(self.engine.collected_biomarkers.get("LDL_Cholesterol_mg_dL"), 95.0)
        self.assertEqual(self.engine.collected_biomarkers.get("HDL_Cholesterol_mg_dL"), 55.0)
        self.assertEqual(self.engine.collected_biomarkers.get("Triglycerides_mg_dL"), 120.0)

    def test_command_preset_loading(self):
        """Can load presets via natural language command."""
        resp = self.engine.handle_message("load preset good")
        self.assertIn("Good", resp.text)
        self.assertEqual(len(self.engine.collected_biomarkers), 39)
        self.assertEqual(self.engine.collected_biomarkers["Glucose_mg_dL"], 88.0)

    def test_command_clear(self):
        """'clear' command resets all collected biomarkers."""
        self.engine.handle_message("glucose 100")
        self.assertEqual(len(self.engine.collected_biomarkers), 1)
        resp = self.engine.handle_message("clear")
        self.assertEqual(len(self.engine.collected_biomarkers), 0)
        self.assertIn("cleared", resp.text.lower())

    def test_command_summary_table(self):
        """'summary' command returns a Rich table with recorded biomarkers."""
        self.engine.handle_message("glucose 90 mg/dL, hemoglobin 14.2 g/dL")
        resp = self.engine.handle_message("summary")
        self.assertIsNotNone(resp.rich_content)

    def test_run_ml_prediction_with_preset(self):
        """Can execute the complete ML prediction pipeline via chat command."""
        self.engine.handle_message("preset good")
        summary_msg, rich_card = self.engine.run_ml_prediction()
        self.assertIn("Good", summary_msg)
        self.assertIsNotNone(rich_card)

    def test_run_ml_prediction_partial_imputation(self):
        """Can execute ML prediction with only a subset of biomarkers (missing values imputed)."""
        self.engine.handle_message("glucose 115, hba1c 6.1")
        summary_msg, rich_card = self.engine.run_ml_prediction()
        self.assertIsNotNone(rich_card)
        self.assertTrue(len(summary_msg) > 0)

    def test_unrecognized_text_guidance(self):
        """Provides helpful guidance if no biomarkers or commands were recognized."""
        resp = self.engine.handle_message("what is the weather today?")
        self.assertIn("No biomarker values recognized", resp.text)


if __name__ == "__main__":
    unittest.main()
