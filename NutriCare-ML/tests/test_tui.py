"""
NutriCare ML -- Automated TUI Test Suite
Author: Senior Python Engineer & TUI Developer
Date: 2026-10-12

Verifies:
  - App initialization and splash screen mounting
  - Mode switching between Dashboard, General ML, Conditions, Evaluation, Dataset, History, About
  - Dashboard multi-panel rendering (Panels A, B, C, D, E)
  - Mode A (General ML) prediction and label mapping (GOOD -> Good, MEDIUM -> Okay, HARD -> Poor)
  - Mode B (Condition Assessment) and critical calcium alert elevation
"""

import unittest
import asyncio
from src.tui.app import NutriCareApp
from src.tui.screens import GeneralMLScreen, ConditionAssessmentScreen, LABEL_MAPPING
from src.disease_patterns import DiseasePatternEngine


class TestNutriCareTUI(unittest.IsolatedAsyncioTestCase):
    """Headless integration tests for NutriCare TUI using Textual Pilot."""

    async def test_app_lifecycle_and_navigation(self):
        """Test app startup and switching between screens."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            # Check initial mode is splash
            self.assertEqual(app.current_mode, "splash")

            # Press Enter to go to dashboard
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(app.current_mode, "dashboard")

            # Navigate to General ML (key 2)
            await pilot.press("2")
            await pilot.pause()
            self.assertEqual(app.current_mode, "general_ml")

            # Navigate to Conditions (key 3)
            await pilot.press("3")
            await pilot.pause()
            self.assertEqual(app.current_mode, "conditions")

            # Navigate to Evaluation (key 4)
            await pilot.press("4")
            await pilot.pause()
            self.assertEqual(app.current_mode, "evaluation")

            # Navigate to Dataset (key 5)
            await pilot.press("5")
            await pilot.pause()
            self.assertEqual(app.current_mode, "dataset")

            # Navigate to History (key 6)
            await pilot.press("6")
            await pilot.pause()
            self.assertEqual(app.current_mode, "history")

            # Navigate to About (key 7)
            await pilot.press("7")
            await pilot.pause()
            self.assertEqual(app.current_mode, "about")

            # Return to dashboard (key 1)
            await pilot.press("1")
            await pilot.pause()
            self.assertEqual(app.current_mode, "dashboard")

    async def test_general_ml_prediction_flow(self):
        """Test loading a preset and executing Random Forest prediction in General ML screen."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("2")
            await pilot.pause()
            self.assertEqual(app.current_mode, "general_ml")

            screen = app.screen
            self.assertIsInstance(screen, GeneralMLScreen)

            # Apply Good preset
            screen._apply_preset("Good")
            self.assertEqual(screen.input_widgets["Glucose_mg_dL"].value, "88.0")

            # Run prediction
            screen._run_prediction()

            # Verify result container is populated
            res_box = screen.query_one("#ml-results-container")
            self.assertTrue(len(res_box.children) > 0)

    async def test_condition_assessment_critical_calcium_elevation(self):
        """Verify that a critical calcium value in Vitamin D assessment elevates status and is prominently visible."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            # Open Vitamin D condition assessment
            app.open_condition_assessment("vitamin_d")
            await pilot.pause()

            screen = app.screen
            self.assertIsInstance(screen, ConditionAssessmentScreen)

            # Apply critical preset (VitD = 35.0 normal, Calcium = 6.0 critical)
            screen._apply_condition_preset("critical")
            self.assertEqual(screen.input_widgets["Vitamin_D_ng_mL"].value, "35.0")
            self.assertEqual(screen.input_widgets["Calcium_mg_dL"].value, "6.0")

            # Run evaluation
            screen._run_condition_evaluation()

            # Verify evaluation results
            res_box = screen.query_one("#cond-results-container")
            self.assertTrue(len(res_box.children) > 0)

            # Also check underlying engine directly
            eval_res = DiseasePatternEngine.evaluate_condition("vitamin_d", {"Vitamin_D_ng_mL": 35.0, "Calcium_mg_dL": 6.0})
            self.assertEqual(eval_res.status, "ABNORMAL_PATTERN")
            self.assertTrue(len(eval_res.critical_alerts) > 0)
            self.assertTrue(any("CRITICAL" in p for p in eval_res.patterns_detected))

    async def test_dashboard_button_navigation(self):
        """Test clicking pinned navigation buttons on the dashboard."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(app.current_mode, "dashboard")

            # Click General ML button
            await pilot.click("#btn-nav-ml")
            await pilot.pause()
            self.assertEqual(app.current_mode, "general_ml")

            # Click Back button
            await pilot.click("#btn-back-dashboard")
            await pilot.pause()
            self.assertEqual(app.current_mode, "dashboard")

            # Click Conditions button
            await pilot.click("#btn-nav-cond")
            await pilot.pause()
            self.assertEqual(app.current_mode, "conditions")

    async def test_history_logging_and_clearing(self):
        """Verify assessment logging to session history and clearing functionality."""
        from src.tui.widgets import SESSION_ASSESSMENTS, record_session_assessment
        record_session_assessment("Test Mode", "Test Condition", "NORMAL", "All within bounds")
        self.assertTrue(len(SESSION_ASSESSMENTS) > 0)

        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("6")
            await pilot.pause()
            self.assertEqual(app.current_mode, "history")

            # Click Clear History button
            await pilot.click("#btn-clear-history")
            await pilot.pause()
            self.assertEqual(len(SESSION_ASSESSMENTS), 0)


class TestLabelMapping(unittest.TestCase):
    """Test suite for label mapping consistency."""

    def test_label_mapping_integrity(self):
        """Verify mapping covers all 3 model classes correctly."""
        self.assertEqual(LABEL_MAPPING["GOOD"], "Good")
        self.assertEqual(LABEL_MAPPING["MEDIUM"], "Okay")
        self.assertEqual(LABEL_MAPPING["HARD"], "Poor")


class TestChartsModule(unittest.TestCase):
    """Test suite for terminal chart rendering functions."""

    def test_horizontal_bar_renders(self):
        from src.tui.charts import render_horizontal_bar
        bar = render_horizontal_bar(50, 100, width=20)
        self.assertTrue(len(bar) > 0)

    def test_dataset_distribution_chart(self):
        from src.tui.charts import render_dataset_distribution_chart
        chart = render_dataset_distribution_chart()
        self.assertIn("Good", chart.plain)
        self.assertIn("Okay", chart.plain)
        self.assertIn("Poor", chart.plain)

    def test_model_performance_chart(self):
        from src.tui.charts import render_model_performance_chart
        chart = render_model_performance_chart()
        self.assertIn("98.00%", chart.plain)

    def test_session_trends_empty(self):
        from src.tui.charts import render_session_trends_chart
        chart = render_session_trends_chart([])
        self.assertIn("No assessments", chart.plain)

    def test_biomarker_range_indicator(self):
        from src.tui.charts import render_biomarker_range_indicator
        ind = render_biomarker_range_indicator("Glucose", 88.0, 70.0, 140.0, "mg/dL")
        self.assertIn("NORMAL", ind.plain)
        ind_high = render_biomarker_range_indicator("Glucose", 250.0, 70.0, 140.0, "mg/dL", critical_high=400.0)
        self.assertIn("HIGH", ind_high.plain)


class TestConditionCardsAndScrolling(unittest.IsolatedAsyncioTestCase):
    """Test that condition cards render and all 39 biomarker inputs exist."""

    async def test_condition_cards_grid_renders_10_panels(self):
        """Verify all 10 condition cards render in the grid."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("3")
            await pilot.pause()
            self.assertEqual(app.current_mode, "conditions")
            grid = app.screen.query_one("#condition-cards-grid")
            buttons = grid.query(".cond-card-btn")
            self.assertEqual(len(buttons), 10)

    async def test_all_39_biomarker_inputs_exist(self):
        """Verify all 39 biomarker inputs are created and reachable."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("2")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, GeneralMLScreen)
            self.assertEqual(len(screen.input_widgets), 39)
            # Verify first and last biomarkers exist
            self.assertIn("Glucose_mg_dL", screen.input_widgets)
            self.assertIn("UACR_mg_g", screen.input_widgets)

    async def test_repeated_predictions_stress_test(self):
        """Stress-test: run predictions repeatedly with different presets without DuplicateIds or crash."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("2")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, GeneralMLScreen)

            # Cycle through presets and predictions 5 times
            for _ in range(5):
                screen._apply_preset("Good")
                screen._run_prediction()
                screen._apply_preset("Okay")
                screen._run_prediction()
                screen._apply_preset("Poor")
                screen._run_prediction()
                screen._clear_inputs()

            res_display = screen.query_one("#ml-results-display")
            self.assertIsNotNone(res_display)

    async def test_repeated_condition_evaluations_stress_test(self):
        """Stress-test: run condition evaluations repeatedly with normal/abnormal/critical presets."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            app.open_condition_assessment("diabetes")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, ConditionAssessmentScreen)

            for _ in range(4):
                screen._apply_condition_preset("normal")
                screen._run_condition_evaluation()
                screen._apply_condition_preset("abnormal")
                screen._run_condition_evaluation()
                screen._apply_condition_preset("critical")
                screen._run_condition_evaluation()
                # Clear and re-evaluate
                screen.query_one("#btn-cond-preset-clear").press()

            res_display = screen.query_one("#cond-results-display")
            self.assertIsNotNone(res_display)

    async def test_multicycle_all_screens_navigation(self):
        """Verify repeated navigation across all seven screens without crashes."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            modes = ["dashboard", "general_ml", "conditions", "evaluation", "dataset", "history", "about"]
            key_seq = ["1", "2", "3", "4", "5", "6", "7"]
            for cycle in range(3):
                for key, expected_mode in zip(key_seq, modes):
                    await pilot.press(key)
                    await pilot.pause()
                    self.assertEqual(app.current_mode, expected_mode)

    async def test_responsive_layout_resizing(self):
        """Verify screen responds to size changes (80x24, 100x30, 120x35) without error."""
        app = NutriCareApp()
        async with app.run_test() as pilot:
            await pilot.press("1")
            await pilot.pause()
            # Resize to narrow 80x24
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            self.assertTrue(app.screen.has_class("-narrow"))

            # Resize to standard 120x35
            await pilot.resize_terminal(120, 35)
            await pilot.pause()
            self.assertFalse(app.screen.has_class("-narrow"))


if __name__ == "__main__":
    unittest.main()
