"""
NutriCare TUI -- Main Textual Application Class
Author: Senior Python Engineer & Terminal UI Designer
Date: 2026-10-12
"""

from textual.app import App
from textual.binding import Binding

from src.tui.screens import (
    SplashScreen,
    DashboardScreen,
    GeneralMLScreen,
    ConditionSelectScreen,
    ConditionAssessmentScreen,
    ModelEvaluationScreen,
    DatasetExplorerScreen,
    AssessmentHistoryScreen,
    AboutScreen,
)


class NutriCareApp(App):
    """Interactive Gemini-inspired Terminal Dashboard for NutriCare ML."""

    TITLE = "NutriCare ML Dashboard"
    SUB_TITLE = "Local Biomarker Analysis & Clinical Pattern Engine"
    CSS_PATH = "theme.tcss"

    MODES = {
        "splash": SplashScreen,
        "dashboard": DashboardScreen,
        "general_ml": GeneralMLScreen,
        "conditions": ConditionSelectScreen,
        "evaluation": ModelEvaluationScreen,
        "dataset": DatasetExplorerScreen,
        "history": AssessmentHistoryScreen,
        "about": AboutScreen,
    }

    DEFAULT_MODE = "splash"

    BINDINGS = [
        Binding("1", "nav_dashboard", "Dashboard", show=True),
        Binding("2", "nav_general_ml", "General ML", show=True),
        Binding("3", "nav_conditions", "Conditions", show=True),
        Binding("4", "nav_evaluation", "Evaluation", show=True),
        Binding("5", "nav_dataset", "Dataset", show=True),
        Binding("6", "nav_history", "History", show=True),
        Binding("7", "nav_about", "About", show=True),
        Binding("f1", "nav_about", "Help", show=False),
        Binding("q", "quit_app", "Quit", show=True),
        Binding("escape", "nav_back", "Back", show=True),
    ]

    def action_nav_dashboard(self):
        self.switch_mode("dashboard")

    def action_nav_general_ml(self):
        self.switch_mode("general_ml")

    def action_nav_conditions(self):
        self.switch_mode("conditions")

    def action_nav_evaluation(self):
        self.switch_mode("evaluation")

    def action_nav_dataset(self):
        self.switch_mode("dataset")

    def action_nav_history(self):
        self.switch_mode("history")

    def action_nav_about(self):
        self.switch_mode("about")

    def action_quit_app(self):
        self.exit()

    def action_nav_back(self):
        # If in a pushed screen, pop it back
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self.switch_mode("dashboard")

    def open_condition_assessment(self, condition_id: str):
        """Pushes a dedicated condition assessment screen."""
        self.push_screen(ConditionAssessmentScreen(condition_id=condition_id))


if __name__ == "__main__":
    app = NutriCareApp()
    app.run()
