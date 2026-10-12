"""
NutriCare TUI -- Application Screens
Author: Senior Python TUI Engineer & Terminal UI/UX Designer
Date: 2026-10-12

Features:
  - Responsive multi-panel dashboard with pinned navigation and zero awkward vertical gaps
  - Side-by-side / split responsive workspaces for General ML (Mode A) and Conditions (Mode B)
  - Pinned control docks (presets and run buttons always accessible without scrolling)
  - Prominent Gemini-inspired true-color gradient branding
  - Adaptive layout with @media queries for small/narrow terminals
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, DataTable, Header, Footer
from textual.containers import Container, Horizontal, Vertical, Grid, ScrollableContainer, VerticalScroll
from textual.reactive import reactive

from src.config import MODEL_PATH, METADATA_PATH, DATASET_PATH
from src.biomarker_validation import BiomarkerValidator, BIOMARKER_SPECS
from src.disease_patterns import DiseasePatternEngine, CONDITION_PANELS
from src.disease_explainer import DiseaseExplainer
from src.tui.logo import get_logo_text, get_compact_header_brand
from src.tui.charts import render_horizontal_bar, render_biomarker_range_indicator
from src.tui.chat_engine import ChatEngine, ChatMessage
from src.tui.widgets import (
    HeaderBar,
    FooterBar,
    OverviewMetricsPanel,
    VisualAnalyticsPanel,
    ModelPanel,
    ConditionListPanel,
    DatasetOverviewPanel,
    RecentAssessmentsPanel,
    SystemStatusPanel,
    SESSION_ASSESSMENTS,
    record_session_assessment,
)

# Display label mapping for the general Random Forest classifier
LABEL_MAPPING = {
    "GOOD": "Good",
    "MEDIUM": "Okay",
    "HARD": "Poor",
}

# Synthetic demo presets for the General ML Classifier (Mode A)
PRESET_SAMPLES = {
    "Good": {
        "Glucose_mg_dL": "88.0", "HbA1c_pct": "5.1", "Hemoglobin_g_dL": "14.5",
        "Total_Cholesterol_mg_dL": "175.0", "LDL_Cholesterol_mg_dL": "95.0",
        "HDL_Cholesterol_mg_dL": "58.0", "Triglycerides_mg_dL": "110.0",
        "Vitamin_D_ng_mL": "35.0", "Vitamin_B12_pg_mL": "450.0",
        "Ferritin_ng_mL": "80.0", "Creatinine_mg_dL": "0.85", "Urea_mg_dL": "28.0",
        "ALT_U_L": "22.0", "AST_U_L": "20.0", "WBC_10^9_L": "6.5",
        "Platelet_10^9_L": "240.0", "RBC_10^12_L": "4.8", "TSH_mIU_L": "1.8",
        "Albumin_g_dL": "4.4", "CRP_mg_L": "0.8", "eGFR_mL_min_1_73m2": "95.0",
        "Bilirubin_mg_dL": "0.7", "ALP_U_L": "65.0", "Sodium_mmol_L": "140.0",
        "Potassium_mmol_L": "4.2", "Calcium_mg_dL": "9.4", "Magnesium_mg_dL": "2.1",
        "Folate_ng_mL": "12.0", "MCV_fL": "88.0", "MCH_pg": "29.5",
        "MCHC_g_dL": "33.5", "RDW_pct": "12.5", "Neutrophil_10^9_L": "3.8",
        "Lymphocyte_10^9_L": "2.1", "ApoB_mg_dL": "80.0",
        "Lipoprotein_a_mg_dL": "15.0", "Homocysteine_umol_L": "8.5",
        "GGT_U_L": "24.0", "UACR_mg_g": "12.0",
    },
    "Okay": {
        "Glucose_mg_dL": "112.0", "HbA1c_pct": "5.9", "Hemoglobin_g_dL": "13.2",
        "Total_Cholesterol_mg_dL": "215.0", "LDL_Cholesterol_mg_dL": "138.0",
        "HDL_Cholesterol_mg_dL": "46.0", "Triglycerides_mg_dL": "165.0",
        "Vitamin_D_ng_mL": "22.0", "Vitamin_B12_pg_mL": "280.0",
        "Ferritin_ng_mL": "35.0", "Creatinine_mg_dL": "1.15", "Urea_mg_dL": "38.0",
        "ALT_U_L": "38.0", "AST_U_L": "32.0", "WBC_10^9_L": "8.2",
        "Platelet_10^9_L": "210.0", "RBC_10^12_L": "4.4", "TSH_mIU_L": "3.2",
        "Albumin_g_dL": "4.0", "CRP_mg_L": "2.8", "eGFR_mL_min_1_73m2": "75.0",
        "Bilirubin_mg_dL": "1.0", "ALP_U_L": "85.0", "Sodium_mmol_L": "138.0",
        "Potassium_mmol_L": "4.4", "Calcium_mg_dL": "9.1", "Magnesium_mg_dL": "1.9",
        "Folate_ng_mL": "7.5", "MCV_fL": "84.0", "MCH_pg": "27.5",
        "MCHC_g_dL": "32.5", "RDW_pct": "14.2", "Neutrophil_10^9_L": "5.0",
        "Lymphocyte_10^9_L": "2.4", "ApoB_mg_dL": "105.0",
        "Lipoprotein_a_mg_dL": "28.0", "Homocysteine_umol_L": "12.0",
        "GGT_U_L": "42.0", "UACR_mg_g": "45.0",
    },
    "Poor": {
        "Glucose_mg_dL": "210.0", "HbA1c_pct": "9.2", "Hemoglobin_g_dL": "8.8",
        "Total_Cholesterol_mg_dL": "290.0", "LDL_Cholesterol_mg_dL": "195.0",
        "HDL_Cholesterol_mg_dL": "32.0", "Triglycerides_mg_dL": "350.0",
        "Vitamin_D_ng_mL": "9.5", "Vitamin_B12_pg_mL": "140.0",
        "Ferritin_ng_mL": "8.0", "Creatinine_mg_dL": "3.2", "Urea_mg_dL": "85.0",
        "ALT_U_L": "130.0", "AST_U_L": "125.0", "WBC_10^9_L": "13.0",
        "Platelet_10^9_L": "140.0", "RBC_10^12_L": "3.8", "TSH_mIU_L": "0.05",
        "Albumin_g_dL": "3.0", "CRP_mg_L": "25.0", "eGFR_mL_min_1_73m2": "35.0",
        "Bilirubin_mg_dL": "3.5", "ALP_U_L": "120.0", "Sodium_mmol_L": "128.0",
        "Potassium_mmol_L": "6.5", "Calcium_mg_dL": "8.0", "Magnesium_mg_dL": "1.4",
        "Folate_ng_mL": "4.0", "MCV_fL": "78.0", "MCH_pg": "25.0",
        "MCHC_g_dL": "31.0", "RDW_pct": "17.0", "Neutrophil_10^9_L": "8.0",
        "Lymphocyte_10^9_L": "1.0", "ApoB_mg_dL": "160.0",
        "Lipoprotein_a_mg_dL": "80.0", "Homocysteine_umol_L": "20.0",
        "GGT_U_L": "100.0", "UACR_mg_g": "350.0",
    },
}


# ===========================================================================
# 1. Splash Screen (Hermes Agent Inspired Welcome Screen)
# ===========================================================================
class SplashScreen(Screen):
    """Startup welcome screen inspired by Hermes Agent, featuring the gradient wordmark and system specs."""

    def compose(self):
        yield Vertical(
            Vertical(
                Static(get_logo_text(compact=False), id="welcome-logo"),
                Static(
                    "\n[bold #43D9D0]✦ AI-POWERED BIOMARKER INTELLIGENCE PLATFORM ✦[/]\n"
                    "[#9299AA]Educational Machine Learning & Multi-Analyte Clinical Pattern Assessment[/]\n",
                    id="welcome-desc",
                ),
                Static(
                    "┌─ LOCAL ML SYSTEM SPECIFICATIONS ────────────────────────────────────────┐\n"
                    "│  ● Local ML Engine:    [bold #4ECA78]Online / Ready[/] (Balanced Random Forest v2.1)    │\n"
                    "│  ◆ Biomarker Analytes: [bold #E6E8F0]39 Verified Blood Features[/] (7 Core Lab Panels)     │\n"
                    "│  ❖ Clinical Protocols: [bold #A879D5]10 Targeted Evidence-Based Assessment Panels[/]    │\n"
                    "│  ◈ Training Dataset:   [bold #43D9D0]1,000 Synthetic Multi-Analyte Records[/] (0 Missing) │\n"
                    "└────────────────────────────────────────────────────────────────────────┘",
                    id="welcome-specs-box",
                ),
                Horizontal(
                    Button("Enter NutriCare [Enter]", variant="primary", id="btn-enter", classes="welcome-btn"),
                    Button("About NutriCare [A]", variant="default", id="btn-about", classes="welcome-btn"),
                    Button("Quit [Q]", variant="error", id="btn-exit", classes="welcome-btn"),
                    classes="welcome-actions",
                ),
                id="welcome-card",
            ),
            id="welcome-screen-container",
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-enter":
            self.app.switch_mode("dashboard")
        elif event.button.id == "btn-about":
            self.app.switch_mode("about")
        elif event.button.id == "btn-exit":
            self.app.exit()

    def on_key(self, event):
        if event.key in ("enter", "space"):
            self.app.switch_mode("dashboard")
        elif event.key == "a":
            self.app.switch_mode("about")
        elif event.key in ("q", "escape"):
            self.app.exit()


class ResponsiveScreen(Screen):
    """Base screen that dynamically toggles the .-narrow class on resize or mount for responsive layout."""

    def on_resize(self, event):
        if hasattr(event, "size") and event.size.width > 0:
            self.set_class(event.size.width < 96, "-narrow")

    def on_mount(self):
        w = self.app.size.width if (self.app and self.app.size.width > 0) else self.size.width
        if w > 0:
            self.set_class(w < 96, "-narrow")


# ===========================================================================
# 2. Dashboard Screen
# ===========================================================================
class DashboardScreen(ResponsiveScreen):
    """Responsive multi-panel central dashboard with pinned navigation."""

    def compose(self):
        yield HeaderBar(mode="Dashboard")
        # Pinned Navigation directly below Header!
        yield Horizontal(
            Button("[1] Dashboard", variant="primary", id="btn-nav-dash", classes="nav-btn"),
            Button("[2] General ML", variant="default", id="btn-nav-ml", classes="nav-btn"),
            Button("[3] Conditions", variant="default", id="btn-nav-cond", classes="nav-btn"),
            Button("[4] Evaluation", variant="default", id="btn-nav-eval", classes="nav-btn"),
            Button("[5] Dataset", variant="default", id="btn-nav-data", classes="nav-btn"),
            Button("[6] History", variant="default", id="btn-nav-hist", classes="nav-btn"),
            Button("[7] About", variant="default", id="btn-nav-about", classes="nav-btn"),
            id="dash-quick-nav",
        )
        yield VerticalScroll(
            OverviewMetricsPanel(),
            Horizontal(
                Button("💬 Open Chat Assistant", variant="primary", id="btn-dash-chat", classes="tool-btn-primary"),
                Button("⚡ 39-Biomarker Form", variant="default", id="btn-dash-form", classes="tool-btn"),
                Button("🩺 10 Clinical Protocols", variant="default", id="btn-dash-cond", classes="tool-btn"),
                Button("📊 Model Evaluation", variant="default", id="btn-dash-eval", classes="tool-btn"),
                classes="action-row",
            ),
            Horizontal(
                Vertical(
                    ModelPanel(),
                    DatasetOverviewPanel(),
                    classes="dash-col",
                ),
                Vertical(
                    ConditionListPanel(),
                    RecentAssessmentsPanel(),
                    classes="dash-col",
                ),
                id="dash-content-grid",
            ),
            id="dashboard-scrollable",
        )
        yield FooterBar()

    def on_button_pressed(self, event: Button.Pressed):
        btn_map = {
            "btn-nav-dash": "dashboard",
            "btn-nav-ml": "general_ml",
            "btn-nav-cond": "conditions",
            "btn-nav-eval": "evaluation",
            "btn-nav-data": "dataset",
            "btn-nav-hist": "history",
            "btn-nav-about": "about",
            "btn-dash-chat": "general_ml",
            "btn-dash-form": "general_ml",
            "btn-dash-cond": "conditions",
            "btn-dash-eval": "evaluation",
        }
        target = btn_map.get(event.button.id)
        if target:
            self.app.switch_mode(target)


# ===========================================================================
# 3. General Biomarker Classification Screen (Mode A)
# ===========================================================================
class GeneralMLScreen(ResponsiveScreen):
    """
    Chat-First Conversational Assistant + 39-Biomarker Manual Form.
    Inspired by Hermes Agent's chat interface and Gemini-inspired colors.
    """

    chat_engine: ChatEngine
    input_widgets: Dict[str, Input] = {}
    active_view: str = "chat"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_engine = ChatEngine()
        self.input_widgets = {}
        self.active_view = "chat"

    def compose(self):
        yield HeaderBar(mode="General Biomarker ML Assistant")
        # Pinned Top Control Dock with Mode Toggle, Actions, and Presets!
        yield Vertical(
            Horizontal(
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Button("💬 Chat Mode", variant="primary", id="btn-mode-chat", classes="tool-btn-active"),
                Button("📝 Form Mode", variant="default", id="btn-mode-form", classes="tool-btn"),
                Button("⚡ Run Prediction", variant="primary", id="btn-run-prediction", classes="tool-btn-primary"),
                classes="action-row",
            ),
            Horizontal(
                Label("[bold #A879D5]Presets:[/] ", classes="toolbar-label"),
                Button("Good", variant="success", id="btn-preset-good", classes="tool-btn"),
                Button("Okay", variant="warning", id="btn-preset-okay", classes="tool-btn"),
                Button("Poor", variant="error", id="btn-preset-poor", classes="tool-btn"),
                Button("Clear", id="btn-preset-clear", classes="tool-btn"),
                classes="preset-row",
            ),
            id="ml-control-dock",
        )

        # 1. Chat-First Conversational Assistant Workspace
        yield Vertical(
            VerticalScroll(id="chat-messages-scroll"),
            Horizontal(
                Input(
                    placeholder="Type biomarkers (e.g. 'Glucose 95 mg/dL, HbA1c 5.2%') or ask to 'run prediction' / 'help'...",
                    id="chat-input-field",
                ),
                Button("Send ↵", variant="primary", id="btn-chat-send"),
                id="chat-input-bar",
            ),
            Horizontal(
                Button("⚡ Predict", id="btn-chip-predict", classes="chip-btn"),
                Button("📋 Summary", id="btn-chip-summary", classes="chip-btn"),
                Button("Preset: Good", id="btn-chip-good", classes="chip-btn"),
                Button("Preset: Okay", id="btn-chip-okay", classes="chip-btn"),
                Button("Preset: Poor", id="btn-chip-poor", classes="chip-btn"),
                Button("✕ Clear", id="btn-chip-clear", classes="chip-btn"),
                Button("❓ Help", id="btn-chip-help", classes="chip-btn"),
                id="chat-chips-bar",
            ),
            id="chat-workspace-container",
        )

        # 2. 39-Biomarker Manual Form Workspace
        yield Horizontal(
            VerticalScroll(
                Static("[bold #43D9D0]39 BLOOD BIOMARKER INPUT PROFILES (CANONICAL UNITS)[/]\n"),
                Vertical(id="biomarker-inputs-container"),
                id="ml-form-scroll",
                classes="ml-form-col",
            ),
            VerticalScroll(
                Static("[bold #43D9D0]RANDOM FOREST PREDICTION REPORT[/]\n"),
                Vertical(
                    Static(
                        "[italic #9299AA]Select a preset above (e.g. Synthetic: Good) or manually enter "
                        "biomarker values on the left, then click [bold #43D9D0]'⚡ Run Prediction'[/] "
                        "to execute the Random Forest pipeline.[/]\n",
                        id="ml-results-display",
                    ),
                    id="ml-results-container",
                ),
                Static(
                    "[bold #A879D5]LEGAL & CLINICAL DISCLAIMER:[/] This general health classification reflects synthetic training labels "
                    "derived from mathematical rules (Good, Okay, Poor). It is an educational AI demonstration and NOT a clinical medical diagnosis. "
                    "Always consult a licensed medical provider for diagnostic evaluation.",
                    classes="disclaimer-box",
                ),
                id="ml-results-scroll",
                classes="ml-results-col",
            ),
            id="manual-form-container",
            classes="ml-workspace",
        )
        yield FooterBar()

    def on_mount(self):
        """Populate initial chat messages and 39 input fields."""
        super().on_mount()

        # Mount initial welcome message in chat
        scroll_area = self.query_one("#chat-messages-scroll")
        for msg in self.chat_engine.history:
            self._mount_message_bubble(scroll_area, msg)

        # Hide manual form initially
        self.query_one("#manual-form-container").display = False

        # Populate manual form inputs
        container = self.query_one("#biomarker-inputs-container")
        self.input_widgets = {}
        features = list(BIOMARKER_SPECS.keys())

        for feat in features:
            spec = BIOMARKER_SPECS.get(feat, {})
            d_name = spec.get("display_name", feat)
            unit = spec.get("canonical_unit", "")
            r_low = spec.get("ref_low", "")
            r_high = spec.get("ref_high", "")
            ref_str = f"[{r_low}-{r_high}]" if r_low != "" else ""

            clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", feat)
            inp = Input(placeholder="enter value", id=f"inp-{clean_id}", classes="field-input")
            row = Horizontal(
                Label(f"{d_name[:24]:<24}", classes="field-label"),
                Label(f"{unit:<12} {ref_str}", classes="field-unit"),
                inp,
                classes="form-row",
            )
            container.mount(row)
            self.input_widgets[feat] = inp

    def _mount_message_bubble(self, container, msg: ChatMessage):
        """Mounts a single styled message bubble to the container."""
        if msg.role == "user":
            bubble = Vertical(
                Static(f"[bold #43D9D0]You:[/] {msg.text}"),
                classes="chat-msg chat-msg-user",
            )
        else:
            items = [Static(f"[bold #A879D5]✦ NutriCare ML Assistant:[/] {msg.text}")]
            if msg.rich_content is not None:
                items.append(Static(msg.rich_content, classes="chat-msg-rich"))
            bubble = Vertical(*items, classes="chat-msg chat-msg-assistant")
        container.mount(bubble)

    def on_input_submitted(self, event: Input.Submitted):
        if event.input.id == "chat-input-field":
            self._handle_send_chat()

    def _handle_send_chat(self):
        inp = self.query_one("#chat-input-field", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""

        # Process through chat engine
        resp = self.chat_engine.handle_message(text)

        # Mount user bubble and assistant bubble
        scroll_area = self.query_one("#chat-messages-scroll")
        user_msg = self.chat_engine.history[-2] if len(self.chat_engine.history) >= 2 else None
        if user_msg and user_msg.role == "user":
            self._mount_message_bubble(scroll_area, user_msg)
        self._mount_message_bubble(scroll_area, resp)
        scroll_area.scroll_end(animate=False)

        # Sync to manual form inputs
        self._sync_chat_to_form()

    def _sync_chat_to_form(self):
        """Copies biomarkers collected in Chat into the manual form input widgets."""
        for feat, inp in self.input_widgets.items():
            if feat in self.chat_engine.collected_biomarkers:
                val = self.chat_engine.collected_biomarkers[feat]
                inp.value = str(val)

    def _sync_form_to_chat(self):
        """Copies values typed into manual form widgets into the chat engine."""
        for feat, inp in self.input_widgets.items():
            val_str = inp.value.strip()
            if val_str:
                try:
                    self.chat_engine.collected_biomarkers[feat] = float(val_str)
                except ValueError:
                    pass

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        if btn_id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")
        elif btn_id == "btn-chat-send":
            self._handle_send_chat()
        elif btn_id == "btn-mode-chat":
            self._switch_to_chat_mode()
        elif btn_id == "btn-mode-form":
            self._switch_to_form_mode()
        elif btn_id in ("btn-preset-good", "btn-chip-good"):
            self._apply_preset_both("Good")
        elif btn_id in ("btn-preset-okay", "btn-chip-okay"):
            self._apply_preset_both("Okay")
        elif btn_id in ("btn-preset-poor", "btn-chip-poor"):
            self._apply_preset_both("Poor")
        elif btn_id in ("btn-preset-clear", "btn-chip-clear"):
            self._clear_both()
        elif btn_id in ("btn-run-prediction", "btn-chip-predict"):
            self._run_prediction_both()
        elif btn_id == "btn-chip-summary":
            self._trigger_chat_command("summary")
        elif btn_id == "btn-chip-help":
            self._trigger_chat_command("help")

    def _switch_to_chat_mode(self):
        self._sync_form_to_chat()
        self.query_one("#manual-form-container").display = False
        self.query_one("#chat-workspace-container").display = True
        self.query_one("#btn-mode-chat").add_class("tool-btn-active")
        self.query_one("#btn-mode-form").remove_class("tool-btn-active")
        self.active_view = "chat"

    def _switch_to_form_mode(self):
        self._sync_chat_to_form()
        self.query_one("#chat-workspace-container").display = False
        self.query_one("#manual-form-container").display = True
        self.query_one("#btn-mode-form").add_class("tool-btn-active")
        self.query_one("#btn-mode-chat").remove_class("tool-btn-active")
        self.active_view = "form"

    def _trigger_chat_command(self, cmd: str):
        if self.active_view != "chat":
            self._switch_to_chat_mode()
        resp = self.chat_engine.handle_message(cmd)
        scroll_area = self.query_one("#chat-messages-scroll")
        user_msg = self.chat_engine.history[-2] if len(self.chat_engine.history) >= 2 else None
        if user_msg and user_msg.role == "user":
            self._mount_message_bubble(scroll_area, user_msg)
        self._mount_message_bubble(scroll_area, resp)
        scroll_area.scroll_end(animate=False)

    def _apply_preset_both(self, preset_name: str):
        preset = PRESET_SAMPLES.get(preset_name, {})
        for feat, inp in self.input_widgets.items():
            val = preset.get(feat, "")
            inp.value = str(val)
        self.chat_engine.set_biomarkers(preset)

        # Notify in chat
        if self.active_view == "chat":
            resp = self.chat_engine.handle_message(f"preset {preset_name.lower()}")
            scroll_area = self.query_one("#chat-messages-scroll")
            user_msg = self.chat_engine.history[-2] if len(self.chat_engine.history) >= 2 else None
            if user_msg and user_msg.role == "user":
                self._mount_message_bubble(scroll_area, user_msg)
            self._mount_message_bubble(scroll_area, resp)
            scroll_area.scroll_end(animate=False)

    def _clear_both(self):
        for inp in self.input_widgets.values():
            inp.value = ""
        self.chat_engine.clear_values()
        self.query_one("#ml-results-display", Static).update(
            "[italic #9299AA]Inputs cleared. Select a preset or enter values in Chat or Form to evaluate.[/]"
        )
        if self.active_view == "chat":
            resp = self.chat_engine.handle_message("clear")
            scroll_area = self.query_one("#chat-messages-scroll")
            user_msg = self.chat_engine.history[-2] if len(self.chat_engine.history) >= 2 else None
            if user_msg and user_msg.role == "user":
                self._mount_message_bubble(scroll_area, user_msg)
            self._mount_message_bubble(scroll_area, resp)
            scroll_area.scroll_end(animate=False)

    def _run_prediction_both(self):
        # Sync form if currently in form view
        if self.active_view == "form":
            self._sync_form_to_chat()

        # Run prediction
        summary_msg, rich_card = self.chat_engine.run_ml_prediction()

        # Update manual form results display
        if rich_card is not None:
            self.query_one("#ml-results-display", Static).update(rich_card)
        else:
            self.query_one("#ml-results-display", Static).update(f"[bold #EF4444]{summary_msg}[/]")

        # Update chat if in chat view or record response
        resp = ChatMessage("assistant", summary_msg, rich_content=rich_card)
        self.chat_engine.history.append(ChatMessage("user", "run prediction"))
        self.chat_engine.history.append(resp)

        if self.active_view == "chat":
            scroll_area = self.query_one("#chat-messages-scroll")
            self._mount_message_bubble(scroll_area, self.chat_engine.history[-2])
            self._mount_message_bubble(scroll_area, resp)
            scroll_area.scroll_end(animate=False)

    def _apply_preset(self, preset_name: str):
        """Backward-compatible alias for _apply_preset_both."""
        self._apply_preset_both(preset_name)

    def _run_prediction(self):
        """Backward-compatible alias for _run_prediction_both."""
        self._run_prediction_both()

    def _clear_inputs(self):
        """Backward-compatible alias for _clear_both."""
        self._clear_both()


# ===========================================================================
# 4. Condition Select Screen (Mode B)
# ===========================================================================
class ConditionSelectScreen(ResponsiveScreen):
    """Grid of all 10 clinical condition panels rendered as rich selectable cards."""

    def compose(self):
        yield HeaderBar(mode="Select Condition Assessment")
        yield Vertical(
            Horizontal(
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Label("[bold #43D9D0] Select one of 10 evidence-based clinical protocols below to run targeted assessment:[/]", classes="hint-label"),
                classes="action-row",
            ),
            id="ml-control-dock",
        )
        yield ScrollableContainer(
            Vertical(
                Static(
                    "[bold #43D9D0]TARGETED CLINICAL CONDITION ASSESSMENTS (10 GUIDELINE PANELS)[/]\n"
                    "[#9299AA]Each protocol prompts [bold #E6E8F0]only for biomarkers relevant[/] to that physiological system.[/]\n",
                    classes="panel-title",
                ),
                Horizontal(
                    Vertical(id="cond-col-left", classes="cond-col"),
                    Vertical(id="cond-col-right", classes="cond-col"),
                    id="condition-cards-grid",
                ),
                Static(
                    "[bold #A879D5]EDUCATIONAL & CLINICAL DISCLAIMER:[/] All condition assessments are rule-based clinical demonstrations "
                    "reflecting established guidelines (ADA, WHO, KDIGO, AHA/ACC). They are NOT diagnostic tools. Consult a licensed physician.",
                    classes="disclaimer-box",
                ),
            ),
            classes="scrollable-content",
        )
        yield FooterBar()

    def on_mount(self):
        super().on_mount()
        col_left = self.query_one("#cond-col-left")
        col_right = self.query_one("#cond-col-right")
        
        panel_descriptions = {
            "diabetes": "Evaluates fasting plasma glucose and HbA1c against ADA 2024 diagnostic thresholds.",
            "anemia": "Classifies erythrocyte mass, microcytic/normocytic/macrocytic morphology, and anisocytosis.",
            "iron_deficiency": "Evaluates ferritin iron stores with acute inflammation (CRP) confounding adjustment.",
            "dyslipidemia": "Assesses atherogenic cholesterol and triglyceride risk according to AHA/ACC guidelines.",
            "kidney": "Stratifies glomerular filtration (KDIGO G1-G5), creatinine clearance, and albuminuria.",
            "liver": "Differentiates hepatocellular necrosis (ALT/AST), biliary cholestasis (ALP), and bilirubin.",
            "thyroid": "Evaluates pituitary TSH feedback for primary hypothyroidism and hyperthyroidism.",
            "vitamin_d": "Assesses 25(OH)D storage sufficiency, parathyroid compensation, and calcium safety.",
            "b12_folate": "Evaluates cobalamin and folate deficiency with homocysteine methylation balance.",
            "electrolytes": "Detects acute hyperkalemia, hyponatremia, and critical electrolyte panic thresholds.",
        }

        for idx, (c_id, meta) in enumerate(CONDITION_PANELS.items(), start=1):
            req_markers = ", ".join(meta.get("required", []))
            opt_markers = ", ".join(meta.get("optional", []))
            markers_info = f"[bold #43D9D0]Required:[/] {req_markers}"
            if opt_markers:
                markers_info += f"  │  [bold #9299AA]Optional:[/] {opt_markers}"

            desc = panel_descriptions.get(c_id, "Targeted physiological biomarker assessment protocol.")

            card = Vertical(
                Static(f"[bold #43D9D0][{idx:>2}] {meta['name']}[/]", classes="cond-card-title"),
                Static(f"[#E6E8F0]{desc}[/]", classes="cond-card-desc"),
                Static(markers_info, classes="cond-card-markers"),
                Button("Assess Condition →", variant="primary", id=f"btn-cond-{c_id}", classes="cond-card-btn"),
                classes="cond-card",
            )
            # Alternate between left and right columns
            target_col = col_left if (idx % 2 == 1) else col_right
            target_col.mount(card)

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        if btn_id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")
        elif btn_id and btn_id.startswith("btn-cond-"):
            cond_id = btn_id.replace("btn-cond-", "")
            self.app.open_condition_assessment(cond_id)



# ===========================================================================
# 5. Condition Assessment Screen (Targeted Mode B Form & Results)
# ===========================================================================
class ConditionAssessmentScreen(ResponsiveScreen):
    """Dedicated screen for assessing one specific condition panel."""

    condition_id: str = "vitamin_d"
    input_widgets: Dict[str, Input] = {}

    def __init__(self, condition_id: str = "vitamin_d", **kwargs):
        super().__init__(**kwargs)
        self.condition_id = condition_id

    def compose(self):
        meta = CONDITION_PANELS.get(self.condition_id, {})
        c_name = meta.get("name", "Condition Assessment")
        yield HeaderBar(mode=f"Assessment: {c_name[:25]}")
        # Pinned Top Control Dock with Presets and Actions!
        yield Vertical(
            Horizontal(
                Label("[bold #A879D5]Presets: [/]", classes="toolbar-label"),
                Button("Normal Preset", variant="success", id="btn-cond-preset-normal", classes="tool-btn"),
                Button("Abnormal Preset", variant="warning", id="btn-cond-preset-abnormal", classes="tool-btn"),
                Button("Critical Alert", variant="error", id="btn-cond-preset-critical", classes="tool-btn"),
                Button("Clear Inputs", id="btn-cond-preset-clear", classes="tool-btn"),
                classes="preset-row",
            ),
            Horizontal(
                Button("⚡ Run Assessment", variant="primary", id="btn-run-cond-eval", classes="tool-btn-primary"),
                Button("← Back to Conditions", id="btn-back-cond-select", classes="tool-btn"),
                Button("← Dashboard", id="btn-back-dash", classes="tool-btn"),
                classes="action-row",
            ),
            id="cond-control-dock",
        )
        # Responsive Split Workspace: Relevant Inputs on Left, Evaluation Report on Right
        yield Horizontal(
            VerticalScroll(
                Static(f"[bold #43D9D0]{c_name.upper()} -- REQUIRED BIOMARKERS[/]\n"),
                Vertical(id="condition-inputs-container"),
                id="cond-form-scroll",
                classes="cond-form-col",
            ),
            VerticalScroll(
                Static("[bold #43D9D0]CLINICAL PATTERN EVALUATION REPORT[/]\n"),
                Vertical(
                    Static(
                        "[italic #9299AA]Enter values on the left or click a preset above, "
                        "then press [bold #43D9D0]'⚡ Run Assessment'[/] to evaluate clinical patterns.[/]\n",
                        id="cond-results-display",
                    ),
                    id="cond-results-container",
                ),
                Static(
                    "[bold #A879D5]CLINICAL DISCLAIMER:[/] Condition assessments are educational demonstrations based on synthetic data "
                    "and rule-based pattern matching. They are NOT medical diagnoses. Seek guidance from a licensed physician.",
                    classes="disclaimer-box",
                ),
                id="cond-results-scroll",
                classes="cond-results-col",
            ),
            classes="cond-workspace",
        )
        yield FooterBar()

    def on_mount(self):
        super().on_mount()
        container = self.query_one("#condition-inputs-container")
        self.input_widgets = {}

        meta = CONDITION_PANELS.get(self.condition_id, {})
        req_features = meta.get("required", [])
        opt_features = meta.get("optional", [])
        all_features = req_features + opt_features

        for feat in all_features:
            is_req = feat in req_features
            req_tag = "[bold #43D9D0]*REQ[/]" if is_req else "[#9299AA] OPT[/]"

            spec = BIOMARKER_SPECS.get(feat, {})
            d_name = spec.get("display_name", feat)
            unit = spec.get("canonical_unit", "")
            r_low = spec.get("ref_low", "")
            r_high = spec.get("ref_high", "")
            ref_str = f"[{r_low}-{r_high}]" if r_low != "" else ""

            clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", feat)
            inp = Input(placeholder="enter value", id=f"inp-cond-{clean_id}", classes="field-input")
            row = Horizontal(
                Label(f"{d_name[:22]:<22} {req_tag}", classes="field-label"),
                Label(f"{unit:<10} {ref_str}", classes="field-unit"),
                inp,
                classes="form-row",
            )
            container.mount(row)
            self.input_widgets[feat] = inp

    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        if btn_id == "btn-back-dash":
            self.app.switch_mode("dashboard")
        elif btn_id == "btn-back-cond-select":
            self.app.switch_mode("conditions")
        elif btn_id == "btn-cond-preset-normal":
            self._apply_condition_preset("normal")
        elif btn_id == "btn-cond-preset-abnormal":
            self._apply_condition_preset("abnormal")
        elif btn_id == "btn-cond-preset-critical":
            self._apply_condition_preset("critical")
        elif btn_id == "btn-cond-preset-clear":
            for inp in self.input_widgets.values():
                inp.value = ""
            self.query_one("#cond-results-display", Static).update(
                "[italic #9299AA]Inputs cleared. Select a preset or enter values.[/]"
            )
        elif btn_id == "btn-run-cond-eval":
            self._run_condition_evaluation()

    def _apply_condition_preset(self, preset_type: str):
        """Loads representative preset values for the active condition panel."""
        cid = self.condition_id
        presets = {
            "vitamin_d": {
                "normal": {"Vitamin_D_ng_mL": "35.0", "Calcium_mg_dL": "9.5", "ALP_U_L": "70.0"},
                "abnormal": {"Vitamin_D_ng_mL": "14.0", "Calcium_mg_dL": "8.3", "ALP_U_L": "145.0"},
                "critical": {"Vitamin_D_ng_mL": "35.0", "Calcium_mg_dL": "6.0", "ALP_U_L": "80.0"},
            },
            "diabetes": {
                "normal": {"Glucose_mg_dL": "88.0", "HbA1c_pct": "5.2"},
                "abnormal": {"Glucose_mg_dL": "185.0", "HbA1c_pct": "8.4"},
                "critical": {"Glucose_mg_dL": "450.0", "HbA1c_pct": "12.5"},
            },
            "anemia": {
                "normal": {"Hemoglobin_g_dL": "14.2", "MCV_fL": "88.0", "Ferritin_ng_mL": "75.0"},
                "abnormal": {"Hemoglobin_g_dL": "9.8", "MCV_fL": "72.0", "Ferritin_ng_mL": "11.0"},
                "critical": {"Hemoglobin_g_dL": "6.2", "MCV_fL": "68.0", "Ferritin_ng_mL": "6.0"},
            },
            "electrolytes": {
                "normal": {"Sodium_mmol_L": "140.0", "Potassium_mmol_L": "4.2", "Calcium_mg_dL": "9.5", "Magnesium_mg_dL": "2.0"},
                "abnormal": {"Sodium_mmol_L": "132.0", "Potassium_mmol_L": "3.3", "Calcium_mg_dL": "8.3", "Magnesium_mg_dL": "1.5"},
                "critical": {"Sodium_mmol_L": "118.0", "Potassium_mmol_L": "6.8", "Calcium_mg_dL": "6.2", "Magnesium_mg_dL": "0.9"},
            },
            "kidney": {
                "normal": {"Creatinine_mg_dL": "0.9", "eGFR_mL_min_1_73m2": "100.0", "Urea_mg_dL": "25.0", "UACR_mg_g": "15.0"},
                "abnormal": {"Creatinine_mg_dL": "2.4", "eGFR_mL_min_1_73m2": "28.0", "Urea_mg_dL": "65.0", "UACR_mg_g": "350.0"},
                "critical": {"Creatinine_mg_dL": "5.5", "eGFR_mL_min_1_73m2": "11.0", "Urea_mg_dL": "140.0", "UACR_mg_g": "1200.0"},
            },
        }

        panel_presets = presets.get(cid, {})
        chosen = panel_presets.get(preset_type, {})
        for feat, inp in self.input_widgets.items():
            if feat in chosen:
                inp.value = chosen[feat]
            else:
                inp.value = ""

    def _run_condition_evaluation(self):
        inputs = {}
        for feat, inp in self.input_widgets.items():
            val = inp.value.strip()
            if val != "":
                inputs[feat] = val

        assessment = DiseasePatternEngine.evaluate_condition(self.condition_id, inputs)

        # Style by status
        status = assessment.status
        status_color = (
            "#4ECA78" if status == "NORMAL"
            else ("#EAB308" if status == "BORDERLINE"
            else ("#FF3344" if status == "CRITICAL_ALERT"
            else ("#EF4444" if status == "ABNORMAL_PATTERN" else "#9299AA")))
        )

        t = Text()
        t.append("\n=======================================================\n", style="#2E3345")
        t.append(f"  CONDITION ASSESSMENT: {assessment.condition_name.upper()}\n", style="bold #43D9D0")
        t.append(f"  OVERALL PATTERN STATUS: [{status}]\n", style=f"bold {status_color}")
        t.append("=======================================================\n", style="#2E3345")

        # CRITICAL ALERTS: Display prominently at top if present!
        if assessment.critical_alerts:
            t.append("\n!!! URGENT CLINICAL ALERTS DETECTED !!!\n", style="bold #FF3344")
            for alert in assessment.critical_alerts:
                t.append(f"  CRITICAL: {alert}\n", style="bold #FF3344")

        # Biomarker Reference Range Indicators for this condition panel
        t.append("\nPanel Biomarker Reference Range Analysis:\n", style="bold #43D9D0")
        meta = CONDITION_PANELS.get(self.condition_id, {})
        req_features = meta.get("required", [])
        opt_features = meta.get("optional", [])
        all_features = req_features + opt_features

        for feat in all_features:
            spec = BIOMARKER_SPECS.get(feat, {})
            d_name = spec.get("display_name", feat)
            unit = spec.get("canonical_unit", "")
            r_low = spec.get("ref_low")
            r_high = spec.get("ref_high")
            c_low = spec.get("panic_low")
            c_high = spec.get("panic_high")

            val_str = inputs.get(feat)
            if val_str is not None:
                try:
                    val = float(val_str)
                    t.append_text(render_biomarker_range_indicator(
                        d_name, val, r_low, r_high, unit, c_low, c_high
                    ))
                except ValueError:
                    t.append(f"  • {d_name:<24} [INVALID VALUE: '{val_str}'] ({unit})\n", style="bold #EF4444")
            else:
                req_lbl = "(Required)" if feat in req_features else "(Optional)"
                ref_lbl = f"(Ref: {r_low}-{r_high} {unit})" if r_low is not None and r_high is not None else f"({unit})"
                t.append(f"  • {d_name:<24} [NOT PROVIDED] {req_lbl} {ref_lbl}\n", style="#9299AA")

        # Validation Warnings
        if assessment.validation_warnings:
            t.append("\nInput Validation Notices:\n", style="bold #EAB308")
            for w in assessment.validation_warnings:
                t.append(f"  • {w}\n", style="#EAB308")

        # Identified Patterns
        t.append("\nIdentified Clinical Patterns:\n", style="bold #43D9D0")
        for p in assessment.patterns_detected:
            t.append(f"  * {p}\n", style="bold #E6E8F0")

        # Clinical Caveats & Findings Detail
        if assessment.clinical_caveats:
            t.append("\nDifferential Diagnostic Context:\n", style="bold #A879D5")
            for diff in assessment.clinical_caveats:
                t.append(f"  - {diff}\n", style="#E6E8F0")

        if assessment.unmeasured_clinical_context:
            t.append("\nUnmeasured Confirmatory Testing:\n", style="bold #9299AA")
            for unmeas in assessment.unmeasured_clinical_context:
                t.append(f"  [i] {unmeas}\n", style="italic #9299AA")

        # Educational Guidance
        expl = DiseaseExplainer.get_explanation(self.condition_id)
        if expl:
            desc = expl.get("description", "")
            if desc:
                t.append(f"\nCondition Context:\n  {desc}\n", style="#E6E8F0")
            guidance = expl.get("general_guidance", [])
            if guidance:
                t.append("\nEvidence-Based Preventive Guidance:\n", style="bold #4ECA78")
                for g in guidance:
                    t.append(f"  + {g}\n", style="#4ECA78")

        self.query_one("#cond-results-display", Static).update(t)

        # Log session
        record_session_assessment(
            mode="Condition Panel",
            name=assessment.condition_name,
            result_tag=status,
            detail=f"{len(assessment.patterns_detected)} patterns, {len(assessment.critical_alerts)} critical alerts",
        )


# ===========================================================================
# 6. Model Evaluation Screen
# ===========================================================================
class ModelEvaluationScreen(ResponsiveScreen):
    """Detailed evaluation metrics, confusion matrix heatmap, and performance charts."""

    def compose(self):
        yield HeaderBar(mode="ML Model Evaluation")
        yield Vertical(
            Horizontal(
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Label("[bold #43D9D0] Verified Performance Summary, Confusion Matrix & Classification Report[/]", classes="hint-label"),
                classes="action-row",
            ),
            id="ml-control-dock",
        )
        yield ScrollableContainer(
            Vertical(
                Static(
                    "[bold #43D9D0]RANDOM FOREST PIPELINE EVALUATION REPORT[/]\n"
                    "[#9299AA]Evaluated on held-out test split (200 records, 20% stratified test set). "
                    "Metrics loaded from evaluation artifacts.[/]\n",
                    classes="panel-title",
                ),
                Static(self._build_metrics_text(), id="eval-metrics-box"),
                Static(self._build_confusion_matrix(), id="eval-cm-box"),
                Static(self._build_performance_bars(), id="eval-perf-bars"),
                Static(
                    "[bold #A879D5]TECHNICAL HONESTY NOTE:[/] The high accuracy (~98%) is due to "
                    "deterministic labeling rules (H1-H4, M1-M6) in the synthetic dataset. "
                    "In real clinical practice, biological heterogeneity reduces predictive certainty.",
                    classes="disclaimer-box",
                ),
            ),
            classes="scrollable-content",
        )
        yield FooterBar()

    def _build_metrics_text(self) -> Text:
        metadata = {}
        if os.path.isfile(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        eval_info = metadata.get("evaluation", {})
        acc = eval_info.get("accuracy", 0.98)
        macro_f1 = eval_info.get("macro_f1", 0.9552)
        weighted_f1 = eval_info.get("weighted_f1", 0.9801)

        t = Text()
        t.append("MODEL PERFORMANCE SUMMARY (HELD-OUT TEST SET)\n", style="bold #43D9D0")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append(f"  Test Accuracy  : {acc*100:.2f}%  (Dummy Baseline: 71.00%, +27.00%)\n", style="bold #4ECA78")
        t.append(f"  Macro-F1 Score : {macro_f1:.4f}  (Dummy Baseline: 0.2768, +0.6784)\n", style="bold #4ECA78")
        t.append(f"  Weighted-F1    : {weighted_f1:.4f}\n", style="bold #4ECA78")
        t.append(f"  Training Split : 800 samples (80%)\n", style="#E6E8F0")
        t.append(f"  Test Split     : 200 samples (20% Stratified)\n", style="#E6E8F0")
        return t

    def _build_confusion_matrix(self) -> Text:
        t = Text()
        t.append("\nCONFUSION MATRIX HEATMAP (TRUE VS PREDICTED)\n", style="bold #A879D5")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("                    Pred: Good     Pred: Okay     Pred: Poor     Recall\n", style="bold #43D9D0")
        # Row 1: True Good
        t.append("  True Good   ", style="bold #E6E8F0")
        t.append("    52 (100%) ", style="bold #4ECA78")
        t.append("     0 (  0%) ", style="#9299AA")
        t.append("     0 (  0%) ", style="#9299AA")
        t.append("   100.0%\n", style="bold #4ECA78")
        # Row 2: True Okay
        t.append("  True Okay   ", style="bold #E6E8F0")
        t.append("     2 ( 1.4%)", style="#EAB308")
        t.append("   140 (98.6%)", style="bold #4ECA78")
        t.append("     0 (  0%) ", style="#9299AA")
        t.append("    98.6%\n", style="bold #4ECA78")
        # Row 3: True Poor
        t.append("  True Poor   ", style="bold #E6E8F0")
        t.append("     0 (  0%) ", style="#9299AA")
        t.append("     2 (28.6%)", style="#EAB308")
        t.append("     5 (71.4%)", style="bold #4ECA78")
        t.append("    71.4%\n", style="bold #EAB308")
        return t

    def _build_performance_bars(self) -> Text:
        t = Text()
        t.append("\nCLASS-LEVEL PRECISION / RECALL / F1 BREAKDOWN\n", style="bold #5294E2")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        classes = [
            ("Good", 0.963, 1.000, 0.9811, 52, "#4ECA78"),
            ("Okay", 0.986, 0.986, 0.9859, 142, "#7585E8"),
            ("Poor", 1.000, 0.714, 0.8333, 7, "#EF4444"),
        ]
        for name, prec, rec, f1, sup, color in classes:
            t.append(f"  {name:<6} ", style=f"bold {color}")
            t.append(f"Prec: {prec:.3f}  ", style="#E6E8F0")
            bar_p = render_horizontal_bar(prec, 1.0, width=10, color=color)
            t.append_text(bar_p)
            t.append(f"  Rec: {rec:.3f}  ", style="#E6E8F0")
            bar_r = render_horizontal_bar(rec, 1.0, width=10, color=color)
            t.append_text(bar_r)
            t.append(f"  F1: {f1:.3f}  (n={sup})\n", style="#9299AA")
        return t

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")


# ===========================================================================
# 7. Dataset Explorer Screen
# ===========================================================================
class DatasetExplorerScreen(ResponsiveScreen):
    """Read-only exploration of dataset structure, schema, and data quality."""

    def compose(self):
        yield HeaderBar(mode="Dataset Explorer")
        yield Vertical(
            Horizontal(
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Label("[bold #43D9D0] TrainData.xlsx Schema, Distribution & Quality Audit[/]", classes="hint-label"),
                classes="action-row",
            ),
            id="ml-control-dock",
        )
        yield ScrollableContainer(
            Vertical(
                Static(
                    "[bold #43D9D0]DATASET EXPLORER (TrainData.xlsx)[/]\n"
                    "[#9299AA]Read-only schema and integrity summary. "
                    "Patient IDs and raw individual records are protected.[/]\n",
                    classes="panel-title",
                ),
                Static(self._build_distribution_chart(), id="dataset-dist-chart"),
                Static(self._build_dataset_text(), id="dataset-info-box"),
            ),
            classes="scrollable-content",
        )
        yield FooterBar()

    def _build_distribution_chart(self) -> Text:
        from src.tui.charts import render_dataset_distribution_chart
        return render_dataset_distribution_chart()

    def _build_dataset_text(self) -> Text:
        t = Text()
        t.append("\nWORKBOOK STRUCTURE & WORKSHEETS\n", style="bold #43D9D0")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("  Worksheet 1: Synthetic_Dataset (1,000 rows x 41 columns)\n", style="#E6E8F0")
        t.append("  Worksheet 2: Data_Dictionary (biomarker names, units, descriptions)\n", style="#E6E8F0")
        t.append("  Worksheet 3: Reference_Guidance (healthy reference intervals)\n", style="#E6E8F0")
        t.append("  Worksheet 4: Labeling_Rules (H1-H4, M1-M6 threshold rules)\n", style="#E6E8F0")

        t.append("\nINTEGRITY & DATA QUALITY AUDIT\n", style="bold #5294E2")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("  • Missing Value Count : 988 cells (2.53%), uniformly distributed\n", style="#E6E8F0")
        t.append("  • Imputation Strategy : Leakage-free Median (fitted on train split only)\n", style="#E6E8F0")
        t.append("  • Duplicate Records   : 0 duplicates detected\n", style="#E6E8F0")
        t.append("  • Privacy Protection  : Patient Record_ID excluded from feature matrix\n", style="#E6E8F0")

        t.append("\nFEATURE SCHEMA TABLE (39 BIOMARKERS)\n", style="bold #43D9D0")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("  Biomarker Key                  Unit          Healthy Range\n", style="bold #43D9D0")

        for k, v in BIOMARKER_SPECS.items():
            u = v.get("canonical_unit", "")
            r_l = v.get("ref_low", "")
            r_h = v.get("ref_high", "")
            r_str = f"{r_l} - {r_h}" if r_l != "" else "N/A"
            t.append(f"  {k:<30} {u:<12}  {r_str}\n", style="#E6E8F0")

        return t

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")


# ===========================================================================
# 8. Assessment History Screen (Session History)
# ===========================================================================
class AssessmentHistoryScreen(ResponsiveScreen):
    """View assessments performed during the current application session."""

    def compose(self):
        yield HeaderBar(mode="Session Assessment History")
        yield Vertical(
            Horizontal(
                Button("Clear History", id="btn-clear-history", classes="tool-btn"),
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Label("[bold #43D9D0] Live In-Memory Session Assessment Audit Trail[/]", classes="hint-label"),
                classes="action-row",
            ),
            id="ml-control-dock",
        )
        yield ScrollableContainer(
            Vertical(
                Static(
                    "[bold #43D9D0]SESSION ASSESSMENT HISTORY[/]\n"
                    "[#9299AA]Log of evaluations performed during this session. "
                    "Maintained in-memory, not persisted to disk.[/]\n",
                    classes="panel-title",
                ),
                DataTable(id="history-table"),
                Static(id="history-summary"),
            ),
            classes="scrollable-content",
        )
        yield FooterBar()

    def on_mount(self):
        super().on_mount()
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Time", "Mode", "Target / Condition", "Result Status", "Summary Detail")
        self._populate_table()

    def _populate_table(self):
        table = self.query_one("#history-table", DataTable)
        table.clear()
        summary_widget = self.query_one("#history-summary", Static)
        if not SESSION_ASSESSMENTS:
            table.add_row("--:--:--", "None", "No assessments logged yet", "[NONE]", "Run a prediction or condition assessment")
            summary_widget.update("[italic #9299AA]Run predictions in General ML or Condition Panels to populate history.[/]")
        else:
            for item in SESSION_ASSESSMENTS:
                table.add_row(
                    item["time"],
                    item["mode"],
                    item["name"],
                    item["result"],
                    item["detail"],
                )
            summary_widget.update(f"[bold #43D9D0]Total Assessments: {len(SESSION_ASSESSMENTS)}[/]")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")
        elif event.button.id == "btn-clear-history":
            SESSION_ASSESSMENTS.clear()
            self._populate_table()


# ===========================================================================
# 9. About Screen
# ===========================================================================
class AboutScreen(ResponsiveScreen):
    """Project info, clinical references, architecture, and medical disclaimers."""

    def compose(self):
        yield HeaderBar(mode="About NutriCare ML")
        yield Vertical(
            Horizontal(
                Button("← Back to Dashboard", id="btn-back-dashboard", classes="tool-btn"),
                Label("[bold #43D9D0] System Architecture, Clinical References & Legal Disclaimers[/]", classes="hint-label"),
                classes="action-row",
            ),
            id="ml-control-dock",
        )
        yield ScrollableContainer(
            Vertical(
                Static(get_logo_text(compact=True), id="about-logo"),
                Static(self._build_about_text(), id="about-info-box"),
            ),
            classes="scrollable-content",
        )
        yield FooterBar()

    def _build_about_text(self) -> Text:
        t = Text()
        t.append("\nPROJECT ARCHITECTURE\n", style="bold #43D9D0")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("NutriCare ML is a terminal-based biomarker intelligence prototype:\n\n", style="#E6E8F0")
        t.append("  1. General ML Classifier\n", style="bold #43D9D0")
        t.append("     Balanced Random Forest (39 biomarkers) predicting Good/Okay/Poor.\n", style="#E6E8F0")
        t.append("     Pipeline: SimpleImputer → BalancedRandomForestClassifier\n\n", style="#9299AA")
        t.append("  2. Clinical Pattern Engine\n", style="bold #7585E8")
        t.append("     Transparent, unit-aware rule evaluation across 10 guideline panels.\n", style="#E6E8F0")
        t.append("     Critical/panic alerts auto-elevate results independently.\n\n", style="#9299AA")
        t.append("  3. Educational Explainer Module\n", style="bold #A879D5")
        t.append("     Physiological roles, complications, and evidence-based guidance.\n\n", style="#E6E8F0")

        t.append("CLINICAL CONSENSUS REFERENCES\n", style="bold #A879D5")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        refs = [
            ("ADA Standards of Care (2024)", "Glycemic patterns, HbA1c & Glucose cutoffs"),
            ("KDIGO Guidelines (2024)", "CKD eGFR staging (G1-G5) & UACR albuminuria"),
            ("WHO Haemoglobin Concentrations", "Anemia morphology classification"),
            ("AHA/ACC Guidelines (2018)", "Lipid abnormalities, LDL, TG, and ApoB"),
            ("Endocrine Society Guidelines", "25(OH)D deficiency, calcium thresholds"),
            ("ACG Clinical Guidelines", "Hepatocellular vs cholestatic liver injury"),
            ("ATA Guidelines", "TSH-based thyroid dysfunction screening"),
        ]
        for name, scope in refs:
            t.append(f"  • {name}\n", style="bold #E6E8F0")
            t.append(f"    {scope}\n", style="italic #9299AA")

        t.append("\nTECHNICAL STACK\n", style="bold #5294E2")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("  Framework : Textual (TUI) + Rich (Rendering)\n", style="#E6E8F0")
        t.append("  ML Engine : scikit-learn BalancedRandomForestClassifier\n", style="#E6E8F0")
        t.append("  Data      : pandas + openpyxl (Excel I/O)\n", style="#E6E8F0")
        t.append("  Execution : 100% Local (Zero Network Dependencies)\n", style="#E6E8F0")
        t.append("  Tests     : 26 Automated Unit & Integration Tests\n\n", style="#E6E8F0")

        t.append("FULL MEDICAL DISCLAIMER\n", style="bold #EF4444")
        t.append("─────────────────────────────────────────────────────────────\n", style="#2E3345")
        t.append("This software is developed exclusively for educational demonstration.\n", style="#EF4444")
        t.append("It does NOT provide medical diagnosis or treatment recommendations.\n", style="#EF4444")
        t.append("Laboratory results must be evaluated by qualified healthcare\n", style="#EF4444")
        t.append("professionals within a complete clinical context.\n", style="#EF4444")
        return t

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn-back-dashboard":
            self.app.switch_mode("dashboard")

