#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log'd (Lite)
Time Tracker with Projects (single-file)

- Python 3.x
- PyQt6
- SQLite (app.db in working directory)

Run:
    pip install PyQt6
    python logd_lite.py
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

from PyQt6.QtCore import Qt, QDate, QTimer
from PyQt6.QtGui import QAction, QIcon, QFontMetrics
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QDateEdit, QComboBox, QDialog,
    QFormLayout, QDialogButtonBox, QSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QGroupBox, QSystemTrayIcon, QMenu
)

# =========================
# Module-level global state
# =========================
active_task_id: Optional[int] = None          # None when no task active or paused; otherwise stores current/last task ID
active_session_id: Optional[int] = None       # ID of currently running session; None if paused or no session
stored_time_seconds: int = 0                  # Accumulates elapsed time for current/last active task
ALWAYS_ON_TOP: bool = False

DB_PATH = "app.db"

# =========================
# Helper functions
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_iso_utc(ts_text: Optional[str]) -> Optional[datetime]:
    """
    Returns timezone-aware UTC datetime from ISO 8601 string (accepts trailing 'Z' or +00:00).
    Accepts None and returns None.
    """
    if ts_text is None:
        return None
    s = ts_text.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def fmt_hhmmss(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def to_local_str(dt_utc: Optional[datetime]) -> str:
    if dt_utc is None:
        return ""
    return dt_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")

def overlap_seconds(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> int:
    """
    Compute overlap (seconds) between intervals [a_start,a_end) and [b_start,b_end).
    """
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0
    return int((end - start).total_seconds())

# =========================
# Database layer
# =========================
class DB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_schema()

    def _ensure_schema(self):
        # projects, tasks (per-project unique), sessions
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            number TEXT NOT NULL UNIQUE,
            city TEXT,
            state TEXT NOT NULL,
            client TEXT,
            type TEXT NOT NULL CHECK(type IN ('New','Remodel')),
            scope_area INTEGER NOT NULL,
            discipline_plumbing INTEGER NOT NULL DEFAULT 0,
            discipline_hvac INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE RESTRICT,
            task TEXT NOT NULL COLLATE NOCASE,
            archived INTEGER NOT NULL DEFAULT 0,
            UNIQUE(project_id, task)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL REFERENCES tasks(id),
            start_time TEXT NOT NULL,
            end_time TEXT
        );
        """)
        self.conn.commit()

    # ---- Projects ----
    def add_project(self, name: str, number: str, city: str, state: str, client: str,
                    ptype: str, scope_area: int, plumbing: bool, hvac: bool) -> Tuple[bool, Optional[str]]:
        try:
            self.conn.execute("""
                INSERT INTO projects (name, number, city, state, client, type, scope_area, discipline_plumbing, discipline_hvac)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name.strip(), number.strip(), city.strip(), state.strip(), client.strip(), ptype, int(scope_area), int(plumbing), int(hvac)))
            self.conn.commit()
            return True, None
        except sqlite3.IntegrityError as e:
            # Could be duplicate name or number
            msg = str(e)
            if "projects.name" in msg or "UNIQUE" in msg:
                return False, "Project name or number must be unique."
            return False, "Integrity error."
        except sqlite3.Error as e:
            return False, str(e)

    def delete_project(self, project_id: int) -> Tuple[bool, Optional[str]]:
        # Only allow delete if no tasks reference it
        cur = self.conn.execute("SELECT COUNT(*) AS cnt FROM tasks WHERE project_id = ?", (project_id,))
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            return False, "Cannot delete a project that has tasks."
        try:
            self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self.conn.commit()
            return True, None
        except sqlite3.Error as e:
            return False, str(e)

    def projects_all(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM projects ORDER BY name COLLATE NOCASE ASC")
        return cur.fetchall()

    def get_project(self, project_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        return cur.fetchone()

    # ---- Tasks ----
    def add_task(self, project_id: int, name: str) -> Tuple[bool, Optional[str]]:
        try:
            self.conn.execute("INSERT INTO tasks (project_id, task) VALUES (?, ?)", (project_id, name.strip()))
            self.conn.commit()
            return True, None
        except sqlite3.IntegrityError as e:
            return False, "Task name must be unique within the project."
        except sqlite3.Error as e:
            return False, str(e)

    def soft_delete_task(self, task_id: int):
        self.conn.execute("UPDATE tasks SET archived = 1 WHERE id = ?", (task_id,))
        self.conn.commit()

    def non_archived_tasks(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("""
            SELECT t.id, t.task, t.project_id, p.name AS project_name
            FROM tasks t
            JOIN projects p ON p.id = t.project_id
            WHERE t.archived = 0
        """)
        return cur.fetchall()

    def get_task(self, task_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("""
            SELECT t.*, p.name AS project_name, p.number AS project_number
            FROM tasks t
            LEFT JOIN projects p ON p.id = t.project_id
            WHERE t.id = ?
        """, (task_id,))
        return cur.fetchone()

    # ---- Sessions ----
    def start_session(self, task_id: int) -> int:
        now_iso = utc_now_iso()
        cur = self.conn.execute("INSERT INTO sessions (task_id, start_time, end_time) VALUES (?, ?, NULL)", (task_id, now_iso))
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int):
        now_iso = utc_now_iso()
        self.conn.execute("UPDATE sessions SET end_time = ? WHERE id = ?", (now_iso, session_id))
        self.conn.commit()

    def fetch_sessions_overlapping(self, start_inclusive_local: datetime, end_inclusive_local: datetime) -> List[sqlite3.Row]:
        """
        Fetch sessions that overlap the inclusive local date bounds.
        We'll convert local inclusive dates to UTC window and select sessions where start < window_end and (end IS NULL OR end > window_start).
        """
        # make inclusive window: start at 00:00:00 local, end at 23:59:59 local
        start_local = start_inclusive_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = end_inclusive_local.replace(hour=23, minute=59, second=59, microsecond=0)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = (end_local + timedelta(seconds=1)).astimezone(timezone.utc)  # exclusive

        start_iso = start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_iso = end_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        cur = self.conn.execute("""
            SELECT s.id, s.task_id, s.start_time, s.end_time, t.task, t.project_id, p.name AS project_name
            FROM sessions s
            JOIN tasks t ON t.id = s.task_id
            LEFT JOIN projects p ON p.id = t.project_id
            WHERE (s.start_time < ?)
              AND (s.end_time IS NULL OR s.end_time > ?)
            ORDER BY s.start_time ASC
        """, (end_iso, start_iso))
        return cur.fetchall()

    def sessions_for_last_7_days(self) -> List[sqlite3.Row]:
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(days=7)
        start_iso = window_start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_iso = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        cur = self.conn.execute("""
            SELECT s.id, s.task_id, s.start_time, s.end_time, t.task, t.project_id, p.name AS project_name
            FROM sessions s
            JOIN tasks t ON t.id = s.task_id
            LEFT JOIN projects p ON p.id = t.project_id
            WHERE (s.end_time IS NULL OR s.end_time >= ?)
              AND s.start_time <= ?
        """, (start_iso, end_iso))
        return cur.fetchall()

# =========================
# UI Components
# =========================

@dataclass
class TaskListItemData:
    task_id: int
    task_name: str
    project_name: str
    total_seconds_7d: int

class ProjectsTab(QWidget):
    def __init__(self, db: DB, parent: 'MainWindow'):
        super().__init__()
        self.db = db
        self.parent = parent

        # List
        self.list_projects = QListWidget(self)
        self.btn_new = QPushButton("New Project", self)
        self.btn_delete = QPushButton("Delete Project", self)
        self.btn_delete.setEnabled(False)

        # Layout
        root = QVBoxLayout(self)
        root.addWidget(self.list_projects, 1)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_delete)
        root.addLayout(btn_row)

        # Signals
        self.btn_new.clicked.connect(self.open_new_project_dialog)
        self.btn_delete.clicked.connect(self.on_delete_project)
        self.list_projects.itemSelectionChanged.connect(self.on_selection_changed)

        self.refresh_projects()

    def refresh_projects(self):
        self.list_projects.blockSignals(True)
        self.list_projects.clear()
        for p in self.db.projects_all():
            disciplines = []
            if p["discipline_plumbing"]:
                disciplines.append("Plumbing")
            if p["discipline_hvac"]:
                disciplines.append("HVAC")
            disc = ", ".join(disciplines) if disciplines else ""
            display = f"{p['name']}  |  {p['number']}  |  {p['city'] or ''}, {p['state']}  |  {p['client'] or ''}  |  {p['type']}  |  {p['scope_area']} sqft  |  {disc}"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            self.list_projects.addItem(item)
        self.list_projects.blockSignals(False)
        self.on_selection_changed()

    def on_selection_changed(self):
        has = self.list_projects.currentRow() >= 0
        self.btn_delete.setEnabled(has)

    def open_new_project_dialog(self):
        dlg = NewProjectDialog(self.db, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # If added, refresh list
            self.refresh_projects()

    def on_delete_project(self):
        item = self.list_projects.currentItem()
        if not item:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        pname = self.db.get_project(pid)["name"] if self.db.get_project(pid) else "(unknown)"
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f'Are you sure you want to delete project "{pname}"?')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Ok:
            ok, err = self.db.delete_project(pid)
            if not ok:
                QMessageBox.critical(self, "Delete Failed", err)
            self.refresh_projects()

class NewProjectDialog(QDialog):
    STATES = [
        # WA, ID, OR first as requested
        "WA", "ID", "OR"
    ] + sorted([
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","IL","IN","KS","KY","LA",
        "MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY",
        "OH","OK","PA","RI","SC","SD","TN","TX","UT","VT","VA","WV","WI","WY"
    ])

    def __init__(self, db: DB, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("New Project")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.input_name = QLineEdit(self)
        self.input_number = QLineEdit(self)
        self.input_city = QLineEdit(self)
        self.combo_state = QComboBox(self)
        self.combo_state.addItems(self.STATES)
        self.input_client = QLineEdit(self)

        # Project type radio buttons
        self.radio_new = QRadioButton("New")
        self.radio_remodel = QRadioButton("Remodel")
        self.radio_new.setChecked(True)
        type_group = QHBoxLayout()
        type_group.addWidget(self.radio_new)
        type_group.addWidget(self.radio_remodel)
        type_box = QWidget(self)
        type_box.setLayout(type_group)

        self.input_scope = QLineEdit(self)
        self.input_scope.setPlaceholderText("numbers only (sqft)")

        # Disciplines
        self.chk_plumbing = QCheckBox("Plumbing")
        self.chk_hvac = QCheckBox("HVAC")
        disc_layout = QHBoxLayout()
        disc_layout.addWidget(self.chk_plumbing)
        disc_layout.addWidget(self.chk_hvac)
        disc_widget = QWidget(self)
        disc_widget.setLayout(disc_layout)

        form.addRow("Project Name", self.input_name)
        form.addRow("Number (####-###)", self.input_number)
        form.addRow("City", self.input_city)
        form.addRow("State", self.combo_state)
        form.addRow("Client", self.input_client)
        form.addRow("Project type", type_box)
        # scope with "sqft" label at end: we'll place input and label
        scope_row = QHBoxLayout()
        scope_row.addWidget(self.input_scope, 1)
        scope_row.addWidget(QLabel("sqft"))
        scope_widget = QWidget(self)
        scope_widget.setLayout(scope_row)
        form.addRow("Scope area", scope_widget)
        form.addRow("Disciplines", disc_widget)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancel")
        buttons.accepted.connect(self.on_add)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_add(self):
        name = self.input_name.text().strip()
        number = self.input_number.text().strip()
        city = self.input_city.text().strip()
        state = self.combo_state.currentText().strip()
        client = self.input_client.text().strip()
        ptype = "New" if self.radio_new.isChecked() else "Remodel"
        scope_text = self.input_scope.text().strip()
        plumbing = self.chk_plumbing.isChecked()
        hvac = self.chk_hvac.isChecked()

        # Validations
        if not name:
            QMessageBox.warning(self, "Validation", "Project Name is required.")
            return
        if not number or not re.fullmatch(r"\d{4}-\d{3}", number):
            QMessageBox.warning(self, "Validation", "Number must match format ####-###.")
            return
        if not state:
            QMessageBox.warning(self, "Validation", "State is required.")
            return
        if not scope_text or not scope_text.isdigit():
            QMessageBox.warning(self, "Validation", "Scope area must be numeric (sqft).")
            return
        scope_area = int(scope_text)

        ok, err = self.db.add_project(name, number, city, state, client, ptype, scope_area, plumbing, hvac)
        if not ok:
            QMessageBox.critical(self, "Add Failed", err or "Failed to add project.")
            return
        self.accept()

class AutoResizeCombo(QComboBox):
    def showPopup(self):
        """Resize dropdown width to fit the longest item text when opening"""
        metrics = QFontMetrics(self.font())
        max_width = max(metrics.horizontalAdvance(self.itemText(i)) for i in range(self.count())) if self.count() > 0 else 0
        self.view().setMinimumWidth(max_width + 40)  # +40 for arrow + padding
        super().showPopup()


class TasksTab(QWidget):
    def __init__(self, db: DB, parent: 'MainWindow'):
        super().__init__()
        self.db = db
        self.parent = parent

        # Top: task input + project dropdown + Add/Delete
        self.input_task = QLineEdit(self)
        self.input_task.setPlaceholderText("Type task name...")
        self.combo_projects = AutoResizeCombo(self)
        self.combo_projects.setPlaceholderText("Select Project")
        self.btn_add_task = QPushButton("Add Task", self)
        self.btn_delete_task = QPushButton("Delete Task", self)
        self.btn_delete_task.setEnabled(False)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Task"))
        top_row.addWidget(self.input_task, 1)
        top_row.addWidget(QLabel("Project"))
        top_row.addWidget(self.combo_projects)
        top_row.addWidget(self.btn_add_task)
        top_row.addWidget(self.btn_delete_task)

        # Middle: task list
        self.list_tasks = QListWidget(self)

        # Bottom: status and controls
        self.status_label = QLabel("No task active", self)
        mono = self.status_label.font()
        mono.setFamily("Consolas")
        mono.setPointSize(mono.pointSize() + 5)
        self.status_label.setFont(mono)
        self.btn_toggle = QPushButton("Start", self)
        self.btn_end_day = QPushButton("End Day", self)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.status_label, 1)
        bottom_row.addWidget(self.btn_toggle)
        bottom_row.addWidget(self.btn_end_day)

        # Layout
        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(self.list_tasks, 1)
        root.addLayout(bottom_row)

        # Signals
        self.btn_add_task.clicked.connect(self.on_add_task)
        self.btn_delete_task.clicked.connect(self.on_delete_task)
        self.list_tasks.itemSelectionChanged.connect(self.on_selection_changed)
        self.btn_toggle.clicked.connect(self.on_toggle_clicked)
        self.btn_end_day.clicked.connect(self.on_end_day)

        # Populate projects & tasks
        self.refresh_projects_dropdown()
        self.refresh_task_list()

    def refresh_projects_dropdown(self):
        self.combo_projects.blockSignals(True)
        self.combo_projects.clear()
        projects = self.db.projects_all()
        for p in projects:
            self.combo_projects.addItem(p["name"], p["id"])
        self.combo_projects.blockSignals(False)

    def on_add_task(self):
        project_index = self.combo_projects.currentIndex()
        if project_index < 0:
            QMessageBox.information(self, "No Project", "Please select a project to add the task to.")
            return
        project_id = self.combo_projects.currentData()
        name = self.input_task.text().strip()
        if not name:
            return
        ok, err = self.db.add_task(project_id, name)
        if not ok:
            QMessageBox.warning(self, "Duplicate Task", "Task name must be unique within the selected project.")
            self.input_task.clear()
            return
        self.input_task.clear()
        # Refresh task list and keep selection on new task
        self.refresh_task_list()

    def on_delete_task(self):
        item = self.list_tasks.currentItem()
        if not item:
            return
        data: TaskListItemData = item.data(Qt.ItemDataRole.UserRole)
        # confirm
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f'Are you sure you want to delete "{data.task_name}" (Project: {data.project_name})?')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Ok:
            self.db.soft_delete_task(data.task_id)
            self.refresh_task_list()

    def on_selection_changed(self):
        has = self.list_tasks.currentRow() >= 0
        self.btn_delete_task.setEnabled(has)

    def selected_task_id(self) -> Optional[int]:
        item = self.list_tasks.currentItem()
        if not item:
            return None
        data: TaskListItemData = item.data(Qt.ItemDataRole.UserRole)
        return data.task_id

    # ---- Timer / Start-Pause-Switch logic ----
    def on_toggle_clicked(self):
        global active_task_id, active_session_id, stored_time_seconds

        selected = self.selected_task_id()

        # Case: No session active (Start pressed)
        if active_session_id is None:
            if selected is None and active_task_id is None:
                QMessageBox.information(self, "No Selection", "No task selected. Please select a task.")
                return

            target_task = selected if selected is not None else active_task_id

            if active_task_id is None or active_task_id == target_task:
                # Resume same task or start selected task; do not reset stored_time_seconds
                active_task_id = target_task
                active_session_id = self.db.start_session(active_task_id)
                self.update_status_active()
                self.btn_toggle.setText("Pause/Switch Task")
            else:
                # active_task_id different from selected: reset stored_time_seconds
                active_task_id = target_task
                stored_time_seconds = 0
                active_session_id = self.db.start_session(active_task_id)
                self.update_status_active()
                self.btn_toggle.setText("Pause/Switch Task")
        else:
            # Session active -> Pause or Switch
            if (selected is None) or (selected == active_task_id):
                # Pause
                try:
                    self.db.end_session(active_session_id)
                finally:
                    active_session_id = None
                self.update_status_paused()
                self.btn_toggle.setText("Start")
            else:
                # Switch to different task
                try:
                    self.db.end_session(active_session_id)
                finally:
                    active_session_id = None
                stored_time_seconds = 0
                active_task_id = selected
                active_session_id = self.db.start_session(active_task_id)
                self.update_status_active()
                self.btn_toggle.setText("Pause/Switch Task")

        # Refresh ordering
        self.refresh_task_list(preserve_selection=True)

    def on_end_day(self):
        global active_task_id, active_session_id, stored_time_seconds
        if active_session_id is not None:
            try:
                self.db.end_session(active_session_id)
            finally:
                active_session_id = None
        active_task_id = None
        stored_time_seconds = 0
        self.status_label.setText("No task active")
        self.btn_toggle.setText("Start")

    def update_status_active(self):
        global active_task_id, stored_time_seconds
        task_name = "(unknown)"
        project_name = ""
        if active_task_id is not None:
            row = self.db.get_task(active_task_id)
            if row:
                task_name = row["task"]
                project_name = row["project_name"] or ""
        display = f"{project_name} — {task_name} — {fmt_hhmmss(stored_time_seconds)}" if project_name else f"{task_name} — {fmt_hhmmss(stored_time_seconds)}"
        self.status_label.setText(display)

    def update_status_paused(self):
        global active_task_id, stored_time_seconds
        task_name = "(unknown)"
        project_name = ""
        if active_task_id is not None:
            row = self.db.get_task(active_task_id)
            if row:
                task_name = row["task"]
                project_name = row["project_name"] or ""
        display_body = f"{project_name} — {task_name} — {fmt_hhmmss(stored_time_seconds)}" if project_name else f"{task_name} — {fmt_hhmmss(stored_time_seconds)}"
        self.status_label.setText(f"Paused: {display_body}")

    def timer_tick(self):
        """Called each second by MainWindow."""
        global active_session_id, stored_time_seconds
        if active_session_id is not None:
            stored_time_seconds += 1
            self.update_status_active()
        self.refresh_task_list(preserve_selection=True)

    def refresh_task_list(self, preserve_selection: bool = False):
        """
        Show non-archived tasks ordered by total time spent in last 7 days (descending),
        ties broken alphabetically by project then task.
        Display format: "ProjectName — TaskName  (HH:MM:SS)"
        """
        prev_selected = self.selected_task_id() if preserve_selection else None

        # compute totals
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(days=7)
        sessions = self.db.sessions_for_last_7_days()
        totals = {}  # task_id -> seconds
        for s in sessions:
            tid = s["task_id"]
            st = parse_iso_utc(s["start_time"])
            et = parse_iso_utc(s["end_time"]) or now_utc
            sec = overlap_seconds(st, et, window_start, now_utc)
            if sec > 0:
                totals[tid] = totals.get(tid, 0) + sec

        tasks = self.db.non_archived_tasks()
        items: List[TaskListItemData] = []
        for t in tasks:
            tid = t["id"]
            items.append(TaskListItemData(tid, t["task"], t["project_name"], totals.get(tid, 0)))

        # sort by total desc, then project asc, then task asc
        items.sort(key=lambda d: (-d.total_seconds_7d, d.project_name.lower(), d.task_name.lower()))

        # rebuild list
        self.list_tasks.blockSignals(True)
        self.list_tasks.clear()
        for d in items:
            item_text = f"{d.project_name} — {d.task_name}  ({fmt_hhmmss(d.total_seconds_7d)})"
            li = QListWidgetItem(item_text)
            li.setData(Qt.ItemDataRole.UserRole, d)
            self.list_tasks.addItem(li)
        self.list_tasks.blockSignals(False)

        # restore selection
        if prev_selected is not None:
            for i in range(self.list_tasks.count()):
                it = self.list_tasks.item(i)
                dd: TaskListItemData = it.data(Qt.ItemDataRole.UserRole)
                if dd.task_id == prev_selected:
                    self.list_tasks.setCurrentRow(i)
                    break

# =========================
# Reports tab
# =========================
class ReportsTab(QWidget):
    def __init__(self, db: DB):
        super().__init__()
        self.db = db

        self.start_date = QDateEdit(self)
        self.start_date.setCalendarPopup(True)
        self.end_date = QDateEdit(self)
        self.end_date.setCalendarPopup(True)
        today = QDate.currentDate()
        self.start_date.setDate(today)
        self.end_date.setDate(today)

        self.btn_run = QPushButton("Run Report", self)
        self.btn_export = QPushButton("Export CSV", self)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Session ID", "Task", "Start Time", "End Time", "Elapsed"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        fl = QHBoxLayout()
        fl.addWidget(QLabel("Start Date"))
        fl.addWidget(self.start_date)
        fl.addSpacing(16)
        fl.addWidget(QLabel("End Date"))
        fl.addWidget(self.end_date)
        fl.addStretch(1)
        fl.addWidget(self.btn_run)
        fl.addWidget(self.btn_export)

        root = QVBoxLayout(self)
        root.addLayout(fl)
        root.addWidget(self.table, 1)

        self.btn_run.clicked.connect(self.run_report)
        self.btn_export.clicked.connect(self.export_csv)

        self._cached_rows: List[Tuple] = []

    def run_report(self):
        sd = self.start_date.date()
        ed = self.end_date.date()
        start_local = datetime(sd.year(), sd.month(), sd.day(), 0, 0, 0).astimezone()
        end_local = datetime(ed.year(), ed.month(), ed.day(), 23, 59, 59).astimezone()

        rows = self.db.fetch_sessions_overlapping(start_local, end_local)
        now_utc = datetime.now(timezone.utc)

        display_rows = []
        for r in rows:
            sid = r["id"]
            task_name = r["task"]
            project_name = r["project_name"] or ""
            # Show Task column as "[Project] — Task"
            task_col = f"[{project_name}] — {task_name}" if project_name else task_name
            st_utc = parse_iso_utc(r["start_time"])
            et_utc = parse_iso_utc(r["end_time"])
            elapsed_seconds = int(((et_utc or now_utc) - st_utc).total_seconds())
            elapsed_seconds = max(0, elapsed_seconds)
            display_rows.append((
                str(sid),
                task_col,
                to_local_str(st_utc),
                to_local_str(et_utc) if et_utc else "",
                fmt_hhmmss(elapsed_seconds)
            ))

        # Render table
        self.table.setRowCount(len(display_rows))
        for row_idx, row_vals in enumerate(display_rows):
            for col_idx, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()
        self._cached_rows = display_rows

    def export_csv(self):
        if not self._cached_rows:
            self.run_report()
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "report.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Session ID", "Task", "Start Time", "End Time", "Elapsed"])
                for row in self._cached_rows:
                    writer.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save CSV:\n{e}")
            return
        QMessageBox.information(self, "Export Successful", f"Saved to:\n{path}")

# =========================
# Main Window
# =========================
class MainWindow(QMainWindow):
    def __init__(self, db: DB):
        super().__init__()
        self.db = db
        self.setWindowTitle("Log'd (Lite)")
        self.setMinimumSize(900, 600)
        self.setWindowIcon(QIcon.fromTheme("clock"))

        # Tabs: Tasks, Projects, Reports
        self.tabs = QTabWidget(self)
        self.tasks_tab = TasksTab(self.db, self)
        self.projects_tab = ProjectsTab(self.db, self)
        self.reports_tab = ReportsTab(self.db)
        self.tabs.addTab(self.tasks_tab, "Tasks")
        self.tabs.addTab(self.projects_tab, "Projects")
        self.tabs.addTab(self.reports_tab, "Reports")
        self.setCentralWidget(self.tabs)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.request_quit)
        file_menu.addAction(act_quit)

        view_menu = menubar.addMenu("&View")
        self.act_always_on_top = QAction("Always on top", self, checkable=True)
        self.act_always_on_top.setChecked(ALWAYS_ON_TOP)
        self.act_always_on_top.toggled.connect(self.toggle_always_on_top)
        view_menu.addAction(self.act_always_on_top)

        # Timer tick
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start()

        # Tray icon
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.windowIcon() if not self.windowIcon().isNull() else QIcon("app_icon.png"))
        self.tray.activated.connect(self.on_tray_activated)
        self.tray_menu = QMenu()
        self.tray_toggle = QAction("Start", self)
        self.tray_toggle.triggered.connect(self.tasks_tab.on_toggle_clicked)
        act_tray_quit = QAction("Quit", self)
        act_tray_quit.triggered.connect(self.request_quit)
        self.tray_menu.addAction(self.tray_toggle)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(act_tray_quit)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.show()

        # Keep project dropdown up-to-date when switching to Tasks tab
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # initial tick to sync UI/tray
        self.on_tick()

    def on_tab_changed(self, idx: int):
        # If Tasks tab is shown, refresh projects dropdown in case projects changed
        if self.tabs.widget(idx) is self.tasks_tab:
            self.tasks_tab.refresh_projects_dropdown()
            self.tasks_tab.refresh_task_list()

        if self.tabs.widget(idx) is self.projects_tab:
            self.projects_tab.refresh_projects()

    def on_tick(self):
        # Forward tick to tasks tab
        self.tasks_tab.timer_tick()
        # Update tray label
        global active_session_id
        self.tray_toggle.setText("Pause/Switch Task" if active_session_id is not None else "Start")

    def toggle_always_on_top(self, checked: bool):
        global ALWAYS_ON_TOP
        ALWAYS_ON_TOP = checked
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    def request_quit(self):
        global active_session_id
        if active_session_id is not None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Active Session Running")
            msg.setText("An active session is running. End it before exiting?")
            end_exit = msg.addButton("End & Exit", QMessageBox.ButtonRole.AcceptRole)
            exit_wo = msg.addButton("Exit Without Ending", QMessageBox.ButtonRole.DestructiveRole)
            cancel = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(cancel)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked is end_exit:
                try:
                    self.db.end_session(active_session_id)
                except Exception:
                    pass
                QApplication.quit()
                return
            elif clicked is exit_wo:
                QApplication.quit()
                return
            else:
                return
        QApplication.quit()

    def closeEvent(self, event):
        event.ignore()
        self.request_quit()

# =========================
# Entry point
# =========================
def main():
    app = QApplication(sys.argv)
    db = DB(DB_PATH)
    win = MainWindow(db)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()