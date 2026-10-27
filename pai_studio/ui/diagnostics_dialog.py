"""User-controlled presentation/export for local diagnostics results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
        QTableWidget, QTableWidgetItem, QVBoxLayout,
    )
    QT_AVAILABLE = True
except ImportError:  # Headless CI keeps the diagnostics service independently testable.
    QT_AVAILABLE = False


if QT_AVAILABLE:
    class DiagnosticsDialog(QDialog):
        def __init__(self, context: dict[str, Any], parent=None):
            super().__init__(parent)
            from ..workers import DiagnosticsWorker

            self.setWindowTitle("P.AI · Local Diagnostics")
            self.resize(850, 520)
            self._results = []
            self._worker = DiagnosticsWorker(context, self)
            layout = QVBoxLayout(self)
            self.summary = QLabel("Running read-only local checks. No model will be downloaded or inference started.")
            self.summary.setWordWrap(True)
            layout.addWidget(self.summary)
            self.table = QTableWidget(0, 4, self)
            self.table.setHorizontalHeaderLabels(["Check", "Status", "Result", "Suggested action"])
            self.table.setWordWrap(True)
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.horizontalHeader().setSectionResizeMode(0, self.table.horizontalHeader().ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(2, self.table.horizontalHeader().ResizeMode.Stretch)
            layout.addWidget(self.table, 1)
            buttons = QHBoxLayout()
            self.export_button = QPushButton("Export redacted report…", self)
            self.export_button.setEnabled(False)
            self.export_button.clicked.connect(self._export_report)
            self.cancel_button = QPushButton("Cancel checks", self)
            self.cancel_button.clicked.connect(self._cancel_checks)
            self.close_button = QPushButton("Close", self)
            self.close_button.clicked.connect(self.close)
            buttons.addWidget(self.export_button)
            buttons.addStretch(1)
            buttons.addWidget(self.cancel_button)
            buttons.addWidget(self.close_button)
            layout.addLayout(buttons)
            self._worker.completed.connect(self._show_results)
            self._worker.failed.connect(self._show_failure)
            self._worker.finished.connect(self._worker_finished)
            self._worker.start()

        def _show_results(self, results):
            self._results = list(results or [])
            self.table.setRowCount(len(self._results))
            for row, item in enumerate(self._results):
                values = (item.check_id, item.status.replace("_", " ").title(), item.summary, item.remediation)
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(str(value))
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    cell.setToolTip(str(value))
                    self.table.setItem(row, column, cell)
            failures = sum(item.status == "fail" for item in self._results)
            warnings = sum(item.status == "warning" for item in self._results)
            self.summary.setText(f"Completed {len(self._results)} read-only checks · {failures} failures · {warnings} warnings. Unsupported telemetry is shown as Not applicable, not as a runtime failure.")
            self.export_button.setEnabled(True)

        def _show_failure(self, message):
            self.summary.setText(str(message))

        def _worker_finished(self):
            self.cancel_button.setEnabled(False)

        def _cancel_checks(self):
            if self._worker.isRunning():
                self._worker.request_cancel()
                self.summary.setText("Cancellation requested; current bounded health check will return before remaining checks are skipped.")
                self.cancel_button.setEnabled(False)

        def _export_report(self):
            if not self._results:
                return
            path, _filter = QFileDialog.getSaveFileName(self, "Save redacted diagnostics report", str(Path.home() / "pai-diagnostics.json"), "JSON report (*.json)")
            if not path:
                return
            from ..diagnostics import save_diagnostics_report
            try:
                saved = save_diagnostics_report(self._results, path)
                QMessageBox.information(self, "Report saved", f"Redacted report saved locally:\n{saved}")
            except OSError as exc:
                QMessageBox.warning(self, "Could not save report", str(exc))

        def closeEvent(self, event):
            if self._worker.isRunning():
                self._worker.request_cancel()
                self._worker.wait(1000)
            super().closeEvent(event)
else:
    class DiagnosticsDialog:  # pragma: no cover - only available with PySide6
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PySide6 is required for the diagnostics dialog.")
