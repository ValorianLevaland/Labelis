from __future__ import annotations

from typing import Optional

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from .df_model import DataFrameModel


class ResultsDialog(QtWidgets.QDialog):
    def __init__(self, df: pd.DataFrame, summary_text: str = "", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Labelis - Results")
        self.resize(1000, 700)

        layout = QtWidgets.QVBoxLayout(self)

        if summary_text:
            lbl = QtWidgets.QLabel(summary_text)
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        self.model = DataFrameModel(df)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn, 0)
