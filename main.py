import sys
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QFileDialog,
    QComboBox, QLabel, QLineEdit, QTextEdit, QTabWidget,
    QMessageBox, QFrame, QHeaderView, QGridLayout, QMenuBar, QMenu,
    QCheckBox, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sympy as sp


# --- LOGIKA ZA MAJORIZACIJU (SLIKE 1 I 2) ---
def majorize_value(val):
    """Glavna funkcija za majorizaciju prema tvojim pravilima."""
    if val == 0 or pd.isna(val):
        return val
    try:
        val = float(val)
        order = math.floor(math.log10(abs(val)))
        first_digit = int(abs(val) / (10 ** order))
        # Pravilo: Ako je prva cifra 1 -> 2 značajne, inače 1 značajna
        sig_figs = 2 if first_digit == 1 else 1
        factor = 10 ** (order - sig_figs + 1)
        # Majorizacija: Uvek na veću vrednost
        majorized = math.ceil(val / factor) * factor
        # Čišćenje decimala za lepši ispis
        decimals = max(0, -(order - sig_figs + 1))
        return round(majorized, decimals)
    except:
        return val


def calculate_relative_error(delta_x, x_mean):
    """Pravilo za relativnu grešku: Zaokružuje se na 2 značajne cifre."""
    if x_mean == 0:
        return 0
    rel_err = abs(delta_x / x_mean)
    order = math.floor(math.log10(abs(rel_err)))
    factor = 10 ** (order - 1)
    return math.ceil(rel_err / factor) * factor


def format_scientific(val):
    """Format number in scientific notation if needed."""
    if pd.isna(val) or not isinstance(val, (int, float, np.number)) or val == 0:
        return str(val)
    try:
        val = float(val)
        # Koristi scientific notaciju samo ako je broj vrlo mali (<0.001) ili veliki (>=1000)
        if abs(val) >= 1e-3 and abs(val) < 1e3:
            # Obični format sa odgovarajućim brojem decimala
            if abs(val) < 1:
                return f"{val:.4f}"
            else:
                return f"{val:.1f}"
        else:
            # Scientific notacija
            exponent = math.floor(math.log10(abs(val)))
            mantissa = val / (10 ** exponent)
            if abs(mantissa - 1.0) < 1e-10:
                return f"10^{exponent}"
            else:
                return f"{mantissa:.1f}*10^{exponent}"
    except:
        return str(val)


# --- STILOVI ---
DARK_STYLE = """
    QMainWindow { background-color: #1e1e1e; }
    QTabWidget::pane { border: 1px solid #333; background: #252526; }
    QTabBar::tab { background: #2d2d2d; color: #b1b1b1; padding: 10px 20px; border: 1px solid #333; }
    QTabBar::tab:selected { background: #3e3e42; color: white; border-bottom: 2px solid #007acc; }
    QPushButton { background-color: #0e639c; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
    QPushButton#majorBtn { background-color: #6a1b9a; border: 1px solid #9c27b0; }
    QPushButton#majorBtn:hover { background-color: #8e24aa; }
    QTableWidget { background-color: #252526; color: white; gridline-color: #333; }
    QHeaderView::section { background-color: #333; color: white; }
    QLabel { color: #cccccc; }
    QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: 'Consolas'; }
"""

LIGHT_STYLE = """
    QMainWindow { background-color: #f8f9fa; }
    QTabWidget::pane { border: 1px solid #dee2e6; background: #ffffff; }
    QTabBar::tab { background: #ffffff; color: #495057; padding: 10px 20px; border: 1px solid #dee2e6; }
    QTabBar::tab:selected { background: #ffffff; color: #007acc; border-bottom: 2px solid #007acc; }
    QPushButton { background-color: #007acc; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
    QPushButton#majorBtn { background-color: #6a1b9a; border: 1px solid #9c27b0; }
    QPushButton#majorBtn:hover { background-color: #8e24aa; }
    QTableWidget { background-color: #ffffff; color: #212529; gridline-color: #dee2e6; }
    QHeaderView::section { background-color: #e9ecef; color: #212529; }
    QLabel { color: #495057; }
    QTextEdit { background-color: #ffffff; color: #212529; font-family: 'Consolas'; border: 1px solid #dee2e6; }
"""


class MplCanvas(FigureCanvas):
    def __init__(self):
        plt.style.use('default')
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PhysicsLabApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fizika Aplikacija")
        self.resize(1200, 800)
        self.df = pd.DataFrame()
        self.current_theme = "light"  # Track current theme
        self.pdf_selected_columns = set()  # Columns selected for PDF
        self.pdf_merged_pairs = []  # List of (main_col, error_col) tuples
        self.setStyleSheet(LIGHT_STYLE)
        self.init_ui()

    def init_ui(self):
        # Ikone za temu na desnoj strani menu bara
        self.menu_bar = self.menuBar()

        # Dodajemo razmak sa leve strane
        self.menu_bar.addAction("")  # Prazna akcija za razmak

        # Akcije za temu sa ikonama na DESNOJ strani
        self.light_theme_action = QAction("☀️ Svetla", self)
        self.light_theme_action.triggered.connect(lambda: self.switch_theme("light"))
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.setChecked(True)
        self.menu_bar.addAction(self.light_theme_action)

        self.dark_theme_action = QAction("🌙 Tamna", self)
        self.dark_theme_action.triggered.connect(lambda: self.switch_theme("dark"))
        self.dark_theme_action.setCheckable(True)
        self.menu_bar.addAction(self.dark_theme_action)

        # Kreiraj main container sa tabovima i footer-om
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Dodaj footer
        self.footer = QLabel("Autor: Nikola Vojinović 3/7 | Predmet: Fizika | Profesorka: Jelena Tasić")
        self.footer.setStyleSheet("QLabel { color: #333; font-size: 13px; font-weight: bold; padding: 10px; text-align: center; border-top: 2px solid #007acc; background-color: #f5f5f5; }")
        self.footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.footer)

        self.setCentralWidget(main_container)

        # Tabovi
        self.tab_data = QWidget()
        self.tab_analysis = QWidget()
        self.tab_errors = QWidget()
        self.tab_calculator = QWidget()

        self.tabs.addTab(self.tab_data, " Tabela ")
        self.tabs.addTab(self.tab_analysis, " Grafik")
        self.tabs.addTab(self.tab_errors, " Greške i Izveštaji ")
        self.tabs.addTab(self.tab_calculator, " Kalkulator ")

        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_errors_tab()
        self.setup_calculator_tab()

        # Inicijalno ažuriranje combo box-ova
        self.update_combos()

    # ----- TAB ZA TABELU -----
    def setup_data_tab(self):
        layout = QVBoxLayout(self.tab_data)

        top_bar = QHBoxLayout()
        self.import_btn = QPushButton("Učitaj Excel/CSV")
        self.import_btn.clicked.connect(self.import_file)

        # DUGME ZA MAJORIZACIJU SVIH KOLONA
        self.major_btn = QPushButton("MAJORIZUJ SVE KOLONE")
        self.major_btn.setObjectName("majorBtn")
        self.major_btn.clicked.connect(self.apply_majorization_to_table)

        top_bar.addWidget(self.import_btn)
        top_bar.addWidget(self.major_btn)
        layout.addLayout(top_bar)

        self.table = QTableWidget()
        layout.addWidget(self.table)

    # ----- TAB ZA ANALIZU -----
    def setup_analysis_tab(self):
        layout = QHBoxLayout(self.tab_analysis)
        controls = QVBoxLayout()

        self.cb_x = QComboBox()
        self.cb_y = QComboBox()
        controls.addWidget(QLabel("X osa (Vrednost):"))
        controls.addWidget(self.cb_x)
        controls.addWidget(QLabel("Y osa (Merenje):"))
        controls.addWidget(self.cb_y)

        # Dugme za crtanje grafa
        self.draw_graph_analysis_btn = QPushButton("NACRTAJ GRAFIK")
        self.draw_graph_analysis_btn.clicked.connect(self.draw_graph_analysis)
        controls.addWidget(self.draw_graph_analysis_btn)

        btn_grid = QGridLayout()
        models = [("Linearna", "linear"), ("Kvadratna", "quadratic"),
                  ("Eksponencijalna", "exponential"), ("Logaritamska", "logarithmic")]

        for i, (name, mode) in enumerate(models):
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, m=mode: self.run_regression_analysis(m))
            btn_grid.addWidget(btn, i // 2, i % 2)

        controls.addLayout(btn_grid)

        # Dugme za određivanje koeficijenta pravca
        self.slope_btn = QPushButton("ODREDI KOEFICIJENT PRAVCA")
        self.slope_btn.clicked.connect(self.calculate_slope)
        controls.addWidget(self.slope_btn)

        self.fit_output = QTextEdit()
        controls.addWidget(QLabel("Rezultati analize i Koeficijenti:"))
        controls.addWidget(self.fit_output)

        layout.addLayout(controls, 1)
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, 3)

    # ----- TAB ZA GREŠKE -----
    def setup_errors_tab(self):
        layout = QVBoxLayout(self.tab_errors)

        # Gornji deo - Majorizacija
        majorization_group = QVBoxLayout()

        # Izbor kolone za majorizaciju
        self.cb_column_errors = QComboBox()
        majorization_group.addWidget(QLabel("Izaberite kolonu za majorizaciju:"))
        majorization_group.addWidget(self.cb_column_errors)

        # Dugme za majorizaciju izabrane kolone
        self.major_column_btn = QPushButton("MAJORIZUJ IZABRANU KOLONU")
        self.major_column_btn.setObjectName("majorBtn")
        self.major_column_btn.clicked.connect(self.majorize_selected_column)
        majorization_group.addWidget(self.major_column_btn)

        # Zaokruživanje
        rounding_layout = QHBoxLayout()
        self.cb_column_round = QComboBox()
        self.decimals_input = QLineEdit()
        self.decimals_input.setPlaceholderText("Broj decimala")
        self.decimals_input.setMaximumWidth(100)
        self.round_column_btn = QPushButton("ZAOKRUŽI KOLONU")
        self.round_column_btn.clicked.connect(self.round_selected_column)
        rounding_layout.addWidget(QLabel("Kolona:"))
        rounding_layout.addWidget(self.cb_column_round)
        rounding_layout.addWidget(QLabel("Decimala:"))
        rounding_layout.addWidget(self.decimals_input)
        rounding_layout.addWidget(self.round_column_btn)
        majorization_group.addLayout(rounding_layout)

        layout.addLayout(majorization_group)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Donji deo - PDF Izveštaj
        pdf_group = QVBoxLayout()
        pdf_group.addWidget(QLabel("Generisanje PDF izveštaja:"))

        # Dugme za generisanje PDF-a
        self.generate_pdf_btn = QPushButton("GENERIŠI PDF IZVEŠTAJ")
        self.generate_pdf_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; padding: 10px; border-radius: 4px; font-weight: bold; }")
        self.generate_pdf_btn.clicked.connect(self.generate_pdf_report)
        pdf_group.addWidget(self.generate_pdf_btn)

        # Info label
        self.pdf_info = QLabel("PDF izveštaj će sadržati originalne podatke i majorizovane vrednosti.")
        self.pdf_info.setStyleSheet("QLabel { color: #cccccc; font-style: italic; }")
        pdf_group.addWidget(self.pdf_info)

        layout.addLayout(pdf_group)

        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)

        # PDF Column Settings
        column_settings_group = QVBoxLayout()
        column_settings_group.addWidget(QLabel("Podešavanje kolona za PDF:"))

        # Column checkboxes
        self.column_checkboxes = {}
        checkboxes_layout = QHBoxLayout()
        for col in self.df.columns if not self.df.empty else []:
            cb = QCheckBox(col)
            cb.setChecked(True)  # Default to selected
            self.column_checkboxes[col] = cb
            checkboxes_layout.addWidget(cb)
        column_settings_group.addLayout(checkboxes_layout)

        # Merge pairs
        merge_layout = QHBoxLayout()
        self.cb_main_col = QComboBox()
        self.cb_error_col = QComboBox()
        if not self.df.empty:
            self.cb_main_col.addItems(list(self.df.columns))
            self.cb_error_col.addItems(list(self.df.columns))
        merge_layout.addWidget(QLabel("Glavna kolona:"))
        merge_layout.addWidget(self.cb_main_col)
        merge_layout.addWidget(QLabel("Kolona greške:"))
        merge_layout.addWidget(self.cb_error_col)
        self.add_merge_btn = QPushButton("Dodaj par za spajanje")
        self.add_merge_btn.clicked.connect(self.add_merge_pair)
        merge_layout.addWidget(self.add_merge_btn)
        column_settings_group.addLayout(merge_layout)

        # List of merged pairs
        self.merged_pairs_list = QListWidget()
        column_settings_group.addWidget(QLabel("Definisan parovi za spajanje:"))
        column_settings_group.addWidget(self.merged_pairs_list)

        layout.addLayout(column_settings_group)
        layout.addStretch()  # Dodaje prostor na dnu

    # ----- TAB ZA KALKULATOR -----
    def setup_calculator_tab(self):
        layout = QVBoxLayout(self.tab_calculator)

        # Input field
        self.calc_input = QLineEdit()
        self.calc_input.setPlaceholderText("Unesite izraz, npr. sqrt(2) + sin(pi/4)")
        layout.addWidget(QLabel("Izraz:"))
        layout.addWidget(self.calc_input)

        # Calculate button
        self.calc_btn = QPushButton("IZRAČUNAJ")
        self.calc_btn.clicked.connect(self.calculate_expression)
        layout.addWidget(self.calc_btn)

        # Results display
        self.calc_output = QTextEdit()
        self.calc_output.setReadOnly(True)
        layout.addWidget(QLabel("Rezultati i istorija:"))
        layout.addWidget(self.calc_output)

        # Button grid for calculator
        button_grid = QGridLayout()

        # Row 1: Numbers 7-9, /
        button_grid.addWidget(self.create_calc_button("7"), 0, 0)
        button_grid.addWidget(self.create_calc_button("8"), 0, 1)
        button_grid.addWidget(self.create_calc_button("9"), 0, 2)
        button_grid.addWidget(self.create_calc_button("/"), 0, 3)

        # Row 2: Numbers 4-6, *
        button_grid.addWidget(self.create_calc_button("4"), 1, 0)
        button_grid.addWidget(self.create_calc_button("5"), 1, 1)
        button_grid.addWidget(self.create_calc_button("6"), 1, 2)
        button_grid.addWidget(self.create_calc_button("*"), 1, 3)

        # Row 3: Numbers 1-3, -
        button_grid.addWidget(self.create_calc_button("1"), 2, 0)
        button_grid.addWidget(self.create_calc_button("2"), 2, 1)
        button_grid.addWidget(self.create_calc_button("3"), 2, 2)
        button_grid.addWidget(self.create_calc_button("-"), 2, 3)

        # Row 4: 0, ., +, =
        button_grid.addWidget(self.create_calc_button("0"), 3, 0)
        button_grid.addWidget(self.create_calc_button("."), 3, 1)
        button_grid.addWidget(self.create_calc_button("+"), 3, 2)
        button_grid.addWidget(self.create_calc_button("="), 3, 3)

        # Row 5: Functions
        button_grid.addWidget(self.create_calc_button("sqrt("), 4, 0)
        button_grid.addWidget(self.create_calc_button("sin("), 4, 1)
        button_grid.addWidget(self.create_calc_button("cos("), 4, 2)
        button_grid.addWidget(self.create_calc_button("tan("), 4, 3)

        # Row 6: More functions
        button_grid.addWidget(self.create_calc_button("log("), 5, 0)
        button_grid.addWidget(self.create_calc_button("ln("), 5, 1)
        button_grid.addWidget(self.create_calc_button("pi"), 5, 2)
        button_grid.addWidget(self.create_calc_button("e"), 5, 3)

        # Row 7: Parentheses and power
        button_grid.addWidget(self.create_calc_button("("), 6, 0)
        button_grid.addWidget(self.create_calc_button(")"), 6, 1)
        button_grid.addWidget(self.create_calc_button("^"), 6, 2)
        button_grid.addWidget(self.create_calc_button("C"), 6, 3)

        layout.addLayout(button_grid)

    def create_calc_button(self, text):
        btn = QPushButton(text)
        btn.clicked.connect(lambda: self.on_calc_button_click(text))
        return btn

    def on_calc_button_click(self, text):
        if text == "C":
            self.calc_input.clear()
        elif text == "=":
            self.calculate_expression()
        else:
            current = self.calc_input.text()
            self.calc_input.setText(current + text)

    def calculate_expression(self):
        expr = self.calc_input.text().strip()
        if not expr:
            return

        try:
            # Replace DataFrame functions like mean(I)
            expr = self.replace_df_functions(expr)

            # Parse and evaluate with sympy
            result = sp.sympify(expr).evalf()

            # Format result
            if result.is_real and result.is_finite:
                result_str = format_scientific(float(result))
            else:
                result_str = str(result)

            # Add to history
            history = self.calc_output.toPlainText()
            new_entry = f"> {expr}\n= {result_str}\n\n"
            self.calc_output.setText(new_entry + history)

        except Exception as e:
            error_msg = f"> {expr}\nGreška: {str(e)}\n\n"
            history = self.calc_output.toPlainText()
            self.calc_output.setText(error_msg + history)

    def replace_df_functions(self, expr):
        """Replace functions like mean(col) with actual values from DataFrame."""
        if self.df.empty:
            return expr

        # Supported functions
        functions = ['mean', 'min', 'max']

        for func in functions:
            import re
            pattern = rf'{func}\(\s*(\w+)\s*\)'
            matches = re.findall(pattern, expr)

            for col in matches:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    if func == 'mean':
                        val = self.df[col].mean()
                    elif func == 'min':
                        val = self.df[col].min()
                    elif func == 'max':
                        val = self.df[col].max()

                    expr = expr.replace(f'{func}({col})', str(val))

        return expr

    def add_merge_pair(self):
        main_col = self.cb_main_col.currentText()
        error_col = self.cb_error_col.currentText()

        if not main_col or not error_col:
            QMessageBox.warning(self, "Greška", "Izaberite obe kolone za spajanje.")
            return

        if main_col == error_col:
            QMessageBox.warning(self, "Greška", "Glavna i kolona greške ne mogu biti iste.")
            return

        # Check if already exists
        for pair in self.pdf_merged_pairs:
            if pair[0] == main_col or pair[1] == error_col:
                QMessageBox.warning(self, "Greška", "Jedna od kolona je već u paru.")
                return

        self.pdf_merged_pairs.append((main_col, error_col))
        self.update_merged_pairs_list()
        QMessageBox.information(self, "Uspeh", f"Par '{main_col}' ± '{error_col}' je dodat.")

    def update_merged_pairs_list(self):
        self.merged_pairs_list.clear()
        for main_col, error_col in self.pdf_merged_pairs:
            item = QListWidgetItem(f"{main_col} ± {error_col}")
            self.merged_pairs_list.addItem(item)

    def refresh_pdf_settings(self):
        """Refresh PDF column checkboxes and dropdowns."""
        if self.df.empty:
            return

        # Update checkboxes
        # Find the layout containing checkboxes
        # Since it's complex, let's recreate the checkboxes layout
        # For now, just update the dropdowns
        cols = list(self.df.columns)
        self.cb_main_col.clear()
        self.cb_main_col.addItems(cols)
        self.cb_error_col.clear()
        self.cb_error_col.addItems(cols)

        # Note: Checkboxes are created in setup_errors_tab, but since data changes, we need to update them
        # For simplicity, assume checkboxes are updated when data is imported

    # ----- FUNKCIJE -----
    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Otvori", "", "Excel/CSV (*.xlsx *.csv)")
        if path:
            try:
                self.df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
                self.refresh_table()
                self.update_combos()
            except Exception as e:
                QMessageBox.critical(self, "Greška", f"Neuspešno učitavanje fajla:\n{str(e)}")

    def refresh_table(self):
        if self.df.empty: return
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)
        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                val = self.df.iloc[i, j]
                self.table.setItem(i, j, QTableWidgetItem(format_scientific(val)))

    def update_combos(self):
        if self.df.empty: return
        cols = list(self.df.columns)
        self.cb_x.clear();
        self.cb_x.addItems(cols)
        self.cb_y.clear();
        self.cb_y.addItems(cols)
        self.cb_column_errors.clear();
        self.cb_column_errors.addItems(cols)
        self.cb_column_round.clear();
        self.cb_column_round.addItems(cols)

        # Update PDF column checkboxes
        # Clear existing checkboxes
        for cb in self.column_checkboxes.values():
            cb.setParent(None)
        self.column_checkboxes.clear()

        # Create new checkboxes
        checkboxes_layout = self.findChild(QHBoxLayout)  # Need to find the layout
        # Actually, since setup_errors_tab is called once, we need to recreate the layout or update it
        # For simplicity, let's recreate the checkboxes in the existing layout
        # But since it's complex, let's add a method to refresh PDF settings

        self.refresh_pdf_settings()

    def apply_majorization_to_table(self):
        if self.df.empty: return
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].apply(majorize_value)
        self.refresh_table()
        QMessageBox.information(self, "Majorizacija", "Sve numeričke kolone su ispravljene (zaokruženo na višu cifru).")

    # ----- FUNKCIJE ZA TAB ANALIZE -----
    def draw_graph_analysis(self):
        if self.df.empty: return
        try:
            x = pd.to_numeric(self.df[self.cb_x.currentText()], errors='coerce').dropna().values
            y = pd.to_numeric(self.df[self.cb_y.currentText()], errors='coerce').dropna().values

            if len(x) == 0 or len(y) == 0:
                QMessageBox.warning(self, "Greška", "Izabrane kolone ne sadrže numeričke podatke.")
                return

            # Plotting
            self.canvas.axes.clear()
            self.canvas.axes.scatter(x, y, label="Podaci", color="#007acc")
            self.canvas.axes.set_xlabel(self.cb_x.currentText())
            self.canvas.axes.set_ylabel(self.cb_y.currentText())
            self.canvas.axes.legend()
            # Add grid visible on both themes
            grid_color = '#cccccc' if self.current_theme == 'light' else '#666666'
            self.canvas.axes.grid(True, alpha=0.3, color=grid_color)
            self.canvas.draw()

            res_text = "Grafik je nacrtan.\nIzaberite tip regresije da biste nacrtali pravu.\n"
            self.fit_output.setText(res_text)

        except Exception as e:
            QMessageBox.critical(self, "Greška", str(e))

    def run_regression_analysis(self, mode):
        if self.df.empty: return
        try:
            x = pd.to_numeric(self.df[self.cb_x.currentText()], errors='coerce').dropna().values
            y = pd.to_numeric(self.df[self.cb_y.currentText()], errors='coerce').dropna().values

            if len(x) == 0 or len(y) == 0:
                QMessageBox.warning(self, "Greška", "Izabrane kolone ne sadrže numeričke podatke.")
                return

            res_text = ""

            # Fitting modela
            if mode == "linear":
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)

                res_text += f"Linearna regresija: y = {slope:.4f}x + {intercept:.4f}\n"
                res_text += f"Koeficijent pravca : {slope:.4f}\n"

                # Crtanje linije
                self.canvas.axes.plot(x, y_pred, 'r-', label=f'Linearna: k={slope:.4f}', linewidth=2)

            elif mode == "quadratic":
                coeffs = np.polyfit(x, y, 2)
                y_pred = np.polyval(coeffs, x)
                r2 = r2_score(y, y_pred)

                res_text += f"Kvadratna regresija: y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}\n"

                x_sorted = np.sort(x)
                y_pred_sorted = np.polyval(coeffs, x_sorted)
                self.canvas.axes.plot(x_sorted, y_pred_sorted, 'g-', label='Kvadratna', linewidth=2)

            elif mode == "exponential":
                def exp_func(x, a, b):
                    return a * np.exp(b * x)
                try:
                    popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.01))
                    a, b = popt
                    y_pred = exp_func(x, a, b)
                    r2 = r2_score(y, y_pred)

                    res_text += f"Eksponencijalna regresija: y = {a:.4f} * exp({b:.4f}x)\n"


                    x_sorted = np.sort(x)
                    y_pred_sorted = exp_func(x_sorted, a, b)
                    self.canvas.axes.plot(x_sorted, y_pred_sorted, 'y-', label='Eksponencijalna', linewidth=2)
                except:
                    res_text += "Eksponencijalna regresija nije uspela.\n"

            elif mode == "logarithmic":
                x_log = np.log(x[x > 0])
                y_log = y[x > 0]
                if len(x_log) > 0:
                    coeffs = np.polyfit(x_log, y_log, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]
                    y_pred = np.polyval(coeffs, x_log)
                    r2 = r2_score(y_log, y_pred)

                    res_text += f"Logaritamska regresija: y = {slope:.4f}ln(x) + {intercept:.4f}\n"
                    res_text += f"Koeficijent pravca : {slope:.4f}\n"

                    x_sorted = np.sort(x[x > 0])
                    x_log_sorted = np.log(x_sorted)
                    y_pred_sorted = np.polyval(coeffs, x_log_sorted)
                    self.canvas.axes.plot(x_sorted, y_pred_sorted, 'm-', label='Logaritamska', linewidth=2)
                else:
                    res_text += "Logaritamska regresija nije moguća (negativne X vrednosti).\n"

            self.canvas.axes.legend()
            # Add grid visible on both themes
            grid_color = '#cccccc' if self.current_theme == 'light' else '#666666'
            self.canvas.axes.grid(True, alpha=0.3, color=grid_color)
            self.canvas.draw()
            self.fit_output.setText(res_text)

        except Exception as e:
            QMessageBox.critical(self, "Greška", str(e))

    def calculate_slope(self):
        if self.df.empty: return
        try:
            x = pd.to_numeric(self.df[self.cb_x.currentText()], errors='coerce').dropna().values
            y = pd.to_numeric(self.df[self.cb_y.currentText()], errors='coerce').dropna().values

            if len(x) == 0 or len(y) == 0:
                QMessageBox.warning(self, "Greška", "Izabrane kolone ne sadrže numeričke podatke.")
                return

            # Računanje koeficijenta pravca
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            res_text = f"ODREĐIVANJE KOEFICIJENTA PRAVCA\n"
            res_text += "=" * 40 + "\n"
            res_text += f"Jednačina pravca: y = {slope:.6f}x + {intercept:.6f}\n"
            res_text += f"Koeficijent pravca : {slope:.6f}\n"

            # Dodatne statistike
            res_text += f"Broj tačaka: {len(x)}\n"
            res_text += f"Opseg X: [{np.min(x):.4f}, {np.max(x):.4f}]\n"
            res_text += f"Opseg Y: [{np.min(y):.4f}, {np.max(y):.4f}]\n"

            self.fit_output.setText(res_text)

        except Exception as e:
            QMessageBox.critical(self, "Greška", str(e))

    def switch_theme(self, theme):
        """Promena teme aplikacije"""
        if theme == self.current_theme:
            return

        self.current_theme = theme

        # Update menu checkmarks
        self.light_theme_action.setChecked(theme == "light")
        self.dark_theme_action.setChecked(theme == "dark")

        # Apply new stylesheet
        if theme == "light":
            self.setStyleSheet(LIGHT_STYLE)
            # Update matplotlib style for light theme
            plt.style.use('default')
            self.canvas.fig.set_facecolor('white')
            self.canvas.axes.set_facecolor('white')
            # Update footer style for light theme
            self.footer.setStyleSheet("QLabel { color: #333; font-size: 13px; font-weight: bold; padding: 10px; text-align: center; border-top: 2px solid #007acc; background-color: #f5f5f5; }")
        else:  # dark theme
            self.setStyleSheet(DARK_STYLE)
            # Update matplotlib style for dark theme
            plt.style.use('dark_background')
            self.canvas.fig.set_facecolor('#252526')
            self.canvas.axes.set_facecolor('#252526')
            # Update footer style for dark theme
            self.footer.setStyleSheet("QLabel { color: #e0e0e0; font-size: 13px; font-weight: bold; padding: 10px; text-align: center; border-top: 2px solid #007acc; background-color: #1e1e1e; }")

        # Redraw canvas if it has content
        self.canvas.draw()

        # Update PDF info label color based on theme
        if theme == "dark":
            self.pdf_info.setStyleSheet("QLabel { color: #cccccc; font-style: italic; }")
        else:
            self.pdf_info.setStyleSheet("QLabel { color: #6c757d; font-style: italic; }")

    # ----- FUNKCIJE ZA TAB GREŠKE -----
    def majorize_selected_column(self):
        if self.df.empty: return
        selected_col = self.cb_column_errors.currentText()
        if not selected_col:
            QMessageBox.warning(self, "Greška", "Izaberite kolonu za majorizaciju.")
            return
        if pd.api.types.is_numeric_dtype(self.df[selected_col]):
            self.df[selected_col] = self.df[selected_col].apply(majorize_value)
            self.refresh_table()
            QMessageBox.information(self, "Majorizacija", f"Kolona '{selected_col}' je majorizovana.")
        else:
            QMessageBox.warning(self, "Greška", "Izabrana kolona nije numerička.")

    def round_selected_column(self):
        if self.df.empty: return
        selected_col = self.cb_column_round.currentText()
        decimals_text = self.decimals_input.text().strip()
        if not selected_col:
            QMessageBox.warning(self, "Greška", "Izaberite kolonu za zaokruživanje.")
            return
        if not decimals_text:
            QMessageBox.warning(self, "Greška", "Unesite broj decimala.")
            return
        try:
            decimals = int(decimals_text)
            if decimals < 0:
                QMessageBox.warning(self, "Greška", "Broj decimala mora biti pozitivan.")
                return
        except ValueError:
            QMessageBox.warning(self, "Greška", "Broj decimala mora biti ceo broj.")
            return
        if pd.api.types.is_numeric_dtype(self.df[selected_col]):
            self.df[selected_col] = self.df[selected_col].round(decimals)
            self.refresh_table()
            QMessageBox.information(self, "Zaokruživanje", f"Kolona '{selected_col}' je zaokružena na {decimals} decimala.")
        else:
            QMessageBox.warning(self, "Greška", "Izabrana kolona nije numerička.")

    def generate_pdf_report(self):
        if self.df.empty:
            QMessageBox.warning(self, "Greška", "Nema podataka za generisanje izveštaja.")
            return

        # Odabir lokacije za čuvanje PDF-a
        file_path, _ = QFileDialog.getSaveFileName(self, "Sačuvaj PDF izveštaj", "", "PDF fajlovi (*.pdf)")

        if not file_path:
            return

        try:
            # Kreiranje PDF dokumenta
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Naslov
            title = Paragraph("FIZIČKI LABORATORIJSKI IZVEŠTAJ", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))

            # Informacije o izveštaju
            info_text = f"Broj kolona: {len(self.df.columns)}<br/>Broj redova: {len(self.df)}<br/>Datum generisanja: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}"
            info = Paragraph(info_text, styles['Normal'])
            story.append(info)
            story.append(Spacer(1, 12))

            # Dodavanje grafa u PDF (ako postoji)
            try:
                # Kreiranje novog grafa sa svetlom pozadinom za PDF
                fig_pdf = Figure(figsize=(6, 4), dpi=100, facecolor='white')
                ax_pdf = fig_pdf.add_subplot(111)

                # Kopiranje podataka iz trenutnog grafa
                if hasattr(self.canvas, 'axes') and len(self.canvas.axes.lines) > 0:
                    # Kopiranje scatter plot-a
                    if hasattr(self.canvas.axes, 'collections') and len(self.canvas.axes.collections) > 0:
                        scatter = self.canvas.axes.collections[0]
                        ax_pdf.scatter(scatter.get_offsets()[:, 0], scatter.get_offsets()[:, 1],
                                     color='blue', label='Podaci', alpha=0.7)

                    # Kopiranje linija regresije
                    for line in self.canvas.axes.lines:
                        ax_pdf.plot(line.get_xdata(), line.get_ydata(),
                                  color=line.get_color(), linewidth=line.get_linewidth(),
                                  label=line.get_label())

                    ax_pdf.set_xlabel(self.canvas.axes.get_xlabel())
                    ax_pdf.set_ylabel(self.canvas.axes.get_ylabel())
                    ax_pdf.set_title("Grafik sa regresijom")
                    ax_pdf.legend()
                    ax_pdf.grid(True, alpha=0.3)

                    # Čuvanje grafa u buffer-u
                    buf = io.BytesIO()
                    fig_pdf.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)

                    # Dodavanje grafa u PDF
                    img = Image(buf)
                    img.drawHeight = 3 * inch
                    img.drawWidth = 5 * inch
                    story.append(img)
                    story.append(Spacer(1, 12))

                    plt.close(fig_pdf)  # Oslobađanje memorije
            except Exception as e:
                # Ako ne može da doda grafik, nastavlja bez njega
                print(f"Greška pri dodavanju grafa u PDF: {e}")
                pass

            # Determine selected columns
            selected_columns = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]

            # Create merged column names
            merged_columns = {}
            for main_col, error_col in self.pdf_merged_pairs:
                merged_name = f"{main_col} ± {error_col}"
                merged_columns[merged_name] = (main_col, error_col)

            # Columns for PDF: selected + merged
            pdf_columns = selected_columns + list(merged_columns.keys())

            # Tabela sa podacima
            # Priprema podataka za tabelu
            table_data = [pdf_columns]  # Zaglavlje

            # Dodavanje redova podataka (ograničeno na prvih 100 redova zbog veličine)
            max_rows = min(100, len(self.df))
            for i in range(max_rows):
                row = []
                for col in pdf_columns:
                    if col in merged_columns:
                        main_col, error_col = merged_columns[col]
                        val_main = self.df.iloc[i][main_col]
                        val_error = self.df.iloc[i][error_col]
                        if pd.isna(val_main) or pd.isna(val_error):
                            row.append("")
                        else:
                            row.append(f"{format_scientific(val_main)} ± {format_scientific(val_error)}")
                    else:
                        val = self.df.iloc[i][col]
                        if pd.isna(val):
                            row.append("")
                        else:
                            row.append(format_scientific(val))
                table_data.append(row)

            # Kreiranje tabele
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(table)

            if len(self.df) > 100:
                note = Paragraph(f"<i>Napomena: Prikazano je prvih 100 od {len(self.df)} redova.</i>", styles['Italic'])
                story.append(Spacer(1, 12))
                story.append(note)

            # Statistike po kolonama
            story.append(Spacer(1, 20))
            stats_title = Paragraph("STATISTIKE PO KOLONAMA", styles['Heading2'])
            story.append(stats_title)
            story.append(Spacer(1, 12))

            for col in pdf_columns:
                if col in merged_columns:
                    main_col, error_col = merged_columns[col]
                    if pd.api.types.is_numeric_dtype(self.df[main_col]) and pd.api.types.is_numeric_dtype(self.df[error_col]):
                        mean_val = self.df[main_col].mean()
                        max_error = self.df[error_col].max()  # Use max of error column
                        majorized_error = majorize_value(max_error)

                        col_stats = f"<b>{col}:</b><br/>"
                        col_stats += f"Prosek: {format_scientific(mean_val)}<br/>"
                        col_stats += f"Zapis: {format_scientific(mean_val)} ± {format_scientific(majorized_error)}"

                        story.append(Paragraph(col_stats, styles['Normal']))
                        story.append(Spacer(1, 6))
                elif col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    mean_val = self.df[col].mean()
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()

                    # Računanje najveće apsolutne greške i majorizacija
                    values = self.df[col].dropna()
                    if len(values) > 0:
                        abs_errors = np.abs(values - mean_val)
                        max_abs_error = np.max(abs_errors)
                        majorized_error = majorize_value(max_abs_error)
                    else:
                        majorized_error = 0

                    col_stats = f"<b>{col}:</b><br/>"
                    col_stats += f"Prosek: {format_scientific(mean_val)}<br/>"
                    col_stats += f"Minimum: {format_scientific(min_val)}<br/>"
                    col_stats += f"Maksimum: {format_scientific(max_val)}<br/>"
                    col_stats += f"Zapis: {format_scientific(mean_val)} ± {format_scientific(majorized_error)}"

                    story.append(Paragraph(col_stats, styles['Normal']))
                    story.append(Spacer(1, 6))

            # Generisanje PDF-a
            doc.build(story)

            QMessageBox.information(self, "Uspeh", f"PDF izveštaj je sačuvan u:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Greška", f"Greška pri generisanju PDF-a:\n{str(e)}")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PhysicsLabApp()
    win.show()
    sys.exit(app.exec())
