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
    QMessageBox, QFrame, QHeaderView, QGridLayout, QMenuBar, QMenu
)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabovi
        self.tab_data = QWidget()
        self.tab_analysis = QWidget()
        self.tab_errors = QWidget()

        self.tabs.addTab(self.tab_data, " Tabela ")
        self.tabs.addTab(self.tab_analysis, " Grafik")
        self.tabs.addTab(self.tab_errors, " Greške i Izveštaji ")

        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_errors_tab()

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
        layout.addStretch()  # Dodaje prostor na dnu

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
                self.table.setItem(i, j, QTableWidgetItem(str(val)))

    def update_combos(self):
        if self.df.empty: return
        cols = list(self.df.columns)
        self.cb_x.clear();
        self.cb_x.addItems(cols)
        self.cb_y.clear();
        self.cb_y.addItems(cols)
        self.cb_column_errors.clear();
        self.cb_column_errors.addItems(cols)

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
        else:  # dark theme
            self.setStyleSheet(DARK_STYLE)
            # Update matplotlib style for dark theme
            plt.style.use('dark_background')
            self.canvas.fig.set_facecolor('#252526')
            self.canvas.axes.set_facecolor('#252526')

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

            # Tabela sa podacima
            # Priprema podataka za tabelu
            table_data = [list(self.df.columns)]  # Zaglavlje

            # Dodavanje redova podataka (ograničeno na prvih 100 redova zbog veličine)
            max_rows = min(100, len(self.df))
            for i in range(max_rows):
                row = []
                for col in self.df.columns:
                    val = self.df.iloc[i][col]
                    if pd.isna(val):
                        row.append("")
                    else:
                        row.append(str(val))
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

            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
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
                    col_stats += f"Prosek: {mean_val:.4f}<br/>"
                    col_stats += f"Minimum: {min_val:.4f}<br/>"
                    col_stats += f"Maksimum: {max_val:.4f}<br/>"
                    col_stats += f"Zapis: {mean_val:.4f} ± {majorized_error}"

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
