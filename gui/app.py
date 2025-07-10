import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QComboBox, QGroupBox, QFileDialog, QMessageBox,
    QProgressBar, QTextEdit, QRadioButton, QFrame, QHeaderView, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from gui.engine import MLEngine, calcular_especificidade_media

STYLE_SHEET = """
    /* Main window and general styling */
    QMainWindow, QWidget { 
        background-color: #f5f7fa; 
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Group boxes - more modern look */
    QGroupBox {
        font-size: 14px; 
        font-weight: bold; 
        border: 1px solid #d1d9e6;
        border-radius: 10px; 
        margin-top: 12px; 
        background-color: #ffffff;
        padding-top: 15px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin; 
        subcontrol-position: top left;
        left: 10px;
        padding: 0 8px; 
        background-color: #ffffff;
        color: #4a5568;
    }
    
    /* Buttons - gradient and better hover effects */
    QPushButton {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #6b46c1, stop:1 #553c9a);
        color: white; 
        font-size: 13px;
        font-weight: 600; 
        padding: 10px 15px; 
        border-radius: 6px; 
        border: none;
        min-width: 120px;
        margin: 3px 0;
    }
    
    QPushButton:hover {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #805ad5, stop:1 #6b46c1);
    }
    
    QPushButton:pressed {
        background-color: #44337a;
    }
    
    QPushButton:disabled {
        background-color: #e2e8f0;
        color: #a0aec0;
    }
    
    /* Special buttons */
    #RunExperimentButton {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #38a169, stop:1 #2f855a);
    }
    
    #RunExperimentButton:hover {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #48bb78, stop:1 #38a169);
    }
    
    /* Tab widgets - more modern styling */
    QTabWidget::pane { 
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin-top: -1px;
        background: white;
    }
    
    QTabBar::tab {
        background: #edf2f7; 
        border: 1px solid #e2e8f0; 
        padding: 8px 16px;
        border-top-left-radius: 6px; 
        border-top-right-radius: 6px; 
        font-weight: 600;
        color: #4a5568;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected { 
        background: white; 
        border-bottom-color: white;
        color: #2d3748;
    }
    
    QTabBar::tab:hover {
        background: #e2e8f0;
    }
    
    /* Progress bar - more modern look */
    QProgressBar {
        border-radius: 6px; 
        text-align: center;
        height: 12px;
        border: 1px solid #e2e8f0;
    }
    
    QProgressBar::chunk { 
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                      stop:0 #4299e1, stop:1 #3182ce);
        border-radius: 6px; 
    }
    
    /* Tables - better styling */
    QTableWidget {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        background-color: white;
        alternate-background-color: #f8fafc;
        gridline-color: #e2e8f0;
    }
    
    QTableWidget QHeaderView::section {
        background-color: #edf2f7;
        padding: 6px;
        border: none;
        font-weight: 600;
        color: #4a5568;
    }
    
    QTableWidget QTableCornerButton::section {
        background-color: #edf2f7;
        border: none;
    }
    
    /* Text edits - better readability */
    QTextEdit {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px;
        background-color: white;
        color: #2d3748;
    }
    
    /* Radio buttons - better styling */
    QRadioButton {
        spacing: 8px;
        color: #4a5568;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border-radius: 8px;
        border: 2px solid #a0aec0;
    }
    
    QRadioButton::indicator:checked {
        background-color: #6b46c1;
        border-color: #6b46c1;
    }
    
    /* Combo boxes - modern look */
    QComboBox {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px;
        background-color: white;
        min-width: 120px;
    }
    
    QComboBox:hover {
        border-color: #cbd5e0;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #e2e8f0;
    }
    
    /* Custom widget IDs */
    #navPanel {
        background-color: white;
        border-radius: 10px;
    }
    
    #contentArea {
        background-color: white;
        border-radius: 10px;
    }
    
    #centralWidget {
        background-color: #f5f7fa;
    }
    
    #resultsTabs {
        border: none;
    }
    
    .MetricValueLabel {
        font-weight: bold;
        font-size: 13px;
        color: #2d3748;
    }
    
    .MetricHeaderLabel {
        font-weight: bold;
        color: #4a5568;
    }
"""

class Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Análise de Vinhos")
        self.setWindowIcon(QIcon('icons/wine_icon.png'))
        self.resize(1600, 950)
        self.setStyleSheet(STYLE_SHEET)
        
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        self.create_nav_panel()
        self.nav_panel.setGraphicsEffect(self.create_shadow_effect())
        main_layout.addWidget(self.nav_panel)
        
        content_area = QFrame()
        content_area.setObjectName("contentArea")
        content_area.setFrameShape(QFrame.StyledPanel)
        content_area.setGraphicsEffect(self.create_shadow_effect())
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.results_tabs = QTabWidget()
        self.results_tabs.setObjectName("resultsTabs")
        content_layout.addWidget(self.results_tabs)
        
        main_layout.addWidget(content_area, stretch=1)
        
        self.create_summary_results_tab()
        self.create_comparison_plot_tab()
        self.create_detailed_analysis_tab()
        self.create_cv_tab()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("QStatusBar { background-color: #edf2f7; border-top: 1px solid #e2e8f0; color: #4a5568; }")
        self.status_bar.addPermanentWidget(self.progress_bar, 1)
        
        self.engine = MLEngine()
        self.thread = None
        self.worker = None
        self.is_centered = False
        self.experiment_results = None
        
        self.update_ui_state()

    def create_shadow_effect(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 30))
        return shadow

    def create_nav_panel(self):
        self.nav_panel = QFrame()
        self.nav_panel.setObjectName("navPanel")
        self.nav_panel.setFrameShape(QFrame.StyledPanel)
        self.nav_panel.setFixedWidth(300)
        
        nav_layout = QVBoxLayout(self.nav_panel)
        nav_layout.setContentsMargins(15, 15, 15, 15)
        nav_layout.setSpacing(10)
        
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 15)
        logo = QLabel()
        logo.setPixmap(QIcon('icons/wine_icon.png').pixmap(40, 40))
        header_layout.addWidget(logo)
        title = QLabel("Análise de Vinhos")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d3748;")
        header_layout.addWidget(title, 1)
        nav_layout.addWidget(header)
        
        data_group = QGroupBox("1. Dados")
        data_layout = QVBoxLayout(data_group)
        self.btn_load_data = QPushButton("Carregar Arquivos de Vinho")
        self.btn_load_data.setIcon(QIcon('icons/upload.png'))
        self.btn_load_data.clicked.connect(self.load_data)
        data_layout.addWidget(self.btn_load_data)
        nav_layout.addWidget(data_group)
        
        self.analysis_group = QGroupBox("2. Análise Exploratória")
        analysis_layout = QVBoxLayout(self.analysis_group)
        self.btn_run_eda = QPushButton("Análise Descritiva e de Nulos")
        self.btn_run_eda.setIcon(QIcon('icons/stats.png'))
        self.btn_run_eda.clicked.connect(self.run_eda)
        self.btn_corr_matrix = QPushButton("Matriz de Correlação")
        self.btn_corr_matrix.setIcon(QIcon('icons/correlation.png'))
        self.btn_corr_matrix.clicked.connect(self.run_correlation)
        self.btn_dist_plots = QPushButton("Gráficos de Distribuição")
        self.btn_dist_plots.setIcon(QIcon('icons/distribution.png'))
        self.btn_dist_plots.clicked.connect(self.run_distributions)
        analysis_layout.addWidget(self.btn_run_eda)
        analysis_layout.addWidget(self.btn_corr_matrix)
        analysis_layout.addWidget(self.btn_dist_plots)
        nav_layout.addWidget(self.analysis_group)

        self.modeling_group = QGroupBox("3. Modelagem Preditiva")
        modeling_layout = QVBoxLayout(self.modeling_group)
        pipeline_frame = QFrame()
        pipeline_layout = QVBoxLayout(pipeline_frame)
        pipeline_layout.setContentsMargins(5, 5, 5, 5)
        self.radio_pipeline_std = QRadioButton("Pipeline Padrão")
        self.radio_pipeline_std.setChecked(True)
        self.radio_pipeline_pca = QRadioButton("Pipeline com PCA")
        pipeline_layout.addWidget(self.radio_pipeline_std)
        pipeline_layout.addWidget(self.radio_pipeline_pca)
        modeling_layout.addWidget(pipeline_frame)
        self.btn_run_experiment = QPushButton("Executar Experimento")
        self.btn_run_experiment.setObjectName("RunExperimentButton")
        self.btn_run_experiment.setIcon(QIcon('icons/experiment.png'))
        self.btn_run_experiment.clicked.connect(self.run_experiment)
        modeling_layout.addWidget(self.btn_run_experiment)
        nav_layout.addWidget(self.modeling_group)

        self.validation_group = QGroupBox("4. Validação do Modelo")
        validation_layout = QVBoxLayout(self.validation_group)
        self.btn_run_cv = QPushButton("Executar Validação Cruzada")
        self.btn_run_cv.setIcon(QIcon('icons/validation.png'))
        self.btn_run_cv.clicked.connect(self.run_cv)
        validation_layout.addWidget(self.btn_run_cv)
        nav_layout.addWidget(self.validation_group)
        
        nav_layout.addStretch()

    def create_summary_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        eda_group = QGroupBox("Análise Exploratória Descritiva")
        eda_layout = QVBoxLayout(eda_group)
        self.eda_results_text = QTextEdit()
        self.eda_results_text.setReadOnly(True)
        self.eda_results_text.setStyleSheet("QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; }")
        eda_layout.addWidget(self.eda_results_text)
        layout.addWidget(eda_group)
        
        summary_group = QGroupBox("Resumo das Métricas do Experimento")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_table = QTableWidget()
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setSortingEnabled(True)
        summary_layout.addWidget(self.summary_table)
        layout.addWidget(summary_group)
        
        self.results_tabs.addTab(widget, QIcon('icons/summary.png'), "Resumos e Métricas")

    def create_comparison_plot_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        btn_save = QPushButton("Salvar Gráfico")
        btn_save.setIcon(QIcon('icons/save.png'))
        btn_save.clicked.connect(self.save_comparison_plot)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)
        
        self.comp_figure = Figure(figsize=(10, 6), dpi=100)
        self.comp_canvas = FigureCanvas(self.comp_figure)
        self.comp_canvas.setStyleSheet("background-color: white; border-radius: 6px;")
        layout.addWidget(self.comp_canvas)
        self.results_tabs.addTab(widget, QIcon('icons/chart.png'), "Gráfico Comparativo")

    def create_detailed_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        controls = QFrame()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(QLabel("Selecione o Modelo:"))
        self.model_combo_details = QComboBox()
        self.model_combo_details.setFixedWidth(200)
        self.model_combo_details.currentTextChanged.connect(self.display_detailed_analysis)
        controls_layout.addWidget(self.model_combo_details)
        controls_layout.addStretch()
        btn_save_report = QPushButton("Salvar Relatório")
        btn_save_report.setIcon(QIcon('icons/save.png'))
        btn_save_report.clicked.connect(self.save_detailed_report)
        controls_layout.addWidget(btn_save_report)
        layout.addWidget(controls)

        metrics_panel_group = QGroupBox("Métricas Principais")
        metrics_layout = QGridLayout(metrics_panel_group)
        metrics_layout.setSpacing(10)
        headers = ["Métrica", "Geral", "Brancos", "Tintos"]
        for col, header_text in enumerate(headers):
            label = QLabel(header_text)
            label.setStyleSheet("font-weight: bold; color: #4a5568;")
            metrics_layout.addWidget(label, 0, col, alignment=Qt.AlignCenter)
        
        self.detail_metric_labels = {}
        metric_names = ["Acurácia", "Precisão", "Recall", "F1-Score", "Especific."]
        
        for row, name in enumerate(metric_names, 1):
            metrics_layout.addWidget(QLabel(name), row, 0)
            for col, cat in enumerate(["geral", "brancos", "tintos"], 1):
                value_label = QLabel("-")
                value_label.setObjectName("MetricValueLabel")
                value_label.setAlignment(Qt.AlignCenter)
                metrics_layout.addWidget(value_label, row, col)
                self.detail_metric_labels[f"{name}_{cat}"] = value_label
        
        layout.addWidget(metrics_panel_group)
        
        report_group = QGroupBox("Relatório de Classificação Completo")
        report_layout = QVBoxLayout(report_group)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; }")
        report_layout.addWidget(self.details_text)
        layout.addWidget(report_group)
        
        matrices_group = QGroupBox("Matrizes de Confusão")
        matrices_layout = QVBoxLayout(matrices_group)
        self.details_figure = Figure(figsize=(12, 4))
        self.details_canvas = FigureCanvas(self.details_figure)
        self.details_canvas.setStyleSheet("background-color: white; border-radius: 6px;")
        matrices_layout.addWidget(self.details_canvas)
        layout.addWidget(matrices_group)
        
        self.results_tabs.addTab(widget, QIcon('icons/analysis.png'), "Análise Detalhada")
        
    def create_cv_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 10)
        btn_save = QPushButton("Salvar Gráfico")
        btn_save.setIcon(QIcon('icons/save.png'))
        btn_save.clicked.connect(self.save_cv_results)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)
        
        cv_summary_group = QGroupBox("Resumo Numérico da Validação Cruzada")
        self.cv_summary_layout = QGridLayout(cv_summary_group)
        self.cv_summary_layout.setSpacing(10)
        
        headers = ["Modelo", "Acurácia Média", "Desvio Padrão (±)"]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold; color: #4a5568;")
            self.cv_summary_layout.addWidget(label, 0, col)
        
        layout.addWidget(cv_summary_group)
        
        cv_plot_group = QGroupBox("Visualização Gráfica (Boxplot)")
        cv_plot_layout = QVBoxLayout(cv_plot_group)
        self.cv_figure = Figure(figsize=(10, 6), dpi=100)
        self.cv_canvas = FigureCanvas(self.cv_figure)
        self.cv_canvas.setStyleSheet("background-color: white; border-radius: 6px;")
        cv_plot_layout.addWidget(self.cv_canvas)
        layout.addWidget(cv_plot_group)
        
        self.results_tabs.addTab(widget, QIcon('icons/validation.png'), "Validação Cruzada")

    def center(self):
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showEvent(self, event):
        super().showEvent(event)
        if not self.is_centered:
            self.center()
            self.is_centered = True

    def update_ui_state(self):
        data_loaded = self.engine.df is not None
        results_loaded = self.experiment_results is not None
        
        self.analysis_group.setEnabled(data_loaded)
        self.modeling_group.setEnabled(data_loaded)
        self.validation_group.setEnabled(data_loaded)
        
        self.results_tabs.setEnabled(True)
        self.results_tabs.widget(0).setEnabled(data_loaded)
        self.results_tabs.widget(1).setEnabled(results_loaded)
        self.results_tabs.widget(2).setEnabled(results_loaded)
        self.results_tabs.widget(3).setEnabled(data_loaded)
        
    def start_task(self, function, on_finished, *args, **kwargs):
        self.set_ui_busy(True)
        self.worker = Worker(function, *args, **kwargs)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.error.connect(self.on_task_error)
        self.worker.finished.connect(on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(lambda: self.set_ui_busy(False))
        self.thread.finished.connect(self.thread.deleteLater)
        self.progress_bar.setRange(0, 0)
        self.thread.start()

    def set_ui_busy(self, busy):
        self.nav_panel.setEnabled(not busy)
        self.progress_bar.setVisible(busy)

    def on_task_error(self, traceback_str):
        self.progress_bar.setRange(0, 1)
        QMessageBox.critical(self, "Erro na Execução", traceback_str)
        self.update_ui_state()
        
    def load_data(self):
        self.status_bar.showMessage("Carregando dados...")
        self.start_task(self.engine.load_data, self.on_data_loaded)

    def on_data_loaded(self, df):
        self.progress_bar.setRange(0, 1)
        if df is not None:
            self.status_bar.showMessage(f"Dados carregados: {len(df)} linhas.")
            QMessageBox.information(self, "Sucesso", "Dados carregados. Análise exploratória habilitada.")
            self.run_eda()
        else:
            QMessageBox.critical(self, "Erro", "Não foi possível carregar os arquivos da pasta 'docs/'.")
        self.update_ui_state()
        
    def run_eda(self):
        self.results_tabs.setCurrentWidget(self.results_tabs.widget(0))
        self.status_bar.showMessage("Executando análise exploratória...")
        self.start_task(self.engine.get_exploratory_analysis, self.on_eda_finished)

    def on_eda_finished(self, results):
        self.progress_bar.setRange(0, 1)
        self.eda_results_text.setText(results)
        self.status_bar.showMessage("Análise exploratória concluída.")

    def run_correlation(self):
        self.results_tabs.setCurrentWidget(self.comp_canvas.parentWidget())
        self.status_bar.showMessage("Gerando matriz de correlação...")
        self.comp_figure.clear()
        try:
            corr = self.engine.df.select_dtypes(include=np.number).corr()
            ax = self.comp_figure.add_subplot(111)
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Matriz de Correlação')
            self.comp_figure.tight_layout()
            self.comp_canvas.draw()
            self.status_bar.showMessage("Matriz de correlação gerada.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar matriz de correlação: {str(e)}")
            self.status_bar.showMessage("Erro ao gerar matriz de correlação")

    def run_distributions(self):
        QMessageBox.information(self, "Aviso", "Os gráficos de distribuição serão mostrados em janelas separadas. Feche uma para ver a próxima.")
        self.visualizar_distribuicoes(self.engine.df)
    
    def run_experiment(self):
        use_pca = self.radio_pipeline_pca.isChecked()
        pipeline_name = "com PCA" if use_pca else "Padrão"
        self.status_bar.showMessage(f"Executando experimento com pipeline {pipeline_name}...")
        self.start_task(self.engine.run_experiment, self.on_experiment_finished, use_pca)

    def on_experiment_finished(self, results):
        self.progress_bar.setRange(0, 1)
        if results is None:
            QMessageBox.critical(self, "Erro", "Ocorreu um erro durante a execução do experimento. Verifique o console.")
            return

        self.experiment_results = results
        self.status_bar.showMessage("Experimento concluído.")
        
        self.model_combo_details.blockSignals(True)
        self.model_combo_details.clear()
        self.model_combo_details.addItems(results.keys())
        self.model_combo_details.blockSignals(False)
        
        self.update_summary_table()
        self.display_comparison_plot()
        self.display_detailed_analysis()
        self.results_tabs.setCurrentIndex(0)
        self.update_ui_state()
    
    def update_summary_table(self):
        self.summary_table.clear()
        if not self.experiment_results: 
            return
        
        models = list(self.experiment_results.keys())
        self.summary_table.setRowCount(len(models))
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels([
            "Modelo", "Acurácia", "Precisão (Pond)", "F1-Score (Pond)", "Especificidade (Macro)"
        ])
        
        for i, model_name in enumerate(models):
            res = self.experiment_results[model_name]
            report = res['report_dict']
            self.summary_table.setItem(i, 0, QTableWidgetItem(model_name))
            self.summary_table.setItem(i, 1, QTableWidgetItem(f"{report['accuracy']:.4f}"))
            self.summary_table.setItem(i, 2, QTableWidgetItem(f"{report['weighted avg']['precision']:.4f}"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(f"{report['weighted avg']['f1-score']:.4f}"))
            self.summary_table.setItem(i, 4, QTableWidgetItem(f"{res.get('specificity', 0.0):.4f}"))

        self.summary_table.resizeColumnsToContents()
        self.summary_table.resizeRowsToContents()
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def display_detailed_analysis(self):
        model_name = self.model_combo_details.currentText()
        if not model_name or not self.experiment_results: 
            return
            
        res = self.experiment_results[model_name]
        
        report_geral = res['report_dict']
        spec_geral = res.get('specificity', 0.0)
        self.detail_metric_labels["Acurácia_geral"].setText(f"{report_geral['accuracy']:.3f}")
        self.detail_metric_labels["Precisão_geral"].setText(f"{report_geral['weighted avg']['precision']:.3f}")
        self.detail_metric_labels["Recall_geral"].setText(f"{report_geral['weighted avg']['recall']:.3f}")
        self.detail_metric_labels["F1-Score_geral"].setText(f"{report_geral['weighted avg']['f1-score']:.3f}")
        self.detail_metric_labels["Especific._geral"].setText(f"{spec_geral:.3f}")

        has_details = 'X_test_df' in res and res['X_test_df'] is not None
        if has_details:
            mask_white = res['X_test_df']['color_white'] == 1
            y_true_white = res['y_true'][mask_white]
            y_pred_white = res['y_pred'][mask_white]
            report_white = classification_report(y_true_white, y_pred_white, zero_division=0, output_dict=True)
            spec_white = calcular_especificidade_media(y_true_white, y_pred_white, res['classes'])
            self.detail_metric_labels["Acurácia_brancos"].setText(f"{report_white['accuracy']:.3f}")
            self.detail_metric_labels["Precisão_brancos"].setText(f"{report_white['weighted avg']['precision']:.3f}")
            self.detail_metric_labels["Recall_brancos"].setText(f"{report_white['weighted avg']['recall']:.3f}")
            self.detail_metric_labels["F1-Score_brancos"].setText(f"{report_white['weighted avg']['f1-score']:.3f}")
            self.detail_metric_labels["Especific._brancos"].setText(f"{spec_white:.3f}")

            mask_red = res['X_test_df']['color_white'] == 0
            y_true_red = res['y_true'][mask_red]
            y_pred_red = res['y_pred'][mask_red]
            report_red = classification_report(y_true_red, y_pred_red, zero_division=0, output_dict=True)
            spec_red = calcular_especificidade_media(y_true_red, y_pred_red, res['classes'])
            self.detail_metric_labels["Acurácia_tintos"].setText(f"{report_red['accuracy']:.3f}")
            self.detail_metric_labels["Precisão_tintos"].setText(f"{report_red['weighted avg']['precision']:.3f}")
            self.detail_metric_labels["Recall_tintos"].setText(f"{report_red['weighted avg']['recall']:.3f}")
            self.detail_metric_labels["F1-Score_tintos"].setText(f"{report_red['weighted avg']['f1-score']:.3f}")
            self.detail_metric_labels["Especific._tintos"].setText(f"{spec_red:.3f}")
        else:
            for cat in ["brancos", "tintos"]:
                for name in ["Acurácia", "Precisão", "Recall", "F1-Score", "Especific."]:
                    self.detail_metric_labels[f"{name}_{cat}"].setText("N/A")

        report_text = classification_report(res['y_true'], res['y_pred'], zero_division=0)
        self.details_text.setText(report_text)
        
        self.details_figure.clear()
        num_plots = 3 if has_details else 1
        ax1 = self.details_figure.add_subplot(1, num_plots, 1)
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='viridis', xticklabels=res['classes'], yticklabels=res['classes'], ax=ax1)
        ax1.set_title("Geral")
        if num_plots == 3:
            ax2 = self.details_figure.add_subplot(1, 3, 2)
            cm_white = confusion_matrix(res['y_true'][mask_white], res['y_pred'][mask_white], labels=res['classes'])
            sns.heatmap(cm_white, annot=True, fmt='d', cmap='Blues', xticklabels=res['classes'], yticklabels=res['classes'], ax=ax2)
            ax2.set_title("Brancos")
            ax3 = self.details_figure.add_subplot(1, 3, 3)
            cm_red = confusion_matrix(res['y_true'][mask_red], res['y_pred'][mask_red], labels=res['classes'])
            sns.heatmap(cm_red, annot=True, fmt='d', cmap='Reds', xticklabels=res['classes'], yticklabels=res['classes'], ax=ax3)
            ax3.set_title("Tintos")
        self.details_figure.suptitle(f"Matrizes de Confusão - {model_name}", fontsize=14)
        self.details_figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.details_canvas.draw()
        
    def display_comparison_plot(self):
        self.results_tabs.setCurrentWidget(self.comp_canvas.parentWidget())
        self.comp_figure.clear()
        self.plotar_comparacao_detalhada(self.experiment_results, self.comp_figure)
        self.comp_canvas.draw()

    def run_cv(self):
        self.results_tabs.setCurrentWidget(self.results_tabs.widget(3))
        self.status_bar.showMessage("Executando validação cruzada...")
        self.start_task(self.engine.run_cross_validation, self.on_cv_finished)

    def on_cv_finished(self, cv_results):
        self.progress_bar.setRange(0, 1)
        if cv_results is None:
            QMessageBox.warning(self, "Aviso", "Os dados de treino não foram gerados. Execute um experimento primeiro.")
            return

        layout = self.cv_summary_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
        headers = ["Modelo", "Acurácia Média", "Desvio Padrão (±)"]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold; color: #4a5568;")
            self.cv_summary_layout.addWidget(label, 0, col)

        row = 1
        for name, scores in cv_results.items():
            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            self.cv_summary_layout.addWidget(QLabel(name), row, 0)
            self.cv_summary_layout.addWidget(QLabel(f"{mean_acc:.4f}"), row, 1)
            self.cv_summary_layout.addWidget(QLabel(f"{std_acc:.4f}"), row, 2)
            row += 1
            
        self.cv_figure.clear()
        ax = self.cv_figure.add_subplot(111)
        sns.boxplot(data=pd.DataFrame(cv_results), ax=ax)
        ax.set_title('Comparação de Desempenho na Validação Cruzada (5 Folds)')
        ax.set_ylabel('Acurácia')
        self.cv_figure.tight_layout()
        self.cv_canvas.draw()
        self.status_bar.showMessage("Validação cruzada concluída.")

    def save_comparison_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Gráfico Comparativo", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)")
        if file_path:
            try:
                self.comp_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Sucesso", f"Gráfico salvo em:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar gráfico:\n{str(e)}")

    def save_detailed_report(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Relatório Detalhado", "", "Arquivos de Texto (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.details_text.toPlainText())
                QMessageBox.information(self, "Sucesso", f"Relatório salvo em:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar relatório:\n{str(e)}")

    def save_cv_results(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Gráfico da Validação Cruzada", "", "PNG (*.png);;JPEG (*.jpg);;PDF (*.pdf)")
        if file_path:
            try:
                self.cv_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Sucesso", f"Gráfico salvo em:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar resultados:\n{str(e)}")

    def plotar_comparacao_detalhada(self, resultados, figure):
        if not resultados: 
            return
        modelos = list(resultados.keys())
        acuracias_geral = [r['report_dict']['accuracy'] for r in resultados.values()]
        tem_detalhes = all(('X_test_df' in r and r['X_test_df'] is not None) for r in resultados.values())
        figure.clear()
        ax = figure.add_subplot(111)
        if not tem_detalhes:
            bars = ax.bar(modelos, acuracias_geral, color=plt.cm.viridis(np.linspace(0, 1, len(modelos))))
            ax.set_title('Comparação de Acurácia Geral (PCA)')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')
        else:
            acuracias_brancos = [accuracy_score(r['y_true'][r['X_test_df']['color_white']==1], r['y_pred'][r['X_test_df']['color_white']==1]) for r in resultados.values()]
            acuracias_tintos = [accuracy_score(r['y_true'][r['X_test_df']['color_white']==0], r['y_pred'][r['X_test_df']['color_white']==0]) for r in resultados.values()]
            x = np.arange(len(modelos))
            width = 0.25
            rects1 = ax.bar(x - width, acuracias_geral, width, label='Geral', color='#581845')
            rects2 = ax.bar(x, acuracias_brancos, width, label='Brancos', color='#2E8B57')
            rects3 = ax.bar(x + width, acuracias_tintos, width, label='Tintos', color='#FF5733')
            ax.set_title('Comparação de Desempenho por Tipo de Vinho')
            ax.set_xticks(x)
            ax.set_xticklabels(modelos, rotation=15, ha="right")
            ax.legend()
            for rects in [rects1, rects2, rects3]:
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        ax.set_ylabel('Acurácia')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        figure.tight_layout()

    def visualizar_distribuicoes(self, df):
        colunas_features = df.select_dtypes(include=['float64', 'int64']).drop('quality', axis=1).columns
        for coluna in colunas_features:
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x=coluna, hue='color', kde=True, palette={'red': '#C70039', 'white': '#DAF7A6'})
            plt.title(f'Distribuição de {coluna}', fontsize=14)
            plt.subplot(1, 2, 2)
            sns.boxplot(data=df, y=coluna, x='color', palette={'red': '#C70039', 'white': '#DAF7A6'})
            plt.title(f'Box Plot de {coluna} (Outliers)', fontsize=14)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())