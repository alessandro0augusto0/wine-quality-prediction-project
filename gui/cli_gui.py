import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from time import time
import warnings
from sklearn.decomposition import PCA
import os
import sys

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class WineQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Qualidade de Vinhos - IA")
        self.set_window_center(1200, 800)
        self.root.configure(bg='#f4f4f4')
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f4f4f4')
        style.configure('TLabel', background='#f4f4f4', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground='#7B1FA2', background='#f4f4f4')
        style.configure('Section.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#512DA8', background='#f4f4f4')
        style.configure('TButton', font=('Segoe UI', 11), padding=6, background='#ede7f6', foreground='#333')
        style.map('TButton', background=[('active', '#d1c4e9')])
        style.configure('Status.TLabel', font=('Segoe UI', 10), background='#ede7f6', foreground='#333')
        style.configure('TScrollbar', background='#ede7f6')

        self.df = None
        self.modelos = None
        self.X_train_b = None
        self.y_train_b = None
        self.X_test_s = None
        self.y_test = None
        self.X_train_orig = None
        self.y_train_orig = None

        self.create_widgets()

    def on_closing(self):
        """Método chamado quando a janela é fechada."""
        if messagebox.askokcancel("Sair", "Deseja realmente sair do programa?"):
            self.root.destroy()
            os._exit(0)

    def set_window_center(self, width, height):
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.control_frame = ttk.Frame(self.main_frame, width=320, relief=tk.RIDGE, borderwidth=2)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 18), pady=0)
        self.control_frame.pack_propagate(False)

        ttk.Label(self.control_frame, text="Análise de Vinhos", style='Title.TLabel').pack(pady=(18, 18))

        ttk.Button(self.control_frame, text="1. Carregar Dados", command=self.carregar_dados).pack(fill=tk.X, pady=(0, 12), padx=18)

        ttk.Label(self.control_frame, text="Análise e Exploração:", style='Section.TLabel').pack(pady=(0, 4), anchor=tk.W, padx=18)
        ttk.Button(self.control_frame, text="2. Análise Geral", command=self.analise_exploratoria).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="3. Visualizar Distribuição", command=self.visualizar_distribuicoes).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="4. Matriz de Correlação", command=self.visualizar_correlacao).pack(fill=tk.X, pady=2, padx=18)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=14, padx=10)

        ttk.Label(self.control_frame, text="Análise de Modelos:", style='Section.TLabel').pack(pady=(0, 4), anchor=tk.W, padx=18)
        ttk.Button(self.control_frame, text="5. Avaliar Baseline", command=lambda: self.avaliar_modelo('Baseline')).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="6. Avaliar K-NN", command=lambda: self.avaliar_modelo('K-NN')).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="7. Avaliar Árvore Decisão", command=lambda: self.avaliar_modelo('Árvore de Decisão')).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="8. Avaliar Rede Neural", command=lambda: self.avaliar_modelo('Rede Neural (MLP)')).pack(fill=tk.X, pady=2, padx=18)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=14, padx=10)

        ttk.Label(self.control_frame, text="Experimentos:", style='Section.TLabel').pack(pady=(0, 4), anchor=tk.W, padx=18)
        ttk.Button(self.control_frame, text="9. Comparar Todos Modelos", command=self.comparar_todos_modelos).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="10. Comparar com PCA", command=self.comparar_com_pca).pack(fill=tk.X, pady=2, padx=18)
        ttk.Button(self.control_frame, text="11. Validação Cruzada", command=self.executar_validacao_cruzada).pack(fill=tk.X, pady=2, padx=18)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill=tk.X, pady=14, padx=10)

        ttk.Button(self.control_frame, text="Sair", command=self.on_closing).pack(fill=tk.X, pady=(10, 0), padx=18)

        self.output_frame = ttk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 0), pady=0)

        self.output_text = scrolledtext.ScrolledText(
            self.output_frame, wrap=tk.WORD, width=100, height=40,
            font=('Consolas', 11), bg='#fff', fg='#222', borderwidth=0, relief=tk.FLAT, padx=10, pady=10
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        self.status_var = tk.StringVar()
        self.status_var.set("Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style='Status.TLabel', relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=4)

    def print_output(self, text, clear=False):
        if clear:
            self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.root.update()

    def carregar_dados(self):
        try:
            self.print_output("Carregando arquivos CSV...", clear=True)
            df_red = pd.read_csv('docs/winequality-red.csv', sep=';')
            df_white = pd.read_csv('docs/winequality-white.csv', sep=';')
            self.print_output("Arquivos CSV carregados com sucesso!")

            df_red['color'] = 'red'
            df_white['color'] = 'white'
            self.df = pd.concat([df_red, df_white], ignore_index=True)

            # Processar dados para modelagem
            self.print_output("\nProcessando dados para modelagem...")
            (self.X_train_b, self.y_train_b, self.X_test_s, self.y_test,
             self.X_train_orig, self.y_train_orig) = self.processar_dados_para_modelagem(self.df.copy(), usar_pca=False)

            # Obter modelos
            self.modelos = self.get_modelos_com_melhores_parametros()

            self.print_output("\nPronto para análise!")
            self.status_var.set("Dados carregados e processados com sucesso")

        except FileNotFoundError:
            messagebox.showerror("Erro", "Arquivos 'winequality-red.csv' e 'winequality-white.csv' não encontrados na pasta 'docs/'")
            self.status_var.set("Erro ao carregar arquivos")

    def analise_exploratoria(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Por favor, carregue os dados primeiro.")
            return

        self.print_output("\n" + "="*60, clear=True)
        self.print_output("--- 1. ANÁLISE EXPLORATÓRIA GERAL ---")
        self.print_output("="*60)
        self.print_output("\n--- INFORMAÇÕES BÁSICAS DO DATASET ---")

        import io
        buffer = io.StringIO()
        self.df.info(verbose=False, buf=buffer)
        self.print_output(buffer.getvalue())

        self.print_output("\n--- VERIFICAÇÃO DE DADOS AUSENTES ---")
        self.print_output(str(self.df.isnull().sum()))
        self.print_output("\n[ANÁLISE]: O resultado acima mostra que não há dados ausentes na base.")

        self.print_output("\n--- MEDIDAS DE LOCALIDADE E ESPALHAMENTO ---")
        self.print_output(str(self.df.describe().transpose()))

        self.status_var.set("Análise exploratória concluída")

    def visualizar_distribuicoes(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Por favor, carregue os dados primeiro.")
            return

        self.print_output("\n--- Distribuição dos Dados e Análise de Outliers ---", clear=True)
        colunas_features = self.df.select_dtypes(include=['float64', 'int64']).drop('quality', axis=1).columns

        for coluna in colunas_features:
            self.print_output(f"\n===== Feature: {coluna} =====")

            self.print_output("\n[Histograma Simplificado]")
            counts, bin_edges = np.histogram(self.df[coluna])
            max_count = counts.max()
            for i in range(len(counts)):
                bar_len = int((counts[i] / max_count) * 30) if max_count > 0 else 0
                bar = '█' * bar_len
                self.print_output(f"{bin_edges[i]:>6.2f} - {bin_edges[i+1]:>6.2f} | {bar} ({counts[i]})")

            self.print_output("\n[Estatísticas Descritivas]")
            self.print_output(str(self.df.groupby('color')[coluna].describe().transpose().round(2)))

            self.print_output("\nPressione Continuar para próxima feature...")
            self.root.wait_variable(tk.IntVar())

        self.status_var.set("Visualização de distribuições concluída")

    def visualizar_correlacao(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Por favor, carregue os dados primeiro.")
            return

        self.print_output("\n--- Matriz de Correlação ---", clear=True)
        corr = self.df.select_dtypes(include=np.number).corr()
        self.print_output(str(corr.round(2)))
        self.print_output("\n[ANÁLISE]: Use a tabela para identificar atributos com correlação forte (>0.7 ou <-0.7) entre si ou com o alvo ('quality').")

        self.status_var.set("Matriz de correlação exibida")

    def processar_dados_para_modelagem(self, df, usar_pca=False, n_components=0.95):
        self.print_output("\n" + "="*60)
        if usar_pca:
            self.print_output("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO COM PCA ---")
        else:
            self.print_output("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO PADRÃO ---")
        self.print_output("="*60)

        num_duplicatas = df.duplicated().sum()
        if num_duplicatas > 0:
            self.print_output(f"\n1. Removendo {num_duplicatas} linhas duplicadas...")
            df.drop_duplicates(inplace=True)
        else:
            self.print_output("\n1. Nenhuma linha duplicada encontrada.")

        self.print_output("2. Criando categorias 'ruim', 'normal', 'bom'...")
        bins = [0, 4, 6, 10]
        labels = ['ruim', 'normal', 'bom']
        df['quality_category'] = pd.cut(df['quality'], bins=bins, labels=labels, right=True)

        self.print_output("3. Convertendo 'color' para formato numérico...")
        df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True, dtype=int)

        self.print_output("4. Criando coluna para estratificação por qualidade E cor...")
        df_encoded['stratify_col'] = df_encoded['quality_category'].astype(str) + '_' + df_encoded['color_white'].astype(str)

        X = df_encoded.drop(columns=['quality', 'quality_category', 'stratify_col'])
        y = df_encoded['quality_category']
        stratify_col = df_encoded['stratify_col']

        self.print_output("5. Separando dados em treino (80%) e teste (20%) com estratificação dupla...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_col)

        self.print_output("6. Normalizando features com StandardScaler...")
        scaler = StandardScaler()
        cols_to_scale = [col for col in X_train.columns if col != 'color_white']
        X_train_scaled = X_train.copy()
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled = X_test.copy()
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        if usar_pca:
            self.print_output(f"7. Aplicando Redução de Dimensionalidade (PCA) para manter {n_components*100}% da variância...")
            pca = PCA(n_components=n_components)
            X_train_final = pca.fit_transform(X_train_scaled)
            X_test_final = pca.transform(X_test_scaled)
            self.print_output(f"   -> PCA selecionou {pca.n_components_} componentes dos {X_train_scaled.shape[1]} originais.")
        else:
            self.print_output("7. Redução de Dimensionalidade (PCA) não aplicada.")
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        self.print_output("8. Balanceando conjunto de treino com SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)

        self.print_output("\nPRÉ-PROCESSAMENTO CONCLUÍDO!")
        return X_train_balanced, y_train_balanced, X_test_final, y_test, X_train, y_train

    def get_modelos_com_melhores_parametros(self):
        baseline = DummyClassifier(strategy="most_frequent")
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')
        dt = DecisionTreeClassifier(max_depth=8, min_samples_split=5, criterion='gini', random_state=42)
        mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=1500, random_state=42)
        return {'Baseline': baseline, 'K-NN': knn, 'Árvore de Decisão': dt, 'Rede Neural (MLP)': mlp}

    def imprimir_matriz_confusao(self, mcm, labels, title):
        self.print_output(f"\n--- {title} ---")
        header = f"{'Verdadeiro\\Previsto':<12}" + " | ".join([f"{str(l):^8}" for l in labels])
        self.print_output(header)
        self.print_output("-" * len(header))
        for i, label in enumerate(labels):
            row_str = f"{str(label):<12}"
            for j in range(len(labels)):
                row_str += f" | {mcm[i, j]:^8}"
            self.print_output(row_str)
        self.print_output("-" * len(header))

    def plotar_comparacao(self, resultados, title):
        self.print_output(f"\n--- {title} ---")
        if not resultados:
            self.print_output("Nenhum resultado para plotar.")
            return
        max_len = max(len(nome) for nome in resultados.keys())
        for nome, res in resultados.items():
            acc = res['accuracy']
            bar_len = int(acc * 40)
            bar = '█' * bar_len
            self.print_output(f"{nome:<{max_len}} | {bar} {acc:.4f}")

    def imprimir_metricas_com_barra(self, label, valor, largura_total=25):
        valor = max(0.0, min(1.0, valor))
        blocos_cheios = int(valor * largura_total)
        barra = '█' * blocos_cheios + ' ' * (largura_total - blocos_cheios)
        label_alinhado = f"{label:<10}"
        self.print_output(f"{label_alinhado}: [{barra}] {valor:.2f}")

    def calcular_especificidade_media(self, y_true, y_pred, labels):
        mcm = confusion_matrix(y_true, y_pred, labels=labels)
        specificities = []
        for i in range(len(labels)):
            tp = mcm[i, i]
            fp = mcm[:, i].sum() - tp
            fn = mcm[i, :].sum() - tp
            tn = mcm.sum() - (tp + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(specificity)
        return np.mean(specificities)

    def avaliar_modelo(self, nome_modelo):
        if self.modelos is None or self.X_train_b is None:
            messagebox.showwarning("Aviso", "Por favor, carregue e processe os dados primeiro.")
            return

        self.print_output(f"\n" + "-"*20 + f" ANÁLISE DETALHADA: {nome_modelo} " + "-"*20, clear=True)

        modelo = self.modelos[nome_modelo]
        start_time = time()
        modelo.fit(self.X_train_b, self.y_train_b)
        end_time = time()
        self.print_output(f"Tempo de treinamento: {end_time - start_time:.2f} segundos")

        y_pred = modelo.predict(self.X_test_s)
        report_dict = classification_report(self.y_test, y_pred, zero_division=0, output_dict=True)
        accuracy = report_dict['accuracy']

        if nome_modelo != 'Baseline':
            precision = report_dict['weighted avg']['precision']
            recall = report_dict['weighted avg']['recall']
            f1 = report_dict['weighted avg']['f1-score']
            specificity = self.calcular_especificidade_media(self.y_test, y_pred, modelo.classes_)
            self.print_output("\n--- MÉTRICAS GERAIS (Ponderadas/Macro) ---")
            self.imprimir_metricas_com_barra("Acurácia", accuracy)
            self.imprimir_metricas_com_barra("Precisão", precision)
            self.imprimir_metricas_com_barra("Recall", recall)
            self.imprimir_metricas_com_barra("F1-Score", f1)
            self.imprimir_metricas_com_barra("Especific.", specificity)

        self.print_output(f"\nACURÁCIA GERAL NO CONJUNTO DE TESTE: {accuracy:.4f}")
        self.print_output("\nRelatório de Classificação Geral:")
        self.print_output(classification_report(self.y_test, y_pred, zero_division=0))

        conf_matrix = confusion_matrix(self.y_test, y_pred, labels=modelo.classes_)
        self.imprimir_matriz_confusao(conf_matrix, modelo.classes_, f"Matriz de Confusão Geral - {nome_modelo}")

        if isinstance(self.X_test_s, pd.DataFrame) and 'color_white' in self.X_test_s.columns:
            self.print_output("\n" + "="*25 + " DESEMPENHO POR TIPO DE VINHO " + "="*25)

            mask_white = self.X_test_s['color_white'] == 1
            y_test_white = self.y_test[mask_white]
            y_pred_white = y_pred[mask_white]
            report_dict_white = classification_report(y_test_white, y_pred_white, zero_division=0, output_dict=True)
            accuracy_white = report_dict_white['accuracy']

            if nome_modelo != 'Baseline':
                precision_white = report_dict_white['weighted avg']['precision']
                recall_white = report_dict_white['weighted avg']['recall']
                f1_white = report_dict_white['weighted avg']['f1-score']
                specificity_white = self.calcular_especificidade_media(y_test_white, y_pred_white, modelo.classes_)
                self.print_output("\n--- Métricas (Vinhos Brancos) ---")
                self.imprimir_metricas_com_barra("Acurácia", accuracy_white)
                self.imprimir_metricas_com_barra("Precisão", precision_white)
                self.imprimir_metricas_com_barra("Recall", recall_white)
                self.imprimir_metricas_com_barra("F1-Score", f1_white)
                self.imprimir_metricas_com_barra("Especific.", specificity_white)

            self.print_output(f"\nPara VINHOS BRANCOS (Acurácia: {accuracy_white:.4f}):")
            self.print_output(classification_report(y_test_white, y_pred_white, zero_division=0))
            conf_matrix_white = confusion_matrix(y_test_white, y_pred_white, labels=modelo.classes_)
            self.imprimir_matriz_confusao(conf_matrix_white, modelo.classes_, f"Matriz de Confusão - Vinhos Brancos ({nome_modelo})")

            mask_red = self.X_test_s['color_white'] == 0
            y_test_red = self.y_test[mask_red]
            y_pred_red = y_pred[mask_red]
            report_dict_red = classification_report(y_test_red, y_pred_red, zero_division=0, output_dict=True)
            accuracy_red = report_dict_red['accuracy']

            if nome_modelo != 'Baseline':
                precision_red = report_dict_red['weighted avg']['precision']
                recall_red = report_dict_red['weighted avg']['recall']
                f1_red = report_dict_red['weighted avg']['f1-score']
                specificity_red = self.calcular_especificidade_media(y_test_red, y_pred_red, modelo.classes_)
                self.print_output("\n--- Métricas (Vinhos Tintos) ---")
                self.imprimir_metricas_com_barra("Acurácia", accuracy_red)
                self.imprimir_metricas_com_barra("Precisão", precision_red)
                self.imprimir_metricas_com_barra("Recall", recall_red)
                self.imprimir_metricas_com_barra("F1-Score", f1_red)
                self.imprimir_metricas_com_barra("Especific.", specificity_red)

            self.print_output(f"\nPara VINHOS TINTOS (Acurácia: {accuracy_red:.4f}):")
            self.print_output(classification_report(y_test_red, y_pred_red, zero_division=0))
            conf_matrix_red = confusion_matrix(y_test_red, y_pred_red, labels=modelo.classes_)
            self.imprimir_matriz_confusao(conf_matrix_red, modelo.classes_, f"Matriz de Confusão - Vinhos Tintos ({nome_modelo})")
        else:
            self.print_output("\n[INFO] Análise por tipo de vinho não é aplicável para dados transformados pelo PCA.")

        self.status_var.set(f"Análise do modelo {nome_modelo} concluída")

    def comparar_todos_modelos(self):
        if self.modelos is None or self.X_train_b is None:
            messagebox.showwarning("Aviso", "Por favor, carregue e processe os dados primeiro.")
            return

        self.print_output("\n--- EXECUTANDO AVALIAÇÃO FINAL DE TODOS OS MODELOS (PIPELINE PADRÃO) ---", clear=True)
        temp_resultados = {}

        for nome, modelo in self.modelos.items():
            self.print_output(f"\n=== Avaliando modelo: {nome} ===")
            resultado_detalhado = self.avaliar_modelo_interno(modelo, nome, self.X_train_b, self.y_train_b, self.X_test_s, self.y_test)
            temp_resultados[nome] = resultado_detalhado

        self.plotar_comparacao(temp_resultados, "Comparação de Acurácia Geral")
        self.status_var.set("Comparação de todos os modelos concluída")

    def comparar_com_pca(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Por favor, carregue os dados primeiro.")
            return

        self.print_output("\n--- REPROCESSANDO DADOS COM PCA ---", clear=True)
        (X_train_pca, y_train_pca, X_test_pca,
         y_test_pca, _, _) = self.processar_dados_para_modelagem(self.df.copy(), usar_pca=True)

        self.print_output("\n--- EXECUTANDO AVALIAÇÃO FINAL DE TODOS OS MODELOS (PIPELINE COM PCA) ---")
        resultados_pca = {}

        for nome, modelo in self.modelos.items():
            self.print_output(f"\n=== Avaliando modelo: {nome} ===")
            resultado_detalhado = self.avaliar_modelo_interno(modelo, nome, X_train_pca, y_train_pca, X_test_pca, y_test_pca)
            resultados_pca[nome] = resultado_detalhado

        self.plotar_comparacao(resultados_pca, "Comparação de Acurácia Geral (com PCA)")
        self.status_var.set("Comparação com PCA concluída")

    def executar_validacao_cruzada(self):
        if self.modelos is None or self.X_train_orig is None:
            messagebox.showwarning("Aviso", "Por favor, carregue e processe os dados primeiro.")
            return

        self.print_output("\n--- AVALIANDO ESTABILIDADE DOS MODELOS COM VALIDAÇÃO CRUZADA ---", clear=True)
        self.print_output("(Executado nos dados de treino originais, antes do SMOTE)")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train_orig)

        resultados_cv = {}
        modelos_cv = self.modelos.copy()
        if 'Baseline' in modelos_cv: del modelos_cv['Baseline']

        for nome, modelo in modelos_cv.items():
            self.print_output(f"\n--- Executando Validação Cruzada para {nome.upper()} ---")
            modelo_clonado = clone(modelo)
            scores = cross_val_score(modelo_clonado, X_scaled, self.y_train_orig, cv=cv, scoring='accuracy', n_jobs=-1)
            self.print_output(f"Acurácia média de {nome}: {scores.mean():.4f} (± {scores.std():.4f})")
            resultados_cv[nome] = scores

        self.print_output("\n--- Resultados da Validação Cruzada (Acurácia por Fold) ---")
        self.print_output(str(pd.DataFrame(resultados_cv).round(4)))

        self.status_var.set("Validação cruzada concluída")

    def avaliar_modelo_interno(self, modelo, nome_modelo, X_train, y_train, X_test, y_test):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        resultados = {
            'accuracy': report_dict['accuracy'],
            'report': report_dict
        }

        if isinstance(X_test, pd.DataFrame) and 'color_white' in X_test.columns:
            mask_white = X_test['color_white'] == 1
            y_test_white = y_test[mask_white]
            y_pred_white = y_pred[mask_white]
            report_dict_white = classification_report(y_test_white, y_pred_white, zero_division=0, output_dict=True)
            resultados['acuracy_white'] = report_dict_white['accuracy']

            mask_red = X_test['color_white'] == 0
            y_test_red = y_test[mask_red]
            y_pred_red = y_pred[mask_red]
            report_dict_red = classification_report(y_test_red, y_pred_red, zero_division=0, output_dict=True)
            resultados['acuracy_red'] = report_dict_red['accuracy']

        return resultados

if __name__ == "__main__":
    root = tk.Tk()
    app = WineQualityApp(root)
    
    def on_closing():
        if messagebox.askokcancel("Sair", "Deseja realmente sair do programa?"):
            root.destroy()
            os._exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()