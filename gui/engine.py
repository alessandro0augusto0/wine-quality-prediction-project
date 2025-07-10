import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.base import clone

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def calcular_especificidade_media(y_true, y_pred, labels):
    """Calcula a média da especificidade para um problema multi-classe (macro average)."""
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


class MLEngine:
    def __init__(self):
        self.df = None
        self.X_train_orig = None
        self.y_train_orig = None
        self.trained_models = {}
        self.cv_results = None
        self.feature_names = None

    def load_data(self):
        """Carrega os dados dos arquivos CSV e faz o pré-processamento básico"""
        try:
            df_red = pd.read_csv('docs/winequality-red.csv', sep=';')
            df_white = pd.read_csv('docs/winequality-white.csv', sep=';')
            
            df_red['color'] = 'red'
            df_white['color'] = 'white'
            df_red['color_white'] = 0
            df_white['color_white'] = 1
            
            self.df = pd.concat([df_red, df_white], ignore_index=True)
            
            if self.df.isnull().sum().sum() > 0:
                print("Aviso: Dados contêm valores ausentes que serão tratados.")
                self.df = self.df.dropna()
            
            return self.df
        except FileNotFoundError as e:
            print(f"Erro ao carregar arquivos: {e}")
            return None
        except Exception as e:
            print(f"Erro inesperado: {e}")
            return None

    def get_exploratory_analysis(self):
        """Gera uma análise exploratória dos dados"""
        if self.df is None:
            return "Dados não carregados. Por favor, carregue os dados primeiro."
        
        analysis = "=== INFORMAÇÕES GERAIS ===\n"
        analysis += f"Dimensões: {self.df.shape}\n"
        analysis += f"Tipos de Dados:\n{self.df.dtypes.to_string()}\n\n"
        
        analysis += "=== DADOS AUSENTES ===\n"
        analysis += f"{self.df.isnull().sum().to_string()}\n\n"
        
        analysis += "=== ESTATÍSTICAS DESCRITIVAS ===\n"
        analysis += f"{self.df.describe().transpose().to_string()}\n\n"
        
        if 'quality' in self.df.columns:
            analysis += "=== DISTRIBUIÇÃO DE QUALIDADE ===\n"
            analysis += f"{self.df['quality'].value_counts().sort_index().to_string()}"
        
        return analysis

    def _preprocess_data(self, use_pca=False):
        """Função interna para pré-processamento dos dados"""
        df_processed = self.df.copy().drop_duplicates().dropna()
        
        bins = [0, 4, 6, 10]
        labels = ['ruim', 'normal', 'bom']
        df_processed['quality_category'] = pd.cut(
            df_processed['quality'], 
            bins=bins, 
            labels=labels, 
            right=True
        )
        
        if 'color_white' not in df_processed.columns:
            df_processed['color_white'] = df_processed['color'].apply(
                lambda x: 1 if x == 'white' else 0
            )
        
        df_processed['stratify_col'] = (
            df_processed['quality_category'].astype(str) + '_' + 
            df_processed['color_white'].astype(str)
        )
        
        X = df_processed.drop(columns=['quality', 'quality_category', 'stratify_col', 'color'])
        y = df_processed['quality_category']
        stratify_col = df_processed['stratify_col']
        
        self.X_train_orig, X_test, self.y_train_orig, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_col
        )
        
        scaler = StandardScaler()
        cols_to_scale = [col for col in self.X_train_orig.columns if col != 'color_white']
        X_train_scaled = self.X_train_orig.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[cols_to_scale] = scaler.fit_transform(self.X_train_orig[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
        if use_pca:
            pca = PCA(n_components=0.95)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            self.feature_names = [f"PC{i+1}" for i in range(X_train_scaled.shape[1])]
        else:
            self.feature_names = list(self.X_train_orig.columns)
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, self.y_train_orig)
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test

    def get_models(self):
        """Retorna um dicionário de modelos para experimentação"""
        return {
            'Baseline': DummyClassifier(strategy="most_frequent"),
            'K-NN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
            'Árvore de Decisão': DecisionTreeClassifier(
                max_depth=8, min_samples_split=5, criterion='gini', random_state=42),
            'Rede Neural (MLP)': MLPClassifier(
                hidden_layer_sizes=(50, 30), activation='relu', solver='adam', 
                max_iter=1500, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42)
        }

    def run_experiment(self, use_pca=False):
        """Executa o experimento completo com todos os modelos"""
        if self.df is None:
            return None
        
        try:
            X_train, X_test, y_train, y_test = self._preprocess_data(use_pca)
            models = self.get_models()
            results = {}
            self.trained_models.clear()
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                report = classification_report(
                    y_test, y_pred, 
                    output_dict=True, 
                    zero_division=0
                )
                
                cm = confusion_matrix(
                    y_test, y_pred, 
                    labels=model.classes_
                )

                specificity = calcular_especificidade_media(y_test, y_pred, labels=model.classes_)
                
                results[name] = {
                    'report_dict': report,
                    'cm': cm,
                    'specificity': specificity,
                    'classes': model.classes_,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'X_test_df': pd.DataFrame(X_test, columns=self.feature_names) if not use_pca else None
                }
                self.trained_models[name] = model
            
            return results
        
        except Exception as e:
            print(f"Erro durante o experimento: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_cross_validation(self):
        """Executa validação cruzada para todos os modelos"""
        if self.X_train_orig is None:
            return None
        
        try:
            models = self.get_models()
            if 'Baseline' in models:
                del models['Baseline']
                
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X_train_orig)
            
            self.cv_results = {}
            
            for name, model in models.items():
                scores = cross_val_score(
                    clone(model), 
                    X_scaled, 
                    self.y_train_orig, 
                    cv=cv, 
                    scoring='accuracy', 
                    n_jobs=-1
                )
                self.cv_results[name] = scores
            
            return self.cv_results
        
        except Exception as e:
            print(f"Erro na validação cruzada: {e}")
            return None