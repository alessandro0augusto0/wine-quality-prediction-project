# ==============================================================================
# TRABALHO PRÁTICO - INTELIGÊNCIA ARTIFICIAL - VERSÃO MANUAL (TEXTO)
# Aluno: Alessandro Augusto F. D. Oliveira
# ==============================================================================

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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def imprimir_matriz_confusao_manual(mcm, labels, title):
    """Imprime uma matriz de confusão formatada no terminal."""
    print(f"\n--- {title} ---")
    
    header = f"{'Verdadeiro\\Previsto':<12}" + " | ".join([f"{str(l):^8}" for l in labels])
    print(header)
    print("-" * len(header))

    for i, label in enumerate(labels):
        row_str = f"{str(label):<12}"
        for j in range(len(labels)):
            row_str += f" | {mcm[i, j]:^8}"
        print(row_str)
    print("-" * len(header))


def plotar_comparacao_manual(resultados, title):
    """Imprime um gráfico de barras de texto comparando a acurácia dos modelos."""
    print(f"\n--- {title} ---")
    if not resultados:
        print("Nenhum resultado para plotar.")
        return

    max_len = max(len(nome) for nome in resultados.keys())
    
    for nome, res in resultados.items():
        acc = res['accuracy']
        bar_len = int(acc * 40)
        bar = '█' * bar_len
        print(f"{nome:<{max_len}} | {bar} {acc:.4f}")


def visualizar_distribuicoes_manual(df):
    """Imprime histogramas de texto e estatísticas descritivas."""
    print("\n--- Distribuição dos Dados e Análise de Outliers (Versão Texto) ---")
    colunas_features = df.select_dtypes(include=['float64', 'int64']).drop('quality', axis=1).columns
    
    for coluna in colunas_features:
        print(f"\n===== Feature: {coluna} =====")
        
        print("\n[Histograma Simplificado]")
        counts, bin_edges = np.histogram(df[coluna])
        max_count = counts.max()
        for i in range(len(counts)):
            bar_len = int((counts[i] / max_count) * 30) if max_count > 0 else 0
            bar = '█' * bar_len
            print(f"{bin_edges[i]:>6.2f} - {bin_edges[i+1]:>6.2f} | {bar} ({counts[i]})")
            
        print("\n[Estatísticas Descritivas (substitui o Boxplot)]")
        print(df.groupby('color')[coluna].describe().transpose().round(2))
        input("\nPressione Enter para continuar para a próxima feature...")

def carregar_dados():
    try:
        df_red = pd.read_csv('docs/winequality-red.csv', sep=';')
        df_white = pd.read_csv('docs/winequality-white.csv', sep=';')
        print("Arquivos CSV carregados com sucesso!")
        df_red['color'] = 'red'
        df_white['color'] = 'white'
        df = pd.concat([df_red, df_white], ignore_index=True)
        return df
    except FileNotFoundError:
        print("ERRO: Arquivos 'winequality-red.csv' e 'winequality-white.csv' não encontrados na pasta 'docs/'.")
        return None

def analise_exploratoria(df):
    print("\n" + "="*60)
    print("--- 1. ANÁLISE EXPLORATÓRIA GERAL ---")
    print("="*60)
    print("\n--- INFORMAÇÕES BÁSICAS DO DATASET ---")
    df.info(verbose=False)
    print("\n--- VERIFICAÇÃO DE DADOS AUSENTES (Item 12d) ---")
    print(df.isnull().sum())
    print("\n[ANÁLISE]: O resultado acima mostra que não há dados ausentes na base.")
    print("\n--- MEDIDAS DE LOCALIDADE E ESPALHAMENTO (Itens 4 e 5) ---")
    print(df.describe().transpose())

def visualizar_correlacao(df):
    """Imprime a matriz de correlação no terminal."""
    print("\n--- Matriz de Correlação (Versão Texto) ---")
    corr = df.select_dtypes(include=np.number).corr()
    print(corr.round(2))
    print("\n[ANÁLISE]: Use a tabela para identificar atributos com correlação forte (>0.7 ou <-0.7) entre si ou com o alvo ('quality').")

def processar_dados_para_modelagem(df, usar_pca=False, n_components=0.95):
    print("\n" + "="*60)
    if usar_pca:
        print("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO COM PCA ---")
    else:
        print("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO PADRÃO ---")
    print("="*60)
    num_duplicatas = df.duplicated().sum()
    if num_duplicatas > 0:
        print(f"\n1. Removendo {num_duplicatas} linhas duplicadas...")
        df.drop_duplicates(inplace=True)
    else:
        print("\n1. Nenhuma linha duplicada encontrada.")
    print("2. Criando categorias 'ruim', 'normal', 'bom'...")
    bins = [0, 4, 6, 10]
    labels = ['ruim', 'normal', 'bom']
    df['quality_category'] = pd.cut(df['quality'], bins=bins, labels=labels, right=True)
    print("3. Convertendo 'color' para formato numérico...")
    df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True, dtype=int)
    print("4. Criando coluna para estratificação por qualidade E cor...")
    df_encoded['stratify_col'] = df_encoded['quality_category'].astype(str) + '_' + df_encoded['color_white'].astype(str)
    X = df_encoded.drop(columns=['quality', 'quality_category', 'stratify_col'])
    y = df_encoded['quality_category']
    stratify_col = df_encoded['stratify_col']
    print("5. Separando dados em treino (80%) e teste (20%) com estratificação dupla...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col)
    print("6. Normalizando features com StandardScaler...")
    scaler = StandardScaler()
    cols_to_scale = [col for col in X_train.columns if col != 'color_white']
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    if usar_pca:
        print(f"7. Aplicando Redução de Dimensionalidade (PCA) para manter {n_components*100}% da variância...")
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"   -> PCA selecionou {pca.n_components_} componentes dos {X_train_scaled.shape[1]} originais.")
    else:
        print("7. Redução de Dimensionalidade (PCA) não aplicada.")
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    print("8. Balanceando conjunto de treino com SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)
    print("\nPRÉ-PROCESSAMENTO CONCLUÍDO!")
    return X_train_balanced, y_train_balanced, X_test_final, y_test, X_train, y_train



def imprimir_metricas_com_barra(label, valor, largura_total=25):
    valor = max(0.0, min(1.0, valor))
    blocos_cheios = int(valor * largura_total)
    barra = '█' * blocos_cheios + ' ' * (largura_total - blocos_cheios)
    label_alinhado = f"{label:<10}"
    print(f"{label_alinhado}: [{barra}] {valor:.2f}")

def calcular_especificidade_media(y_true, y_pred, labels):
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

def treinar_e_avaliar_modelo(modelo, nome_modelo, X_train, y_train, X_test, y_test): # Removido cmap
    print(f"\n" + "-"*20 + f" ANÁLISE DETALHADA: {nome_modelo} " + "-"*20)
    start_time = time()
    modelo.fit(X_train, y_train)
    end_time = time()
    print(f"Tempo de treinamento: {end_time - start_time:.2f} segundos")

    y_pred = modelo.predict(X_test)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    accuracy = report_dict['accuracy']

    if nome_modelo != 'Baseline':
        precision = report_dict['weighted avg']['precision']
        recall = report_dict['weighted avg']['recall']
        f1 = report_dict['weighted avg']['f1-score']
        specificity = calcular_especificidade_media(y_test, y_pred, modelo.classes_)
        print("\n--- MÉTRICAS GERAIS (Ponderadas/Macro) ---")
        imprimir_metricas_com_barra("Acurácia", accuracy)
        imprimir_metricas_com_barra("Precisão", precision)
        imprimir_metricas_com_barra("Recall", recall)
        imprimir_metricas_com_barra("F1-Score", f1)
        imprimir_metricas_com_barra("Especific.", specificity)

    print(f"\nACURÁCIA GERAL NO CONJUNTO DE TESTE: {accuracy:.4f}")
    print("\nRelatório de Classificação Geral:")
    print(classification_report(y_test, y_pred, zero_division=0))

    conf_matrix = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    imprimir_matriz_confusao_manual(conf_matrix, modelo.classes_, f"Matriz de Confusão Geral - {nome_modelo}")
    
    resultados_detalhados = {'accuracy': accuracy}

    if isinstance(X_test, pd.DataFrame) and 'color_white' in X_test.columns:
        print("\n" + "="*25 + " DESEMPENHO POR TIPO DE VINHO " + "="*25)

        mask_white = X_test['color_white'] == 1
        y_test_white = y_test[mask_white]
        y_pred_white = y_pred[mask_white]
        report_dict_white = classification_report(y_test_white, y_pred_white, zero_division=0, output_dict=True)
        accuracy_white = report_dict_white['accuracy']
        resultados_detalhados['acuracy_white'] = accuracy_white
        if nome_modelo != 'Baseline':
            precision_white = report_dict_white['weighted avg']['precision']
            recall_white = report_dict_white['weighted avg']['recall']
            f1_white = report_dict_white['weighted avg']['f1-score']
            specificity_white = calcular_especificidade_media(y_test_white, y_pred_white, modelo.classes_)
            print("\n--- Métricas (Vinhos Brancos) ---")
            imprimir_metricas_com_barra("Acurácia", accuracy_white)
            imprimir_metricas_com_barra("Precisão", precision_white)
            imprimir_metricas_com_barra("Recall", recall_white)
            imprimir_metricas_com_barra("F1-Score", f1_white)
            imprimir_metricas_com_barra("Especific.", specificity_white)

        print(f"\nPara VINHOS BRANCOS (Acurácia: {accuracy_white:.4f}):")
        print(classification_report(y_test_white, y_pred_white, zero_division=0))
        conf_matrix_white = confusion_matrix(y_test_white, y_pred_white, labels=modelo.classes_)
        imprimir_matriz_confusao_manual(conf_matrix_white, modelo.classes_, f"Matriz de Confusão - Vinhos Brancos ({nome_modelo})")
        
        mask_red = X_test['color_white'] == 0
        y_test_red = y_test[mask_red]
        y_pred_red = y_pred[mask_red]
        report_dict_red = classification_report(y_test_red, y_pred_red, zero_division=0, output_dict=True)
        accuracy_red = report_dict_red['accuracy']
        resultados_detalhados['acuracy_red'] = accuracy_red
        if nome_modelo != 'Baseline':
            precision_red = report_dict_red['weighted avg']['precision']
            recall_red = report_dict_red['weighted avg']['recall']
            f1_red = report_dict_red['weighted avg']['f1-score']
            specificity_red = calcular_especificidade_media(y_test_red, y_pred_red, modelo.classes_)
            print("\n--- Métricas (Vinhos Tintos) ---")
            imprimir_metricas_com_barra("Acurácia", accuracy_red)
            imprimir_metricas_com_barra("Precisão", precision_red)
            imprimir_metricas_com_barra("Recall", recall_red)
            imprimir_metricas_com_barra("F1-Score", f1_red)
            imprimir_metricas_com_barra("Especific.", specificity_red)

        print(f"\nPara VINHOS TINTOS (Acurácia: {accuracy_red:.4f}):")
        print(classification_report(y_test_red, y_pred_red, zero_division=0))
        conf_matrix_red = confusion_matrix(y_test_red, y_pred_red, labels=modelo.classes_)
        imprimir_matriz_confusao_manual(conf_matrix_red, modelo.classes_, f"Matriz de Confusão - Vinhos Tintos ({nome_modelo})")
    else:
        print("\n[INFO] Análise por tipo de vinho não é aplicável para dados transformados pelo PCA.")
    return resultados_detalhados

def get_modelos_com_melhores_parametros():
    baseline = DummyClassifier(strategy="most_frequent")
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')
    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=5, criterion='gini', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=1500, random_state=42)
    return {'Baseline': baseline, 'K-NN': knn, 'Árvore de Decisão': dt, 'Rede Neural (MLP)': mlp}

def executar_validacao_cruzada(X_train_orig, y_train_orig, modelos):
    print("\n--- AVALIANDO ESTABILIDADE DOS MODELOS COM VALIDAÇÃO CRUZADA ---")
    print("(Executado nos dados de treino originais, antes do SMOTE)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_orig)
    resultados_cv = {}
    modelos_cv = modelos.copy()
    if 'Baseline' in modelos_cv: del modelos_cv['Baseline']
    for nome, modelo in modelos_cv.items():
        print(f"\n--- Executando Validação Cruzada para {nome.upper()} ---")
        modelo_clonado = clone(modelo) 
        scores = cross_val_score(modelo_clonado, X_scaled, y_train_orig, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"Acurácia média de {nome}: {scores.mean():.4f} (± {scores.std():.4f})")
        resultados_cv[nome] = scores
    print("\n--- Resultados da Validação Cruzada (Acurácia por Fold) ---")
    print(pd.DataFrame(resultados_cv).round(4))


def menu_principal():
    df_inicial = carregar_dados()
    if df_inicial is None: return
    
    X_train_b, y_train_b, X_test_s, y_test, X_train_orig, y_train_orig = processar_dados_para_modelagem(df_inicial.copy(), usar_pca=False)
    modelos = get_modelos_com_melhores_parametros()

    while True:
        print("\n" + "="*24 + " MENU PRINCIPAL (MODO MANUAL) " + "="*24)
        print("--- Análise e Exploração ---")
        print("1. Análise Geral (Info, Stats, Dados Ausentes)")
        print("2. Visualizar Distribuição dos Dados e Outliers")
        print("3. Visualizar Matriz de Correlação")
        print("\n--- Análise Detalhada de Modelos (Pipeline Padrão) ---")
        print("4. Avaliar Modelo Baseline")
        print("5. Avaliar K-NN")
        print("6. Avaliar Árvore de Decisão")
        print("7. Avaliar Rede Neural (MLP)")
        print("\n--- Experimentos e Comparações ---")
        print("8. EXECUTAR E COMPARAR TODOS OS MODELOS")
        print("9. EXECUTAR E COMPARAR TODOS com PCA")
        print("10. Realizar Validação Cruzada (Verificar Estabilidade)")
        print("\n0. Sair")
        print("="*70)

        escolha = input("Digite o número da sua escolha: ")

        if escolha == '1':
            analise_exploratoria(df_inicial)
        elif escolha == '2':
            visualizar_distribuicoes_manual(df_inicial)
        elif escolha == '3':
            visualizar_correlacao(df_inicial)
        elif escolha in ['4', '5', '6', '7']:
            mapa_escolha = {'4': 'Baseline', '5': 'K-NN', '6': 'Árvore de Decisão', '7': 'Rede Neural (MLP)'}
            nome_modelo = mapa_escolha[escolha]
            modelo_obj = modelos[nome_modelo]
            treinar_e_avaliar_modelo(modelo_obj, nome_modelo, X_train_b, y_train_b, X_test_s, y_test)
        elif escolha == '8':
            print("\n--- EXECUTANDO AVALIAÇÃO FINAL DE TODOS OS MODELOS (PIPELINE PADRÃO) ---")
            temp_resultados = {}
            for nome, modelo in modelos.items():
                resultado_detalhado = treinar_e_avaliar_modelo(modelo, nome, X_train_b, y_train_b, X_test_s, y_test)
                temp_resultados[nome] = resultado_detalhado
            plotar_comparacao_manual(temp_resultados, "Comparação de Acurácia Geral")
        elif escolha == '9':
            print("\n--- REPROCESSANDO DADOS COM PCA ---")
            X_train_pca, y_train_pca, X_test_pca, y_test_pca, _, _ = processar_dados_para_modelagem(df_inicial.copy(), usar_pca=True)
            print("\n--- EXECUTANDO AVALIAÇÃO FINAL DE TODOS OS MODELOS (PIPELINE COM PCA) ---")
            resultados_pca = {}
            for nome, modelo in modelos.items():
                resultado_detalhado = treinar_e_avaliar_modelo(modelo, nome, X_train_pca, y_train_pca, X_test_pca, y_test_pca)
                resultados_pca[nome] = resultado_detalhado
            plotar_comparacao_manual(resultados_pca, "Comparação de Acurácia Geral (com PCA)")
        elif escolha == '10':
            executar_validacao_cruzada(X_train_orig, y_train_orig, modelos)
        elif escolha == '0':
            print("Encerrando o programa. Até mais!")
            break
        else:
            print("Escolha inválida. Por favor, tente novamente.")

if __name__ == "__main__":
    menu_principal()