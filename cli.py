# ==============================================================================
# TRABALHO PRÁTICO - INTELIGÊNCIA ARTIFICIAL - VERSÃO HÍBRIDA (CLI)
# Aluno: Alessandro Augusto F. D. Oliveira
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def carregar_dados():
    """Carrega os datasets de vinho tinto e branco e os combina."""
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
    """Exibe informações básicas, estatísticas descritivas e a contagem de dados ausentes."""
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

def visualizar_distribuicoes(df):
    """Gera histogramas e boxplots para cada feature (Itens 6 e 12a)."""
    print("\nGerando gráficos de distribuição e identificação de outliers...")
    print("(Feche cada janela de gráfico para ver a próxima)")
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

def visualizar_correlacao(df):
    """Gera um heatmap de correlação para análise de redundância de atributos."""
    print("\nGerando heatmap de correlação...")
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title('Matriz de Correlação entre Atributos Numéricos', fontsize=16)
    plt.savefig('matriz_correlacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("[INFO] O gráfico da matriz de correlação foi salvo como 'matriz_correlacao.png'")
    print("[ANÁLISE]: Use este gráfico para identificar atributos altamente correlacionados entre si (redundantes) ou com baixa correlação com o alvo ('quality').")

def processar_dados_para_modelagem(df, usar_pca=False, n_components=0.95):
    """Executa todo o pipeline de pré-processamento, com opção de aplicar PCA."""
    print("\n" + "="*60)
    if usar_pca:
        print("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO COM PCA ---")
    else:
        print("--- EXECUTANDO PIPELINE DE PRÉ-PROCESSAMENTO PADRÃO ---")
    print("="*60)
    num_duplicatas = df.duplicated().sum()
    print(f"\n1. Removendo {num_duplicatas} linhas duplicadas...")
    df.drop_duplicates(inplace=True)
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
    """Imprime uma métrica com uma barra de progresso visual."""
    valor = max(0.0, min(1.0, valor))
    blocos_cheios = int(valor * largura_total)
    barra = '█' * blocos_cheios + ' ' * (largura_total - blocos_cheios)
    label_alinhado = f"{label:<10}"
    print(f"{label_alinhado}: [{barra}] {valor:.2f}")

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

def treinar_e_avaliar_modelo(modelo, nome_modelo, X_train, y_train, X_test, y_test, cmap='viridis'):
    """
    Treina, avalia e exibe os resultados de um modelo.
    Se X_test for um DataFrame, também faz a análise por tipo de vinho.
    """
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title(f'Matriz de Confusão Geral - {nome_modelo}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

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
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_white, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.title(f'Matriz de Confusão - Vinhos Brancos\n{nome_modelo}')
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.show()

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
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_red, annot=True, fmt='d', cmap='Reds', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.title(f'Matriz de Confusão - Vinhos Tintos\n{nome_modelo}')
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.show()
    else:
        print("\n[INFO] Análise por tipo de vinho não é aplicável para dados transformados pelo PCA.")
    return resultados_detalhados

def plotar_comparacao_resumo(resultados):
    """Plota gráfico de barras simples comparando a acurácia geral dos modelos."""
    if not resultados:
        print("Nenhum resultado para plotar!")
        return
    modelos = list(resultados.keys())
    acuracias = [res['accuracy'] for res in resultados.values()]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modelos, acuracias, color=plt.cm.viridis(np.linspace(0, 1, len(modelos))))
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácia Geral no Conjunto de Teste', fontsize=16)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    plt.savefig('comparacao_resumo_acuracia.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n[INFO] O gráfico de comparação Resumo foi salvo como 'comparacao_resumo_acuracia.png'")

def plotar_comparacao_detalhada(resultados):
    """Plota gráfico de barras agrupadas comparando desempenho por tipo de vinho."""
    if not resultados:
        print("Nenhum resultado para plotar!")
        return
    modelos = list(resultados.keys())
    acuracias_geral = [res['accuracy'] for res in resultados.values()]
    tem_detalhes = all('acuracy_white' in res and 'acuracy_red' in res for res in resultados.values())
    if not tem_detalhes:
        print("\n[INFO] Não há dados detalhados por tipo de vinho para plotar. Mostrando gráfico de resumo.")
        plotar_comparacao_resumo(resultados)
        return
    acuracias_brancos = [res['acuracy_white'] for res in resultados.values()]
    acuracias_tintos = [res['acuracy_red'] for res in resultados.values()]
    x = np.arange(len(modelos))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, acuracias_geral, width, label='Geral', color='#581845')
    rects2 = ax.bar(x, acuracias_brancos, width, label='Brancos', color='#2E8B57')
    rects3 = ax.bar(x + width, acuracias_tintos, width, label='Tintos', color='#FF5733')
    ax.set_ylabel('Acurácia')
    ax.set_title('Comparação de Desempenho por Tipo de Vinho', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.savefig('comparacao_detalhada_acuracia.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n[INFO] O gráfico de comparação detalhada foi salvo como 'comparacao_detalhada_acuracia.png'")


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
    if 'Baseline' in modelos_cv:
        del modelos_cv['Baseline']
    for nome, modelo in modelos_cv.items():
        print(f"\n--- Executando Validação Cruzada para {nome.upper()} ---")
        modelo_clonado = clone(modelo) 
        scores = cross_val_score(modelo_clonado, X_scaled, y_train_orig, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"Acurácia média de {nome}: {scores.mean():.4f} (± {scores.std():.4f})")
        resultados_cv[nome] = scores
    plotar_comparacao_cv_boxplot(resultados_cv)

def plotar_comparacao_cv_boxplot(resultados_cv):
    plt.figure(figsize=(12, 7))
    df_resultados = pd.DataFrame(resultados_cv)
    sns.boxplot(data=df_resultados)
    plt.title('Comparação de Desempenho na Validação Cruzada (5 Folds)', fontsize=16)
    plt.ylabel('Acurácia')
    plt.xlabel('Modelos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('comparacao_cv.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n[INFO] O gráfico da validação cruzada foi salvo como 'comparacao_cv.png'")


def menu_principal():
    df_inicial = carregar_dados()
    if df_inicial is None: return
    
    X_train_b, y_train_b, X_test_s, y_test, X_train_orig, y_train_orig = processar_dados_para_modelagem(df_inicial.copy(), usar_pca=False)
    modelos = get_modelos_com_melhores_parametros()

    while True:
        print("\n" + "="*24 + " MENU PRINCIPAL " + "="*24)
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
        print("8. EXECUTAR E COMPARAR TODOS OS MODELOS (com gráfico detalhado)")
        print("9. EXECUTAR E COMPARAR TODOS com PCA (com gráfico de resumo)")
        print("10. Realizar Validação Cruzada (Verificar Estabilidade)")
        print("\n0. Sair")
        print("="*66)

        escolha = input("Digite o número da sua escolha: ")

        if escolha == '1':
            analise_exploratoria(df_inicial)
        elif escolha == '2':
            visualizar_distribuicoes(df_inicial)
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
            plotar_comparacao_detalhada(temp_resultados)

        elif escolha == '9':
            print("\n--- REPROCESSANDO DADOS COM PCA ---")
            X_train_pca, y_train_pca, X_test_pca, y_test_pca, _, _ = processar_dados_para_modelagem(df_inicial.copy(), usar_pca=True)
            print("\n--- EXECUTANDO AVALIAÇÃO FINAL DE TODOS OS MODELOS (PIPELINE COM PCA) ---")
            resultados_pca = {}
            for nome, modelo in modelos.items():
                resultado_detalhado = treinar_e_avaliar_modelo(modelo, nome, X_train_pca, y_train_pca, X_test_pca, y_test_pca)
                resultados_pca[nome] = resultado_detalhado
            plotar_comparacao_detalhada(resultados_pca)
            
        elif escolha == '10':
            executar_validacao_cruzada(X_train_orig, y_train_orig, modelos)
            
        elif escolha == '0':
            print("Encerrando o programa. Até mais!")
            break
        else:
            print("Escolha inválida. Por favor, tente novamente.")

if __name__ == "__main__":
    menu_principal()