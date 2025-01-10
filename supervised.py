import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    # 1. Carregamento da base de dados
    data_path = "./database/data.csv"
    df = pd.read_csv(data_path)

    # 2. Criação do campo 'AltoGasto' como variável dependente
    df['AltoGasto'] = (df['Gastos Mensais'] > 3000).astype(int)  # 1: Alto gasto, 0: Baixo/Médio gasto

    # Variáveis independentes: Idade e Renda
    # Variável dependente: AltoGasto
    X = df[['Idade', 'Renda']]
    y = df['AltoGasto']

    # Divisão da base de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Padronização dos dados numéricos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Criação do classificador k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # Treinando o modelo
    knn.fit(X_train_scaled, y_train)

    # 5. Realização de previsões
    y_pred = knn.predict(X_test_scaled)

    # 6. Avaliação do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    # Exibição do relatório de classificação
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=['Baixo/Médio Gasto', 'Alto Gasto']))

    # 7. Visualização dos resultados
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_test['Renda'], X_test['Idade'], c=y_pred, cmap='coolwarm', s=50)
    plt.title('Previsões do Modelo k-NN')
    plt.xlabel('Renda ($)')
    plt.ylabel('Idade')
    plt.colorbar(scatter, label='Alto Gasto (1 = Sim, 0 = Não)')
    plt.show()

except Exception as e:
    print("Erro ao executar o script:", e)
