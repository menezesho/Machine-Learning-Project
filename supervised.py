import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
  # 1. Carregar a base de dados
  data_path = "./database/data.csv"
  df = pd.read_csv(data_path)

  # Exibindo as primeiras linhas para inspeção
  print("Primeiras linhas da base de dados:\n", df.head())

  # 2. Processamento de dados
  # Variáveis independentes: Idade e Comida
  # Variável dependente (target): Marca
  X = df[['Idade', 'Comida']]  
  y = df['Marca']  

  # Transformar colunas categóricas em valores numéricos (Label Encoding)
  label_encoder_comida = LabelEncoder()
  X['Comida'] = label_encoder_comida.fit_transform(X['Comida'])

  label_encoder_marca = LabelEncoder()
  y = label_encoder_marca.fit_transform(y)

  # Divisão da base de dados em treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 3. Padronização dos dados numéricos
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 4. Criar o classificador k-NN
  knn = KNeighborsClassifier(n_neighbors=3)
  
  # Treinando o modelo
  knn.fit(X_train_scaled, y_train)

  # 5. Realizar previsões
  y_pred = knn.predict(X_test_scaled)

  # 6. Avaliar o modelo
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

  # Exibir o relatório de classificação
  print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=label_encoder_marca.classes_))

  # 7. Visualização dos resultados
  plt.figure(figsize=(8, 5))
  plt.scatter(X_test['Idade'], X_test['Comida'], c=y_pred, cmap='viridis', s=50)
  plt.title('Previsões do Modelo k-NN')
  plt.xlabel('Idade')
  plt.ylabel('Comida (codificada)')
  plt.colorbar(label='Marca (codificada)')
  plt.show()

except Exception as e:
  print("Erro ao executar o script:", e)
