import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
  # 1. Carregar a base de dados
  data_path = "./database/data.csv"
  df = pd.read_csv(data_path)

  # Exibindo as primeiras linhas para inspeção
  print("Primeiras linhas da base de dados:\n", df.head())

  # 2. Selecionar variáveis numéricas
  numerical_features = ['Idade']
  data_numeric = df[numerical_features]

  # Padronização dos dados
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data_numeric)

  # 3. Aplicar o algoritmo K-Means
  kmeans = KMeans(n_clusters=3, random_state=42)
  kmeans.fit(data_scaled)

  # Adicionando os rótulos de cluster ao DataFrame original
  df['Cluster'] = kmeans.labels_

  # Exibindo os resultados
  print("Dados com clusters:\n", df)

  # 4. Analisar os resultados
  # Tamanho dos clusters
  cluster_sizes = df['Cluster'].value_counts()
  print("Tamanhos dos clusters:\n", cluster_sizes)

  # Visualização dos clusters com idades reais
  plt.figure(figsize=(8, 5))
  plt.scatter(df['Idade'], [0] * len(df), c=kmeans.labels_, cmap='viridis', s=50)
  plt.title('Distribuição dos Clusters (K-Means)')
  plt.xlabel('Idade')
  plt.yticks([])  # Removendo o eixo y
  plt.show()

except Exception as e:
  print("Erro ao executar o script:", e)
