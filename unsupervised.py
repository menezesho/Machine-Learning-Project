import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
  # 1. Carregamento da base de dados
  data_path = "./database/data.csv"
  df = pd.read_csv(data_path)

  # 2. Normalização dos dados
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(df[['Idade', 'Renda', 'Gastos Mensais']])

  # 3. Aplicação do algoritmo K-Means
  kmeans = KMeans(n_clusters=3, random_state=42)
  df['Cluster'] = kmeans.fit_predict(data_scaled)

  # Exibindo os resultados
  print("Dados com clusters:\n", df)

  # 4. Análise e visualização dos clusters
  plt.figure(figsize=(10, 6))
  for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Renda'], cluster_data['Gastos Mensais'], label=f'Cluster {cluster}')
      
  plt.title('Clusters por Renda e Gastos Mensais')
  plt.xlabel('Renda ($)')
  plt.ylabel('Gastos Mensais ($)')
  plt.legend()
  plt.show()

except Exception as e:
  print("Erro ao executar o script:", e)