# Projeto de Análise de Dados

Este projeto foi desenvolvido para realizar análises de dados utilizando técnicas de aprendizado supervisionado e não supervisionado. O projeto é composto por dois scripts principais: `supervised.py` e `unsupervised.py`, que aplicam algoritmos de aprendizado de máquina em uma base de dados fictícia.

## Análise Supervisionada

### `supervised.py`
Este script realiza uma análise supervisionada utilizando o algoritmo k-NN (K-Nearest Neighbors). Ele segue os seguintes passos:
1. Carrega a base de dados `database/data.csv`.
2. Processa os dados, transformando variáveis categóricas em numéricas.
3. Divide os dados em conjuntos de treino e teste.
4. Padroniza os dados numéricos.
5. Treina o modelo k-NN.
6. Realiza previsões e avalia o modelo.
7. Visualiza os resultados das previsões.

## Análise Não Supervisionada

### `unsupervised.py`
Este script realiza uma análise não supervisionada utilizando o algoritmo K-Means. Ele segue os seguintes passos:
1. Carrega a base de dados `database/data.csv`.
2. Seleciona variáveis numéricas para análise.
3. Padroniza os dados.
4. Aplica o algoritmo K-Means para agrupar os dados.
5. Adiciona os rótulos de cluster ao DataFrame original.
6. Analisa os resultados e visualiza a distribuição dos clusters.

## Disciplina
Mineração de Dados

## Discente
Henrique Menezes Oliveira

## Professor
Rení Aparecido Norberto Pinto