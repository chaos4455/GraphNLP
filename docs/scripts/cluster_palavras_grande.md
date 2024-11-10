# Gerador e Clusterizador de Palavras em Larga Escala - Documentação

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**

**Versão:** 1.0.0 (Micro-revisão 000000001) - 2024-11-06

## Visão Geral

Este documento descreve o script `cluster_palavras_grande.py`, que gera um extenso dataset de palavras, calcula embeddings usando o modelo `all-MiniLM-L6-v2` do Sentence Transformers, aplica o algoritmo K-Means do FAISS para clusterização e, por fim, analisa e salva os resultados.  O objetivo é demonstrar a capacidade de processar e analisar grandes conjuntos de dados textuais usando técnicas de processamento de linguagem natural e aprendizado de máquina.

## Arquitetura

O script segue uma arquitetura modular, composta pelas seguintes etapas:

1. **Geração do Dataset:** Gera um dataset de aproximadamente 9800 palavras, combinando palavras-chave de nove temas diferentes com modificadores.  A geração de palavras é feita de forma programática, criando combinações e variações para aumentar a riqueza do dataset.

2. **Processamento de Embeddings:**  Utiliza o modelo `all-MiniLM-L6-v2` do Sentence Transformers para gerar embeddings para cada palavra no dataset.  Aplica `StandardScaler` para normalizar os embeddings antes da clusterização.

3. **Clusterização com FAISS:**  Utiliza o algoritmo K-Means do FAISS para agrupar as palavras em nove clusters, um para cada tema principal.  O FAISS é escolhido por sua eficiência no processamento de grandes conjuntos de dados.

4. **Análise de Clusters:**  Analisa os clusters gerados, calculando o tamanho de cada cluster, identificando o tema predominante em cada cluster e exibindo exemplos de palavras em cada cluster.  Os resultados da análise são armazenados em um dicionário.

5. **Salvamento de Resultados:**  Salva os resultados em dois arquivos:
    - `clusters_palavras.csv`:  Contém o dataset com as palavras e seus respectivos clusters.
    - `analise_clusters.json`:  Contém a análise detalhada dos clusters.

## Funcionalidades Principais

- **Geração de Dataset Programática:** Gera um dataset extenso de forma eficiente.
- **Embeddings com Sentence Transformers:** Gera embeddings de alta qualidade.
- **Clusterização com FAISS:**  Clusterização eficiente de grandes conjuntos de dados.
- **Análise Detalhada:**  Fornece insights sobre a estrutura dos clusters.
- **Salvamento de Resultados:**  Permite a preservação e posterior análise dos resultados.

## Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **Sentence Transformers:** Biblioteca para geração de embeddings.
- **FAISS:** Biblioteca para busca de similaridade e clustering.
- **Scikit-learn:** Biblioteca para pré-processamento de dados.
- **Rich:** Biblioteca para saída de console aprimorada.
- **NumPy:** Biblioteca para computação numérica.
- **Pandas:** Biblioteca para manipulação de dados.

## Exemplo de Uso

O script é executado diretamente: `python cluster_palavras_grande.py`

## Considerações Finais

Este script demonstra uma abordagem eficiente para processar e analisar grandes conjuntos de dados textuais, combinando técnicas de processamento de linguagem natural e aprendizado de máquina.  A modularidade do código facilita a manutenção e expansão do sistema.


---

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**
