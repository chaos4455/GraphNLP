# Clusterizador de Tecnologias com FAISS - Documentação

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**

**Versão:** 1.0.0 (Micro-revisão 000000001) - 2024-11-06

## Visão Geral

Este documento descreve a classe `ClusterizadorTechFaiss`, um sistema avançado para clusterização de termos relacionados a tecnologias, utilizando o algoritmo K-Means do FAISS (Facebook AI Similarity Search) e embeddings gerados por um modelo BERT fine-tuned.  O sistema é projetado para agrupar termos tecnológicos com alta precisão, considerando diferentes níveis de granularidade e relações semânticas.

A inspiração para este projeto veio da necessidade de organizar e visualizar grandes conjuntos de dados de tecnologias, permitindo uma melhor compreensão das relações entre elas.  Imagine o desafio de organizar milhares de termos tecnológicos, como linguagens de programação, frameworks, ferramentas de DevOps, serviços em nuvem, etc.  Este sistema resolve esse problema, fornecendo uma representação visual e estruturada desses dados.

## Arquitetura

A classe `ClusterizadorTechFaiss` segue uma arquitetura modular e eficiente, composta pelas seguintes etapas:

1. **Inicialização do Modelo:** Carrega e prepara um modelo `AutoModel` do `transformers` (preferencialmente um modelo BERT fine-tuned para código, como o `microsoft/codebert-base`), juntamente com seu `AutoTokenizer`.  O modelo é configurado para o modo de treinamento (`model.train()`) e um otimizador `AdamW` é inicializado para o fine-tuning.

2. **Geração do Dataset:**  Cria um dataset expandido de tecnologias, considerando categorias principais (cloud, devops, ai_ml, data_engineering, security) e suas relações.  Ele gera combinações e variações de termos para capturar a riqueza semântica das tecnologias.  Por exemplo, para a categoria "cloud", além dos termos principais como "AWS", "Azure", etc., ele gera combinações como "AWS serverless", "Kubernetes containerization", etc.

3. **Processamento de Embeddings:** Gera embeddings para cada termo tecnológico usando o modelo BERT.  Aplica uma normalização avançada em múltiplas etapas: normalização L2, `StandardScaler` e `MinMaxScaler`, para garantir a melhor qualidade dos embeddings para o FAISS.

4. **Clustering Hierárquico:**  Utiliza o FAISS para criar clusters hierárquicos em duas etapas:
    - **Clusters Principais:** Aplica o K-Means para criar um conjunto de clusters principais.
    - **Sub-Clusters:** Para cada cluster principal, aplica novamente o K-Means para criar sub-clusters, refinando a granularidade da clusterização.

5. **Visualização 3D:**  Utiliza o `plotly` para gerar uma visualização 3D dos clusters, utilizando o t-SNE para reduzir a dimensionalidade dos embeddings para 3 dimensões.  A visualização é salva como um arquivo HTML e PNG.

## Funcionalidades Principais

- **Alta Precisão:** Utiliza embeddings de alta qualidade gerados por um modelo BERT fine-tuned para capturar as nuances semânticas das tecnologias.
- **Clustering Hierárquico:** Permite uma organização mais granular dos dados, com clusters principais e sub-clusters.
- **Visualização 3D:** Fornece uma representação visual intuitiva dos clusters, facilitando a análise e interpretação dos resultados.
- **Modularidade:** A arquitetura modular facilita a manutenção e expansão do sistema.
- **Escalabilidade:** O uso do FAISS permite lidar com grandes conjuntos de dados de forma eficiente.

## Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **Sentence Transformers:** Biblioteca para geração de embeddings.
- **FAISS:** Biblioteca para busca de similaridade e clustering.
- **Scikit-learn:** Biblioteca para pré-processamento de dados e algoritmos de machine learning.
- **Rich:** Biblioteca para saída de console aprimorada.
- **Plotly:** Biblioteca para geração de gráficos e visualizações.
- **Transformers:** Biblioteca para modelos de linguagem.
- **NumPy:** Biblioteca para computação numérica.
- **Pandas:** Biblioteca para manipulação de dados.

## Exemplo de Uso

```python
clusterer = ClusterizadorTechFaiss()
clusterer.gerar_dataset_tech()
clusterer.processar_embeddings()
clusterer.criar_clusters_hierarquicos()
html_file, png_file = clusterer.visualizar_clusters_3d()
```

## Considerações Finais

Este sistema representa um avanço significativo na organização e visualização de dados de tecnologias.  Sua capacidade de lidar com grandes conjuntos de dados, sua alta precisão e sua interface visual intuitiva o tornam uma ferramenta poderosa para pesquisadores, desenvolvedores e profissionais de tecnologia.  Futuras melhorias podem incluir a integração com outras fontes de dados, a implementação de algoritmos de clustering mais avançados e a personalização da visualização.

## Versões Anteriores

Os arquivos `cluster_palavras_tech_faiss_v2.py` e `cluster_palavras_tech_faiss_v3.py` representam versões anteriores deste código, com funcionalidades e otimizações diferentes.  A documentação para essas versões não foi criada separadamente, pois as diferenças são menores e o código atual (`cluster_palavras_tech_faiss.py`) representa a versão mais completa e otimizada.


---

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**
