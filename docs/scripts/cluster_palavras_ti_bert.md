# Visualizador 3D de Clusters de Palavras em TI/DevOps usando BERT - Documentação

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**

**Versão:** 1.0.0 (Micro-revisão 000000001) - 2024-11-06


## Visão Geral

Este documento descreve a classe `ClusterizadorTIBert`, um sistema que gera um grande conjunto de palavras relacionadas a TI/DevOps, utilizando o modelo BERT para expandir termos-chave, realiza a clusterização desses termos usando o FAISS e, finalmente, visualiza os clusters resultantes em 3D usando Pygame e OpenGL.  O sistema é projetado para fornecer uma representação visual e interativa de um vasto espaço semântico de termos tecnológicos.

A inspiração para este projeto surgiu da necessidade de visualizar e explorar a complexa rede de relações entre conceitos em TI.  Este sistema permite uma exploração interativa, permitindo ao usuário rotacionar, aproximar e afastar a visualização 3D, facilitando a identificação de padrões e relações entre os clusters de palavras.

## Arquitetura

A classe `ClusterizadorTIBert` é composta por várias etapas principais:

1. **Inicialização:** Inicializa o modelo BERT (`bert-base-uncased`), o tokenizer correspondente e configura o ambiente Pygame/OpenGL para a visualização 3D.

2. **Definição de Temas:** Define um dicionário (`self.temas_ti`) contendo 55 temas de TI/DevOps, cada um com um conjunto de palavras-chave e contextos relacionados.

3. **Geração de Palavras:**  Utiliza o modelo BERT para gerar novas palavras relacionadas a cada palavra-chave e contexto definido nos temas.  Este processo expande significativamente o conjunto de palavras, criando um dataset rico e representativo do domínio da TI.

4. **Processamento de Embeddings:** Gera embeddings para cada palavra gerada usando o modelo `all-MiniLM-L6-v2` do Sentence Transformers.  Aplica `StandardScaler` para normalizar os embeddings.

5. **Clusterização:** Utiliza o algoritmo K-Means do FAISS para agrupar as palavras em 55 clusters, um para cada tema definido.

6. **Preparação da Visualização:** Aplica PCA para reduzir a dimensionalidade dos embeddings para 3 dimensões, preparando os dados para a visualização 3D.

7. **Visualização 3D (Pygame/OpenGL):**  Utiliza Pygame e OpenGL para criar uma visualização 3D interativa dos clusters.  O usuário pode interagir com a visualização usando o mouse para rotacionar e controlar o zoom.

8. **Análise de Resultados:**  Após o processamento, realiza uma análise estatística dos resultados, incluindo o número total de palavras, a distribuição de palavras por cluster e a distribuição por tema.  As estatísticas são salvas em um arquivo JSON.

## Funcionalidades Principais

- **Geração de Palavras com BERT:** Expansão significativa do dataset de palavras usando o poder do BERT para capturar relações semânticas.
- **Clusterização com FAISS:** Clusterização eficiente de um grande número de palavras.
- **Visualização 3D Interativa:** Permite uma exploração detalhada dos clusters gerados.
- **Análise Estatística:** Fornece insights sobre a distribuição das palavras nos clusters.

## Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **Sentence Transformers:** Biblioteca para geração de embeddings.
- **FAISS:** Biblioteca para busca de similaridade e clustering.
- **Scikit-learn:** Biblioteca para pré-processamento de dados e algoritmos de machine learning.
- **Rich:** Biblioteca para saída de console aprimorada.
- **Transformers:** Biblioteca para modelos de linguagem.
- **NumPy:** Biblioteca para computação numérica.
- **Pandas:** Biblioteca para manipulação de dados.
- **Pygame:** Biblioteca para desenvolvimento de jogos, usada para a interface gráfica.
- **OpenGL:** Biblioteca para gráficos 3D.

## Exemplo de Uso

```python
visualizador = ClusterizadorTIBert()
visualizador.executar()
visualizador.analisar_resultados()
```

## Considerações Finais

Este sistema demonstra o poder da combinação de modelos de linguagem (BERT), algoritmos de clusterização (FAISS) e visualização 3D (Pygame/OpenGL) para explorar e entender grandes conjuntos de dados textuais em domínios específicos, como TI/DevOps.  A interatividade da visualização 3D torna a análise dos resultados muito mais intuitiva e eficiente.


---

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**
