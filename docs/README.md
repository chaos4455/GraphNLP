# Projeto de Visualização 3D de Clusters de Palavras - Documentação Completa

**Elias Andrade - Arquiteto de Soluções Replika AI Solutions Maringá - PR - 06/11/2024**

Este documento fornece uma visão geral do projeto e a documentação completa para todos os scripts.

## Objetivo do Projeto

O objetivo principal deste projeto é explorar e demonstrar diferentes técnicas de visualização 3D para análise de clusters de palavras, utilizando processamento de linguagem natural (PLN) e aprendizado de máquina.  A ideia central é transformar dados textuais complexos em representações visuais intuitivas e interativas, facilitando a compreensão das relações semânticas entre palavras e a identificação de padrões.

## Scripts e sua Contribuição

O projeto é composto por diversos scripts Python, cada um contribuindo para um aspecto específico da visualização e análise de dados:

* **`cluster_palavras_tech_faiss_v3.py`**: Este script utiliza o algoritmo K-means com o FAISS (Facebook AI Similarity Search) para agrupar palavras, focando em tecnologias.  Ele gera uma visualização estática 3D usando Plotly, mostrando a hierarquia de clusters (macro, meso, micro).  Este script é crucial para a análise de grandes conjuntos de dados de termos tecnológicos.  [Link para documentação](docs/scripts/cluster_palavras_tech_faiss_v3.md)

* **`cluster_3d_hierarquico.py`**: Este script cria uma visualização 3D interativa de clusters hierárquicos usando Pygame e OpenGL.  Ele gera pontos aleatórios e os agrupa em três níveis (macro, meso, micro), permitindo a exploração da estrutura hierárquica dos dados.  Este script é ideal para demonstração e exploração interativa da hierarquia de clusters. [Link para documentação](docs/scripts/cluster_3d_hierarquico.md)

* **`bert_orbital_words_3d.py`**: Este script utiliza o modelo BERT para gerar embeddings de palavras e um motor de física (Pymunk) para criar uma visualização orbital 3D interativa.  As palavras, representadas como pontos, orbitam em torno de seus clusters, simulando uma interação gravitacional.  Este script oferece uma visualização dinâmica e envolvente das relações entre as palavras. [Link para documentação](docs/scripts/bert_orbital_words_3d.md)

* **Outros Scripts:**  O projeto inclui diversos outros scripts (listados abaixo) que provavelmente realizam tarefas de pré-processamento de dados, geração de mapas de calor, ou outras análises relacionadas à visualização de clusters.  A documentação para esses scripts será adicionada em futuras atualizações.

## Lista Completa de Arquivos

Esta lista inclui todos os arquivos do projeto:

**Pasta raiz:**

* `analise_clusters.json`: Dados de análise de clusters.
* `bert_orbital_clusters_3d.py`: Script para visualização orbital de clusters em 3D usando BERT.
* `bert_orbital_words_3d.py`: Visualizador orbital 3D de palavras usando BERT, Pymunk, Pygame e OpenGL.
* `bert_tokens_heatmap.py`: Script para gerar um mapa de calor dos tokens BERT.
* `cluster_3d_hierarquico.py`: Visualizador 3D interativo de clusters hierárquicos usando Pygame e OpenGL.
* `cluster_kmeans_3d_realtime.py`: Script para clustering K-means em 3D em tempo real.
* `cluster_palavras_3d.py`: Script para clustering de palavras em 3D.
* `cluster_palavras_grande.py`: Script para clustering de um grande conjunto de palavras.
* `cluster_palavras_heatmap.py`: Script para gerar um mapa de calor de clusters de palavras.
* `cluster_palavras_mesh3d.py`: Script para gerar uma malha 3D de clusters de palavras.
* `cluster_palavras_tech_faiss_v2.py`: Versão anterior do clusterizador de tecnologias com FAISS.
* `cluster_palavras_tech_faiss_v3.py`: Clusterizador hierárquico de tecnologias usando FAISS e visualização estática 3D com Plotly.
* `cluster_palavras_ti_bert.py`: Script para clustering de palavras de TI usando BERT.
* `cluster_palavras_visualizador.py`: Script para visualização de clusters de palavras.
* `cluster_words_3d_cloud.py`: Script para gerar uma nuvem de pontos 3D de clusters de palavras.
* `clusters_palavras.csv`: Dados CSV de clusters de palavras.
* `cosine_similarity_20241109_213828_cf2ce935.png`: Imagem de similaridade cosseno.
* `dataset_ti_bert.csv`: Dataset para processamento com BERT.
* `dendrograma_20241109_213851_ddc2af3b.png`: Imagem de dendrograma.
* `estatisticas_clusters.json`: Estatísticas dos clusters.
* `gerador_palavras_gestao_heatmap.py`: Script para gerar um mapa de calor de palavras de gestão.
* `heatmap_clusters_20241109_204428_e8d56ca0.png`: Imagem de mapa de calor de clusters.
* `likelihood_distribution_20241109_213850_b130d1dc.png`: Imagem de distribuição de probabilidade.
* `mapa_calor_animado.py`: Script para gerar um mapa de calor animado.
* `palavras_gestao_20241109_220859.txt`: Arquivo de texto com palavras de gestão.
* `readme.md`: Arquivo README principal (este arquivo).
* `requirements.txt`: Arquivo com as dependências do projeto.
* `tech_clusters_3d_20241109_213200_25da37e9.html`: Arquivo HTML de visualização 3D.
* `tech_clusters_3d_20241109_213200_25da37e9.png`: Imagem PNG de visualização 3D.
* `tech_clusters_3d_20241109_213903_b6112b7e.html`: Arquivo HTML de visualização 3D.
* `tech_clusters_3d_20241109_213903_b6112b7e.png`: Imagem PNG de visualização 3D.
* `tech_clusters_3d_v3_20241109_214830_8feeb8d0.html`: Arquivo HTML de visualização 3D.
* `tech_clusters_3d_v3_20241109_214830_8feeb8d0.png`: Imagem PNG de visualização 3D.
* `teste-3d-vetor-paper-1.py`: Script de teste para vetores 3D.
* `transformer_3d_20241109_215043_afbaf875.html`: Arquivo HTML de visualização 3D.
* `transformer_3d_20241109_215043_afbaf875.png`: Imagem PNG de visualização 3D.
* `visualizador_perceptron_3d.py`: Script para visualização de um perceptron em 3D.
* `visualizador_transformer_3d.py`: Script para visualização de um modelo Transformer em 3D.
* `word_generation_heatmap.py`: Script para gerar um mapa de calor de geração de palavras.
* `mesh3d_views_20241109_211451_f31ae098`: Pasta com imagens 3D.
* `word gen`: Pasta com scripts de geração de palavras.

**Pasta `docs`:**

* `README.md`: Este arquivo.
* `scripts`: Pasta com arquivos Markdown de documentação para cada script Python.


## Próximos Passos

* Documentar os scripts Python restantes.
* Criar um sistema de versionamento mais robusto.
* Adicionar badges e shields para indicar a versão, licença, etc.
* Criar uma interface web para acesso aos visualizadores.


---
🚀 **Estado da Arte:** Este projeto busca atingir o estado da arte em visualização de dados de PLN, combinando a potência do BERT com técnicas avançadas de clustering e visualização 3D. A inspiração vem de trabalhos recentes em visualização de embeddings de palavras, buscando superar as limitações das representações 2D tradicionais. Imagine um "Matrix" de dados, mas com clusters semânticos!


---
🤔 **Considerações Filosóficas:** A organização de informações em clusters reflete a própria natureza da cognição humana, onde conceitos relacionados são agrupados na memória. Este projeto busca mimetizar esse processo, tornando a análise de grandes conjuntos de dados mais acessível e intuitiva.


---
🎮 **Cultura Pop:** A visualização 3D lembra os mundos virtuais complexos de jogos como "No Man's Sky", onde a geração procedural de conteúdo cria universos vastos e interconectados. Este projeto, embora focado em dados textuais, compartilha a ambição de revelar estruturas complexas de forma visualmente atraente.
