# Visualizador de Clusters Hierárquicos 3D Interativo - Documentação

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**

**Versão:** 1.0.0 (Micro-revisão 000000001) - 2024-11-06

## Visão Geral

Este documento descreve o script `cluster_3d_hierarquico.py`, que implementa um visualizador 3D interativo de clusters hierárquicos.  O sistema gera pontos aleatórios em um espaço 3D, aplica o algoritmo K-Means em múltiplos níveis para criar uma estrutura hierárquica de clusters e, em seguida, renderiza esses clusters em uma visualização 3D interativa usando Pygame e OpenGL.  A visualização permite ao usuário rotacionar, aproximar e afastar a câmera para explorar a estrutura dos clusters.

A inspiração para este projeto veio da necessidade de visualizar e entender a estrutura complexa de dados multidimensionais.  A visualização 3D, combinada com a hierarquia de clusters, permite uma compreensão mais intuitiva das relações entre os dados.

## Arquitetura

O sistema é composto pelas seguintes partes principais:

1. **Geração de Pontos:** Gera pontos aleatórios em um espaço 3D, simulando dados multidimensionais.

2. **Clustering Hierárquico:** Aplica o algoritmo K-Means em três níveis:
    - **Macro Clusters:**  Agrupa os pontos em um número definido de clusters principais.
    - **Meso Clusters:**  Para cada macro cluster, cria sub-clusters.
    - **Micro Clusters:**  Para cada meso cluster, cria sub-clusters ainda menores.

3. **Visualização 3D (Pygame/OpenGL):**  Utiliza Pygame para criar a janela e lidar com eventos do usuário, e OpenGL para renderizar a cena 3D.  A visualização mostra os clusters em diferentes cores e tamanhos, representando seus níveis na hierarquia.  O usuário pode interagir com a visualização usando o mouse para rotacionar e controlar o zoom.

## Funcionalidades Principais

- **Geração de Dados Aleatórios:** Simula dados multidimensionais para demonstração.
- **Clustering Hierárquico:** Cria uma estrutura de clusters em múltiplos níveis.
- **Visualização 3D Interativa:** Permite ao usuário explorar a estrutura dos clusters.
- **Controle de Câmera:** Permite rotacionar e controlar o zoom da câmera.

## Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **Scikit-learn:** Biblioteca para o algoritmo K-Means.
- **Pygame:** Biblioteca para desenvolvimento de jogos, usada para a interface gráfica.
- **OpenGL:** Biblioteca para gráficos 3D.
- **Rich:** Biblioteca para saída de console aprimorada.

## Exemplo de Uso

O script é executado diretamente: `python cluster_3d_hierarquico.py`

## Considerações Finais

Este script demonstra uma forma eficiente de visualizar dados multidimensionais usando clustering hierárquico e uma interface 3D interativa.  A visualização permite uma compreensão mais intuitiva da estrutura dos dados, facilitando a análise e interpretação.


---

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**
