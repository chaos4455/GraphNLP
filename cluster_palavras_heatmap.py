"""
Visualizador 3D de Clusters com Mapa de Calor
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera clusters de palavras TI e cria mapa de calor de similaridade.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F
import faiss
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.table import Table
import random
from datetime import datetime
import time
import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import hashlib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import MDS
import plotly.io as pio
from scipy.ndimage import gaussian_filter

# Importa a classe base do arquivo anterior
from cluster_palavras_ti_bert import ClusterizadorTIBert

# Inicialização do Rich
console = Console()

class ClusterizadorHeatmap(ClusterizadorTIBert):
    def __init__(self):
        super().__init__()
        self.micro_clusters = 250  # Número de micro-clusters para o heatmap
        
    def gerar_micro_clusters(self):
        """Gera 250 micro-clusters de palavras similares"""
        console.print("\n🔍 Gerando micro-clusters para heatmap...", style="bold yellow")
        
        # Configura FAISS para micro-clustering
        kmeans_micro = faiss.Kmeans(
            d=self.embeddings_norm.shape[1],
            k=self.micro_clusters,
            niter=300,
            nredo=5,
            verbose=True,
            gpu=False
        )
        
        # Treina micro-clusters
        kmeans_micro.train(self.embeddings_norm.astype(np.float32))
        _, micro_labels = kmeans_micro.index.search(self.embeddings_norm.astype(np.float32), 1)
        
        # Obtém centroides
        self.micro_centroids = kmeans_micro.centroids
        self.micro_labels = micro_labels.flatten()
        
        # Calcula tamanho dos clusters
        self.cluster_sizes = np.bincount(self.micro_labels, minlength=self.micro_clusters)
        
        console.print("✅ Micro-clusters gerados!", style="bold green")

    def calcular_matriz_similaridade(self):
        """Calcula matriz de similaridade entre micro-clusters"""
        console.print("\n📊 Calculando matriz de similaridade...", style="bold yellow")
        
        # Calcula distâncias entre centroides
        distances = pdist(self.micro_centroids)
        self.similarity_matrix = squareform(distances)
        
        # Normaliza distâncias
        self.similarity_matrix = (self.similarity_matrix - self.similarity_matrix.min()) / \
                               (self.similarity_matrix.max() - self.similarity_matrix.min())
        
        console.print("✅ Matriz de similaridade calculada!", style="bold green")

    def gerar_heatmap(self):
        """Gera heatmap de similaridade com plotly"""
        console.print("\n🎨 Gerando heatmap...", style="bold yellow")
        
        # Reduz dimensionalidade para 2D para organização espacial
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        pos = mds.fit_transform(self.similarity_matrix)
        
        # Reorganiza clusters baseado na posição 2D
        x, y = pos[:, 0], pos[:, 1]
        grid_size = 50
        heatmap = np.zeros((grid_size, grid_size))
        
        # Normaliza posições para o grid
        x_norm = ((x - x.min()) / (x.max() - x.min()) * (grid_size-1)).astype(int)
        y_norm = ((y - y.min()) / (y.max() - y.min()) * (grid_size-1)).astype(int)
        
        # Preenche o grid com tamanhos dos clusters
        for i in range(len(x_norm)):
            heatmap[y_norm[i], x_norm[i]] += self.cluster_sizes[i]
        
        # Suaviza o heatmap
        heatmap = gaussian_filter(heatmap, sigma=1)
        
        # Gera hash única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_obj = hashlib.md5(str(timestamp).encode())
        hash_id = hash_obj.hexdigest()[:8]
        
        # Cria figura com plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            colorscale='RdBu_r',  # Vermelho no centro, azul nas bordas
            reversescale=True,
            showscale=True,
            colorbar=dict(
                title="Densidade de Palavras",
                titleside="right"
            )
        ))
        
        # Configuração do layout
        fig.update_layout(
            title={
                'text': 'Mapa de Calor de Similaridade de Palavras TI',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            width=1920,
            height=1080,
            xaxis_title="Dimensão X de Similaridade",
            yaxis_title="Dimensão Y de Similaridade",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            annotations=[
                dict(
                    text="Clusters mais densos ao centro (vermelho) | Clusters esparsos nas bordas (azul)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    font=dict(size=16)
                ),
                dict(
                    text=f"ID: {hash_id} | Gerado em: {timestamp}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,
                    font=dict(size=12)
                )
            ]
        )
        
        # Salva o heatmap
        filename = f"heatmap_clusters_{timestamp}_{hash_id}.png"
        pio.write_image(fig, filename)
        
        console.print(f"✅ Heatmap salvo como {filename}!", style="bold green")
        return filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador de Clusters com Heatmap[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Inicialização e processamento
        visualizador = ClusterizadorHeatmap()
        
        # Gera micro-clusters e heatmap
        visualizador.gerar_micro_clusters()
        visualizador.calcular_matriz_similaridade()
        heatmap_file = visualizador.gerar_heatmap()
        
        # Executa visualização 3D
        visualizador.executar()
        
        tempo_total = time.time() - start_time
        console.print(f"\n⏱️ Tempo total de execução: {tempo_total:.2f} segundos", style="bold blue")
        console.print(f"🖼️ Heatmap salvo em: {heatmap_file}", style="bold green")
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise 