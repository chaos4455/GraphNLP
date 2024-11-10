"""
Clusterizador Avançado de Tecnologias com FAISS V3
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 3.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Clusterização hierárquica com micro-clusters bem definidos.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import plotly.io as pio
from sklearn.manifold import TSNE
import os
import time
from scipy.spatial.distance import pdist, squareform
import torch
from transformers import AutoTokenizer, AutoModel

# Importa a classe base
try:
    from cluster_palavras_tech_faiss import ClusterizadorTechFaiss
except ImportError:
    console.print("[bold red]❌ Erro: Arquivo cluster_palavras_tech_faiss.py não encontrado![/]")
    console.print("[yellow]ℹ️ Certifique-se de que o arquivo está no mesmo diretório.[/]")
    raise

# Inicialização do Rich
console = Console()

class ClusterizadorTechFaissV3(ClusterizadorTechFaiss):
    def __init__(self):
        super().__init__()
        # Configurações de clustering refinadas
        self.macro_clusters = 8  # Grandes grupos (ex: Frontend, Backend, DevOps, etc)
        self.meso_clusters = 4   # Grupos médios por macro (ex: React, Vue, Angular em Frontend)
        self.micro_clusters = 3  # Subgrupos por meso (ex: React Native, Next.js, Gatsby em React)
        
        # Parâmetros de separação
        self.min_distance = 0.2  # Distância mínima entre clusters
        self.separation_factor = 1.5  # Fator de separação entre níveis
        
    def criar_clusters_hierarquicos_v3(self):
        """Cria hierarquia de clusters com separação clara"""
        console.print("\n🎯 Criando hierarquia de clusters...", style="bold yellow")
        
        dimension = self.embeddings_norm.shape[1]
        n_samples = len(self.embeddings_norm)
        
        # Ajusta número de clusters baseado no tamanho do dataset
        self.macro_clusters = min(self.macro_clusters, n_samples // 20)
        self.meso_clusters = min(self.meso_clusters, n_samples // 50)
        self.micro_clusters = min(self.micro_clusters, n_samples // 100)
        
        console.print(f"📊 Configuração de clusters:", style="bold blue")
        console.print(f"  • Macro clusters: {self.macro_clusters}")
        console.print(f"  • Meso clusters por macro: {self.meso_clusters}")
        console.print(f"  • Micro clusters por meso: {self.micro_clusters}")
        
        # Nível 1: Macro Clusters
        kmeans_macro = faiss.Kmeans(
            d=dimension,
            k=self.macro_clusters,
            niter=300,
            nredo=20,
            verbose=True,
            gpu=False,
            spherical=True,
            max_points_per_centroid=1000,
            min_points_per_centroid=2  # Reduzido para permitir clusters menores
        )
        
        kmeans_macro.train(self.embeddings_norm.astype(np.float32))
        D_macro, I_macro = kmeans_macro.index.search(self.embeddings_norm.astype(np.float32), 1)
        
        # Arrays para armazenar todos os níveis de labels
        self.macro_labels = I_macro.flatten()
        self.meso_labels = np.zeros(n_samples, dtype=np.int32)
        self.micro_labels = np.zeros(n_samples, dtype=np.int32)
        
        # Para cada macro cluster
        for i in range(self.macro_clusters):
            mask_macro = (self.macro_labels == i)
            data_macro = self.embeddings_norm[mask_macro]
            n_macro_samples = len(data_macro)
            
            if n_macro_samples >= self.meso_clusters * 2:  # Mínimo de 2 pontos por meso cluster
                # Nível 2: Meso Clusters
                n_meso = min(self.meso_clusters, n_macro_samples // 2)
                kmeans_meso = faiss.Kmeans(
                    d=dimension,
                    k=n_meso,
                    niter=200,
                    nredo=10,
                    verbose=False,
                    gpu=False,
                    spherical=True,
                    min_points_per_centroid=2
                )
                
                kmeans_meso.train(data_macro.astype(np.float32))
                D_meso, I_meso = kmeans_meso.index.search(data_macro.astype(np.float32), 1)
                
                # Atribui labels meso
                self.meso_labels[mask_macro] = I_meso.flatten() + i * self.meso_clusters
                
                # Para cada meso cluster
                for j in range(n_meso):
                    mask_meso = np.zeros(n_samples, dtype=bool)
                    mask_meso[mask_macro] = (I_meso.flatten() == j)
                    data_meso = self.embeddings_norm[mask_meso]
                    n_meso_samples = len(data_meso)
                    
                    if n_meso_samples >= self.micro_clusters * 2:  # Mínimo de 2 pontos por micro cluster
                        # Nível 3: Micro Clusters
                        n_micro = min(self.micro_clusters, n_meso_samples // 2)
                        kmeans_micro = faiss.Kmeans(
                            d=dimension,
                            k=n_micro,
                            niter=100,
                            nredo=5,
                            verbose=False,
                            gpu=False,
                            spherical=True,
                            min_points_per_centroid=2
                        )
                        
                        kmeans_micro.train(data_meso.astype(np.float32))
                        D_micro, I_micro = kmeans_micro.index.search(data_meso.astype(np.float32), 1)
                        
                        # Atribui labels micro
                        self.micro_labels[mask_meso] = I_micro.flatten() + \
                                                     (i * self.meso_clusters + j) * self.micro_clusters
            else:
                # Se houver poucos pontos, mantém tudo no mesmo cluster
                self.meso_labels[mask_macro] = i * self.meso_clusters
                self.micro_labels[mask_macro] = i * self.meso_clusters * self.micro_clusters
        
        console.print("\n✅ Clusters hierárquicos criados!", style="bold green")
        
        # Análise dos clusters
        macro_sizes = np.bincount(self.macro_labels)
        meso_sizes = np.bincount(self.meso_labels[self.meso_labels > 0])
        micro_sizes = np.bincount(self.micro_labels[self.micro_labels > 0])
        
        console.print("\n📊 Estatísticas dos clusters:", style="bold blue")
        console.print(f"  Macro clusters:")
        console.print(f"    • Média: {np.mean(macro_sizes):.1f} pontos")
        console.print(f"    • Min: {np.min(macro_sizes)} pontos")
        console.print(f"    • Max: {np.max(macro_sizes)} pontos")
        
        if len(meso_sizes) > 0:
            console.print(f"  Meso clusters:")
            console.print(f"    • Média: {np.mean(meso_sizes):.1f} pontos")
            console.print(f"    • Min: {np.min(meso_sizes)} pontos")
            console.print(f"    • Max: {np.max(meso_sizes)} pontos")
        
        if len(micro_sizes) > 0:
            console.print(f"  Micro clusters:")
            console.print(f"    • Média: {np.mean(micro_sizes):.1f} pontos")
            console.print(f"    • Min: {np.min(micro_sizes)} pontos")
            console.print(f"    • Max: {np.max(micro_sizes)} pontos")

    def visualizar_clusters_3d_v3(self):
        """Visualização 3D com clusters bem separados"""
        console.print("\n🎨 Gerando visualização 3D hierárquica...", style="bold yellow")
        
        # Redução para 3D usando t-SNE com perplexidade ajustada
        tsne = TSNE(
            n_components=3,
            perplexity=30,
            early_exaggeration=20,
            learning_rate=200,
            n_iter=2000,
            random_state=42
        )
        
        coords_3d = tsne.fit_transform(self.embeddings_norm)
        
        # Aplica transformação para separar clusters
        for i in range(self.macro_clusters):
            mask_macro = (self.macro_labels == i)
            center = np.mean(coords_3d[mask_macro], axis=0)
            
            # Separa macro clusters
            coords_3d[mask_macro] = (coords_3d[mask_macro] - center) * self.separation_factor + center
            
            # Separa meso clusters
            for j in range(self.meso_clusters):
                mask_meso = mask_macro & (self.meso_labels == i * self.meso_clusters + j)
                if np.any(mask_meso):
                    center_meso = np.mean(coords_3d[mask_meso], axis=0)
                    coords_3d[mask_meso] = (coords_3d[mask_meso] - center_meso) * \
                                         (self.separation_factor * 0.5) + center_meso
        
        # Gera visualização
        fig = go.Figure(data=[
            go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.micro_labels,
                    colorscale='Viridis',
                    opacity=0.8,
                    symbol='circle'
                ),
                text=self.df['technology'],
                hoverinfo='text',
                name='Tecnologias'
            )
        ])
        
        # Adiciona centroides dos macro clusters
        for i in range(self.macro_clusters):
            mask = (self.macro_labels == i)
            if np.any(mask):
                center = np.mean(coords_3d[mask], axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond'
                    ),
                    name=f'Centro Macro {i}'
                ))
        
        # Layout
        fig.update_layout(
            title="Clusters Hierárquicos de Tecnologias em 3D",
            width=1920,
            height=1080,
            scene=dict(
                xaxis_title="Dimensão X",
                yaxis_title="Dimensão Y",
                zaxis_title="Dimensão Z",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True
        )
        
        # Salva visualização
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        filename = f"tech_clusters_3d_v3_{timestamp}_{hash_id}.html"
        png_filename = f"tech_clusters_3d_v3_{timestamp}_{hash_id}.png"
        
        fig.write_html(filename)
        pio.write_image(fig, png_filename)
        
        return filename, png_filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Clusterizador de Tecnologias com FAISS V3[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Pipeline completo
        clusterer = ClusterizadorTechFaissV3()
        clusterer.gerar_dataset_tech()
        clusterer.processar_embeddings()
        clusterer.criar_clusters_hierarquicos_v3()
        html_file, png_file = clusterer.visualizar_clusters_3d_v3()
        
        tempo_total = time.time() - start_time
        console.print(f"\n⏱️ Tempo total de execução: {tempo_total:.2f} segundos", style="bold blue")
        console.print(f"\n🖼️ Visualizações salvas:", style="bold green")
        console.print(f"  • HTML: {html_file}")
        console.print(f"  • PNG: {png_file}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise
