"""
Visualizador 3D de Clusters com Malhas 3D
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera visualizações 3D em malha dos clusters de palavras.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from scipy.ndimage import gaussian_filter
import plotly.io as pio
from sklearn.manifold import MDS
import os
import time
from scipy.spatial.distance import pdist, squareform

# Importa a classe base
from cluster_palavras_ti_bert import ClusterizadorTIBert

# Inicialização
console = Console()

class ClusterizadorMesh3D(ClusterizadorTIBert):
    def __init__(self):
        super().__init__()
        self.micro_clusters = 250
        self.grid_size = 50
        self.angles = [
            {'camera': dict(eye=dict(x=1.5, y=1.5, z=1.5)), 'name': 'isometric'},
            {'camera': dict(eye=dict(x=0, y=0, z=2)), 'name': 'top'},
            {'camera': dict(eye=dict(x=2, y=0, z=0)), 'name': 'side'},
            {'camera': dict(eye=dict(x=1.5, y=-1.5, z=1)), 'name': 'perspective'},
            {'camera': dict(eye=dict(x=0.5, y=2, z=0.5)), 'name': 'angular'}
        ]
        
    def gerar_micro_clusters(self):
        """Gera 250 micro-clusters de palavras similares"""
        console.print("\n🔍 Gerando micro-clusters...", style="bold yellow")
        
        # Processa embeddings se ainda não foram processados
        if not hasattr(self, 'embeddings_norm'):
            self.processar_embeddings()
        
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

    def processar_embeddings(self):
        """Processa embeddings das palavras"""
        console.print("\n🧠 Processando embeddings...", style="bold yellow")
        
        # Carrega modelo de embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Gera embeddings
        textos = self.df['texto'].tolist()
        embeddings = []
        
        for texto in track(textos, description="Gerando embeddings"):
            embedding = model.encode([texto])[0]
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Normalização
        scaler = StandardScaler()
        self.embeddings_norm = scaler.fit_transform(embeddings)
        
        console.print("✅ Embeddings processados!", style="bold green")

    def preparar_dados_3d(self):
        """Prepara dados para visualização 3D"""
        console.print("\n🔄 Preparando dados para malha 3D...", style="bold yellow")
        
        # Reduz dimensionalidade para 2D
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        pos = mds.fit_transform(self.similarity_matrix)
        
        # Cria grid
        x = np.linspace(pos[:, 0].min(), pos[:, 0].max(), self.grid_size)
        y = np.linspace(pos[:, 1].min(), pos[:, 1].max(), self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Inicializa Z com zeros
        Z = np.zeros((self.grid_size, self.grid_size))
        
        # Normaliza posições para o grid
        x_norm = ((pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min()) * (self.grid_size-1)).astype(int)
        y_norm = ((pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min()) * (self.grid_size-1)).astype(int)
        
        # Preenche Z com tamanhos dos clusters
        for i in range(len(x_norm)):
            Z[y_norm[i], x_norm[i]] += self.cluster_sizes[i]
        
        # Suaviza a superfície
        Z = gaussian_filter(Z, sigma=1)
        
        # Normaliza Z
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        self.X, self.Y, self.Z = X, Y, Z
        console.print("✅ Dados 3D preparados!", style="bold green")

    def gerar_visualizacoes_3d(self):
        """Gera visualizações 3D de diferentes ângulos"""
        console.print("\n🎨 Gerando visualizações 3D...", style="bold yellow")
        
        # Gera hash e timestamp únicos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_obj = hashlib.md5(str(timestamp).encode())
        hash_id = hash_obj.hexdigest()[:8]
        
        # Cria diretório para as imagens
        dir_name = f"mesh3d_views_{timestamp}_{hash_id}"
        os.makedirs(dir_name, exist_ok=True)
        
        filenames = []
        
        # Gera visualização para cada ângulo
        for angle in track(self.angles, description="Gerando visualizações"):
            fig = go.Figure(data=[
                go.Surface(
                    x=self.X,
                    y=self.Y,
                    z=self.Z,
                    colorscale=[
                        [0, 'purple'],      # Baixa densidade
                        [0.5, 'lightblue'], # Média densidade
                        [1, 'green']        # Alta densidade
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Densidade de Palavras",
                        titleside="right"
                    )
                )
            ])
            
            # Configuração do layout
            fig.update_layout(
                title={
                    'text': f'Visualização 3D de Clusters - Vista {angle["name"].title()}',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                scene = dict(
                    camera=angle['camera'],
                    xaxis_title="Dimensão X",
                    yaxis_title="Dimensão Y",
                    zaxis_title="Densidade",
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                width=1920,
                height=1080,
                annotations=[
                    dict(
                        text=f"ID: {hash_id} | Gerado em: {timestamp}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.15,
                        font=dict(size=12)
                    )
                ]
            )
            
            # Salva a visualização
            filename = f"{dir_name}/mesh3d_{angle['name']}_{hash_id}.png"
            pio.write_image(fig, filename)
            filenames.append(filename)
            
        console.print(f"✅ Visualizações salvas em {dir_name}!", style="bold green")
        return filenames

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador 3D de Clusters em Malha[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Inicialização e processamento
        visualizador = ClusterizadorMesh3D()
        
        # Gera micro-clusters e visualizações
        visualizador.gerar_micro_clusters()
        visualizador.calcular_matriz_similaridade()
        visualizador.preparar_dados_3d()
        filenames = visualizador.gerar_visualizacoes_3d()
        
        tempo_total = time.time() - start_time
        console.print(f"\n⏱️ Tempo total de execução: {tempo_total:.2f} segundos", style="bold blue")
        console.print("\n🖼️ Visualizações geradas:", style="bold green")
        for filename in filenames:
            console.print(f"  • {filename}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise

"""
Assinatura: Elias Andrade
Arquiteto de Soluções
Replika AI - Maringá, PR
"""
