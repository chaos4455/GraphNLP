"""
Clusterizador Avançado de Tecnologias com FAISS V2
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 2.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Clusterização avançada com FAISS, cosine similarity e maximum likelihood.
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
import plotly.io as pio
from sklearn.manifold import TSNE
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import multivariate_normal
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel

# Importa a classe base do arquivo anterior
try:
    from cluster_palavras_tech_faiss import ClusterizadorTechFaiss
except ImportError:
    console.print("[bold red]❌ Erro: Arquivo cluster_palavras_tech_faiss.py não encontrado![/]")
    console.print("[yellow]ℹ️ Certifique-se de que o arquivo está no mesmo diretório.[/]")
    raise

# Inicialização do Rich
console = Console()

class ClusterizadorTechFaissV2(ClusterizadorTechFaiss):
    def __init__(self):
        super().__init__()
        self.kmeans_clusters = 50  # Clusters para K-means
        self.cosine_threshold = 0.8  # Limiar para similaridade
        self.likelihood_components = 30  # Componentes para ML
        
    def analisar_clusters_cosine(self):
        """Análise de clusters usando similaridade por cosseno"""
        console.print("\n📐 Analisando clusters por similaridade cosseno...", style="bold yellow")
        
        # Calcula matriz de similaridade
        cosine_sim = cosine_similarity(self.embeddings_norm)
        
        # Agrupa tecnologias similares
        grupos_cosine = []
        used_indices = set()
        
        for i in range(len(self.df)):
            if i in used_indices:
                continue
                
            grupo = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(self.df)):
                if j not in used_indices and cosine_sim[i][j] > self.cosine_threshold:
                    grupo.append(j)
                    used_indices.add(j)
            
            if len(grupo) > 1:
                grupos_cosine.append([self.df['technology'].iloc[idx] for idx in grupo])
        
        # Visualização da matriz de similaridade
        plt.figure(figsize=(20, 20))
        sns.heatmap(cosine_sim, cmap='viridis')
        plt.title('Matriz de Similaridade por Cosseno')
        
        # Gera hash única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        # Salva visualização
        cosine_filename = f"cosine_similarity_{timestamp}_{hash_id}.png"
        plt.savefig(cosine_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return grupos_cosine, cosine_filename

    def analisar_clusters_likelihood(self):
        """Análise de clusters usando máxima verossimilhança"""
        console.print("\n📊 Analisando clusters por máxima verossimilhança...", style="bold yellow")
        
        # Configuração do FAISS para ML clustering
        dimension = self.embeddings_norm.shape[1]
        
        # Inicializa clustering com mixture of Gaussians
        kmeans_ml = faiss.Kmeans(
            d=dimension,
            k=self.likelihood_components,
            niter=300,
            nredo=10,
            verbose=True,
            gpu=False,
            spherical=True,
            max_points_per_centroid=256,  # Controle de densidade
            min_points_per_centroid=3     # Evita clusters vazios
        )
        
        # Treina modelo
        kmeans_ml.train(self.embeddings_norm.astype(np.float32))
        
        # Calcula probabilidades
        distances, assignments = kmeans_ml.index.search(self.embeddings_norm.astype(np.float32), 1)
        
        # Calcula log-likelihood para cada cluster
        log_likelihoods = -0.5 * distances.flatten()
        
        # Visualização da distribuição de likelihood
        plt.figure(figsize=(15, 10))
        sns.histplot(log_likelihoods, bins=50, kde=True)
        plt.title('Distribuição de Log-Likelihood dos Clusters')
        plt.xlabel('Log-Likelihood')
        plt.ylabel('Frequência')
        
        # Gera hash única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        # Salva visualização
        likelihood_filename = f"likelihood_distribution_{timestamp}_{hash_id}.png"
        plt.savefig(likelihood_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return assignments.flatten(), likelihood_filename

    def gerar_dendrograma(self):
        """Gera dendrograma hierárquico dos clusters"""
        console.print("\n🌳 Gerando dendrograma...", style="bold yellow")
        
        # Calcula linkage matrix
        linkage_matrix = linkage(self.embeddings_norm, method='ward')
        
        # Configuração do plot
        plt.figure(figsize=(20, 10))
        dendrogram(
            linkage_matrix,
            labels=self.df['technology'].values,
            leaf_rotation=90,
            leaf_font_size=8
        )
        plt.title('Dendrograma Hierárquico de Tecnologias')
        
        # Gera hash única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        # Salva visualização
        dendro_filename = f"dendrograma_{timestamp}_{hash_id}.png"
        plt.savefig(dendro_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dendro_filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Clusterizador de Tecnologias com FAISS V2[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Pipeline completo
        clusterer = ClusterizadorTechFaissV2()
        clusterer.gerar_dataset_tech()
        clusterer.processar_embeddings()
        clusterer.criar_clusters_hierarquicos()
        
        # Análises adicionais
        grupos_cosine, cosine_file = clusterer.analisar_clusters_cosine()
        assignments_ml, likelihood_file = clusterer.analisar_clusters_likelihood()
        dendro_file = clusterer.gerar_dendrograma()
        
        # Visualização 3D original
        html_file, png_file = clusterer.visualizar_clusters_3d()
        
        # Relatório
        console.print("\n📑 Relatório de Análises:", style="bold green")
        console.print(f"  • Matriz de Similaridade: {cosine_file}")
        console.print(f"  • Distribuição de Likelihood: {likelihood_file}")
        console.print(f"  • Dendrograma: {dendro_file}")
        console.print(f"  • Visualização 3D: {png_file}")
        
        # Grupos por similaridade
        console.print("\n🔍 Grupos de Alta Similaridade:", style="bold yellow")
        for i, grupo in enumerate(grupos_cosine[:5], 1):
            console.print(f"\nGrupo {i}:")
            for tech in grupo:
                console.print(f"  • {tech}")
        
        tempo_total = time.time() - start_time
        console.print(f"\n⏱️ Tempo total de execução: {tempo_total:.2f} segundos", style="bold blue")
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise

"""
Assinatura: Elias Andrade
Arquiteto de Soluções
Replika AI - Maringá, PR
"""
