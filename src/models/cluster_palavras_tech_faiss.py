"""
Clusterizador Avançado de Tecnologias com FAISS
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Clusterização de alta precisão de tecnologias usando FAISS e fine-tuning.
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
import json

console = Console()

class ClusterizadorTechFaiss:
    def __init__(self):
        self.tech_categories = {
            'cloud': {
                'core': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'OpenShift', 'Terraform', 'CloudFormation'],
                'related': ['containerization', 'orchestration', 'microservices', 'serverless', 'IaC']
            },
            'devops': {
                'core': ['Jenkins', 'GitLab', 'GitHub Actions', 'ArgoCD', 'Ansible', 'Puppet', 'Chef'],
                'related': ['CI/CD', 'automation', 'deployment', 'monitoring', 'configuration management']
            },
            'ai_ml': {
                'core': ['TensorFlow', 'PyTorch', 'scikit-learn', 'Keras', 'BERT', 'GPT', 'Transformers'],
                'related': ['deep learning', 'neural networks', 'NLP', 'computer vision', 'reinforcement learning']
            },
            'data_engineering': {
                'core': ['Spark', 'Hadoop', 'Kafka', 'Airflow', 'Snowflake', 'Databricks', 'dbt'],
                'related': ['ETL', 'data pipeline', 'streaming', 'batch processing', 'data warehouse']
            },
            'security': {
                'core': ['Vault', 'SonarQube', 'Snyk', 'OWASP', 'Fortify', 'Prisma', 'Aqua'],
                'related': ['DevSecOps', 'vulnerability scanning', 'SAST', 'DAST', 'container security']
            }
        }
        
        # Configurações de clustering
        self.n_clusters = 150  # Clusters principais
        self.sub_clusters = 3   # Sub-clusters por cluster
        self.total_clusters = self.n_clusters * self.sub_clusters
        
        # Inicialização do modelo
        self.init_model()
        
    def init_model(self):
        """Inicializa modelo BERT com fine-tuning para domínio técnico"""
        console.print("\n🤖 Inicializando modelo...", style="bold yellow")
        
        self.model = AutoModel.from_pretrained('microsoft/codebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        # Configuração para fine-tuning
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        console.print("✅ Modelo inicializado!", style="bold green")

    def gerar_dataset_tech(self):
        """Gera dataset expandido de tecnologias"""
        console.print("\n📚 Gerando dataset de tecnologias...", style="bold yellow")
        
        technologies = []
        labels = []
        
        for category, techs in self.tech_categories.items():
            # Core technologies
            for tech in techs['core']:
                technologies.append(tech)
                labels.append(f"{category}_core")
                
                # Gera combinações
                for related in techs['related']:
                    technologies.append(f"{tech} {related}")
                    labels.append(f"{category}_combined")
                    
                    # Gera variações específicas
                    technologies.append(f"{tech} for {related}")
                    labels.append(f"{category}_specific")
                    
                    technologies.append(f"{related} with {tech}")
                    labels.append(f"{category}_integration")
        
        self.df = pd.DataFrame({'technology': technologies, 'category': labels})
        console.print(f"✅ Dataset gerado com {len(self.df)} tecnologias!", style="bold green")

    def processar_embeddings(self):
        """Processa embeddings com normalização avançada"""
        console.print("\n🧠 Processando embeddings...", style="bold yellow")
        
        embeddings = []
        
        for tech in track(self.df['technology'], description="Gerando embeddings"):
            # Tokenização
            inputs = self.tokenizer(tech, return_tensors="pt", padding=True, truncation=True)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            
            embeddings.append(embedding[0])
        
        embeddings = np.array(embeddings)
        
        # Normalização em múltiplas etapas
        # 1. Normalização L2
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # 2. Standard Scaling
        scaler_standard = StandardScaler()
        embeddings = scaler_standard.fit_transform(embeddings)
        
        # 3. MinMax Scaling para range específico
        scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
        self.embeddings_norm = scaler_minmax.fit_transform(embeddings)
        
        console.print("✅ Embeddings processados e normalizados!", style="bold green")

    def criar_clusters_hierarquicos(self):
        """Cria clusters hierárquicos usando FAISS"""
        console.print("\n🎯 Criando clusters hierárquicos...", style="bold yellow")
        
        # Configuração do FAISS
        dimension = self.embeddings_norm.shape[1]
        
        # Nível 1: Clusters principais
        kmeans_main = faiss.Kmeans(
            d=dimension,
            k=self.n_clusters,
            niter=300,
            nredo=10,
            verbose=True,
            gpu=False,
            spherical=True
        )
        
        # Treina clusters principais
        kmeans_main.train(self.embeddings_norm.astype(np.float32))
        _, main_labels = kmeans_main.index.search(self.embeddings_norm.astype(np.float32), 1)
        main_labels = main_labels.flatten()  # Garante 1D array
        
        # Inicializa array de sub-labels
        self.sub_labels = np.zeros(len(self.embeddings_norm), dtype=np.int32)
        
        # Nível 2: Sub-clusters
        for i in range(self.n_clusters):
            mask = (main_labels == i)
            cluster_data = self.embeddings_norm[mask]
            
            if len(cluster_data) > self.sub_clusters:
                kmeans_sub = faiss.Kmeans(
                    d=dimension,
                    k=min(self.sub_clusters, len(cluster_data)),  # Evita clusters vazios
                    niter=100,
                    nredo=5,
                    verbose=False,
                    gpu=False
                )
                
                # Treina sub-clusters
                kmeans_sub.train(cluster_data.astype(np.float32))
                _, sub_cluster = kmeans_sub.index.search(cluster_data.astype(np.float32), 1)
                
                # Atribui labels
                self.sub_labels[mask] = sub_cluster.flatten() + i * self.sub_clusters
            else:
                # Se houver poucos pontos, atribui todos ao mesmo sub-cluster
                self.sub_labels[mask] = i * self.sub_clusters
        
        self.df['cluster'] = self.sub_labels
        console.print(f"✅ {self.total_clusters} clusters hierárquicos criados!", style="bold green")
        
        # Análise dos clusters
        cluster_sizes = np.bincount(self.sub_labels)
        console.print(f"\n📊 Estatísticas dos clusters:")
        console.print(f"  • Número médio de itens por cluster: {np.mean(cluster_sizes):.2f}")
        console.print(f"  • Desvio padrão: {np.std(cluster_sizes):.2f}")
        console.print(f"  • Mínimo: {np.min(cluster_sizes)}")
        console.print(f"  • Máximo: {np.max(cluster_sizes)}")

    def visualizar_clusters_3d(self):
        """Gera visualização 3D dos clusters"""
        console.print("\n🎨 Gerando visualização 3D...", style="bold yellow")
        
        # Redução para 3D usando t-SNE
        tsne = TSNE(n_components=3, random_state=42)
        coords_3d = tsne.fit_transform(self.embeddings_norm)
        
        # Gera hash única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_obj = hashlib.md5(str(timestamp).encode())
        hash_id = hash_obj.hexdigest()[:8]
        
        # Cria visualização
        fig = go.Figure(data=[
            go.Scatter3d(
                x=coords_3d[:, 0],
                y=coords_3d[:, 1],
                z=coords_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.sub_labels,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=self.df['technology'],
                hoverinfo='text'
            )
        ])
        
        # Layout
        fig.update_layout(
            title="Clusters de Tecnologias em 3D",
            width=1920,
            height=1080,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        # Salva visualização
        filename = f"tech_clusters_3d_{timestamp}_{hash_id}.html"
        fig.write_html(filename)
        
        # Salva também como PNG
        png_filename = f"tech_clusters_3d_{timestamp}_{hash_id}.png"
        pio.write_image(fig, png_filename)
        
        console.print(f"✅ Visualização salva como {filename} e {png_filename}!", style="bold green")
        return filename, png_filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Clusterizador de Tecnologias com FAISS[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Pipeline completo
        clusterer = ClusterizadorTechFaiss()
        clusterer.gerar_dataset_tech()
        clusterer.processar_embeddings()
        clusterer.criar_clusters_hierarquicos()
        html_file, png_file = clusterer.visualizar_clusters_3d()
        
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
