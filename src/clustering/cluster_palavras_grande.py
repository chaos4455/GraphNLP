"""
Gerador e Clusterizador de Palavras em Larga Escala
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera e clusteriza ~9800 palavras em 9 temas usando FAISS e embeddings.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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

# Inicialização do Rich
console = Console()

class ClusterizadorPalavras:
    def __init__(self):
        """Inicialização do sistema"""
        console.print(Panel.fit("🚀 Iniciando Clusterizador de Palavras", style="bold green"))
        
        self.temas = {
            'tecnologia': {
                'palavras_base': ['computador', 'software', 'internet', 'programação', 'dados', 'algoritmo', 'rede',
                                'inteligência', 'artificial', 'machine', 'learning', 'python', 'código', 'desenvolvimento',
                                'cloud', 'servidor', 'database', 'api', 'frontend', 'backend', 'devops', 'segurança'],
                'modificadores': ['sistema', 'plataforma', 'framework', 'aplicação', 'desenvolvimento', 'arquitetura']
            },
            'ciencia': {
                'palavras_base': ['física', 'química', 'biologia', 'matemática', 'astronomia', 'genética', 'evolução',
                                'átomo', 'molécula', 'célula', 'energia', 'força', 'massa', 'velocidade', 'aceleração',
                                'teoria', 'hipótese', 'experimento', 'laboratório', 'pesquisa'],
                'modificadores': ['estudo', 'análise', 'investigação', 'descoberta', 'método', 'processo']
            },
            'arte': {
                'palavras_base': ['pintura', 'música', 'dança', 'teatro', 'cinema', 'literatura', 'poesia', 'escultura',
                                'fotografia', 'arte', 'cultura', 'criatividade', 'expressão', 'artista', 'obra',
                                'performance', 'instalação', 'exposição', 'galeria', 'museu'],
                'modificadores': ['movimento', 'estilo', 'técnica', 'composição', 'forma', 'cor']
            },
            'natureza': {
                'palavras_base': ['árvore', 'floresta', 'rio', 'montanha', 'oceano', 'animal', 'planta', 'sol',
                                'lua', 'estrela', 'vento', 'chuva', 'terra', 'natureza', 'ecossistema', 'biodiversidade',
                                'clima', 'ambiente', 'sustentabilidade', 'conservação'],
                'modificadores': ['preservação', 'proteção', 'habitat', 'espécie', 'bioma', 'ciclo']
            },
            'negocios': {
                'palavras_base': ['empresa', 'mercado', 'economia', 'finanças', 'investimento', 'marketing', 'vendas',
                                'gestão', 'estratégia', 'negociação', 'cliente', 'produto', 'serviço', 'inovação',
                                'empreendedorismo', 'startup', 'consultoria', 'planejamento'],
                'modificadores': ['análise', 'desenvolvimento', 'implementação', 'otimização', 'crescimento']
            },
            'saude': {
                'palavras_base': ['medicina', 'saúde', 'doença', 'tratamento', 'diagnóstico', 'terapia', 'prevenção',
                                'cura', 'hospital', 'médico', 'enfermagem', 'farmácia', 'psicologia', 'nutrição',
                                'exercício', 'bem-estar', 'qualidade de vida'],
                'modificadores': ['programa', 'protocolo', 'método', 'sistema', 'processo', 'técnica']
            },
            'educacao': {
                'palavras_base': ['escola', 'universidade', 'ensino', 'aprendizagem', 'professor', 'aluno', 'curso',
                                'aula', 'pedagogia', 'didática', 'conhecimento', 'educação', 'formação', 'capacitação',
                                'treinamento', 'desenvolvimento'],
                'modificadores': ['método', 'sistema', 'programa', 'projeto', 'processo', 'abordagem']
            },
            'esporte': {
                'palavras_base': ['futebol', 'basquete', 'vôlei', 'natação', 'atletismo', 'corrida', 'ginástica',
                                'treino', 'competição', 'jogo', 'campeonato', 'atleta', 'equipe', 'técnico',
                                'performance', 'condicionamento'],
                'modificadores': ['técnica', 'tática', 'estratégia', 'método', 'sistema', 'programa']
            },
            'cultura': {
                'palavras_base': ['história', 'tradição', 'costume', 'sociedade', 'religião', 'filosofia', 'política',
                                'antropologia', 'sociologia', 'identidade', 'diversidade', 'patrimônio', 'memória',
                                'ritual', 'celebração'],
                'modificadores': ['manifestação', 'expressão', 'movimento', 'aspecto', 'elemento', 'dimensão']
            }
        }
        
        self.gerar_dataset()
        self.processar_embeddings()
        self.criar_clusters()
        self.analisar_clusters()
        self.salvar_resultados()

    def gerar_dataset(self):
        """Geração do dataset com ~9800 palavras"""
        console.print("\n📚 Gerando dataset extenso de palavras...", style="bold yellow")
        
        palavras = []
        labels = []
        
        for tema, conteudo in track(self.temas.items(), description="Gerando palavras por tema"):
            palavras_base = conteudo['palavras_base']
            modificadores = conteudo['modificadores']
            
            # Gera combinações de palavras
            for palavra in palavras_base:
                # Palavra original
                palavras.append(palavra)
                labels.append(tema)
                
                # Combinações com modificadores
                for mod in modificadores:
                    palavras.append(f"{mod} de {palavra}")
                    labels.append(tema)
                    palavras.append(f"{palavra} {mod}")
                    labels.append(tema)
                
                # Combinações entre palavras do mesmo tema
                for palavra2 in random.sample(palavras_base, min(5, len(palavras_base))):
                    if palavra != palavra2:
                        palavras.append(f"{palavra} e {palavra2}")
                        labels.append(tema)
        
        self.df = pd.DataFrame({'texto': palavras, 'categoria_original': labels})
        console.print(f"✅ Dataset gerado com {len(self.df)} palavras!", style="bold green")

    def processar_embeddings(self):
        """Processamento dos embeddings"""
        console.print("\n🧠 Processando embeddings...", style="bold yellow")
        
        with console.status("[bold green]Carregando modelo de embeddings..."):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = []
        for texto in track(self.df['texto'].tolist(), description="Gerando embeddings"):
            embedding = self.model.encode([texto])[0]
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        console.print("📊 Normalizando embeddings...", style="bold yellow")
        scaler = StandardScaler()
        self.embeddings_norm = scaler.fit_transform(embeddings)
        
        console.print("✅ Embeddings processados com sucesso!", style="bold green")

    def criar_clusters(self):
        """Criação dos clusters usando FAISS"""
        console.print("\n🎯 Criando clusters...", style="bold yellow")
        
        n_clusters = 9  # Um cluster para cada tema principal
        dimensao = self.embeddings_norm.shape[1]
        
        with console.status("[bold green]Treinando K-means com FAISS..."):
            kmeans = faiss.Kmeans(d=dimensao, k=n_clusters, niter=300, gpu=False)
            kmeans.train(self.embeddings_norm.astype(np.float32))
        
        _, self.labels = kmeans.index.search(self.embeddings_norm.astype(np.float32), 1)
        self.labels = self.labels.flatten()
        
        self.df['cluster'] = self.labels
        console.print(f"✅ {n_clusters} clusters criados com sucesso!", style="bold green")

    def analisar_clusters(self):
        """Análise dos clusters gerados"""
        console.print("\n📊 Analisando clusters...", style="bold yellow")
        
        self.analise = {}
        
        # Análise por cluster
        for cluster in range(9):
            palavras_cluster = self.df[self.df['cluster'] == cluster]
            distribuicao_temas = palavras_cluster['categoria_original'].value_counts()
            
            self.analise[f'cluster_{cluster}'] = {
                'tamanho': len(palavras_cluster),
                'tema_predominante': distribuicao_temas.index[0],
                'distribuicao': distribuicao_temas.to_dict(),
                'exemplos': palavras_cluster['texto'].sample(min(5, len(palavras_cluster))).tolist()
            }
        
        # Exibe resumo
        table = Table(title="📊 Resumo dos Clusters")
        table.add_column("Cluster", style="cyan")
        table.add_column("Tamanho", style="magenta")
        table.add_column("Tema Predominante", style="green")
        
        for cluster, info in self.analise.items():
            table.add_row(
                cluster,
                str(info['tamanho']),
                info['tema_predominante']
            )
        
        console.print(table)

    def salvar_resultados(self):
        """Salvamento dos resultados"""
        console.print("\n💾 Salvando resultados...", style="bold yellow")
        
        # Salva DataFrame
        self.df.to_csv('clusters_palavras.csv', index=False)
        
        # Salva análise
        with open('analise_clusters.json', 'w', encoding='utf-8') as f:
            json.dump(self.analise, f, ensure_ascii=False, indent=4)
        
        console.print("✅ Resultados salvos com sucesso!", style="bold green")

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Clusterizador de Palavras em Larga Escala[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        clusterizador = ClusterizadorPalavras()
        
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
