"""
Visualizador 3D de Clusters de Palavras em Larga Escala
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera, clusteriza e visualiza ~9800 palavras em 9 temas usando FAISS e PyGame 3D.
"""

# Imports anteriores
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

# Imports para visualização
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.decomposition import PCA

# Inicialização
console = Console()

class ClusterizadorVisualizador:
    def __init__(self):
        """Inicialização do sistema"""
        console.print(Panel.fit("🚀 Iniciando Clusterizador e Visualizador 3D", style="bold green"))
        
        # Configurações de visualização
        self.display_size = (1200, 800)
        self.camera_config = {
            'rotation_x': 0.0,
            'rotation_y': 0.0,
            'rotation_z': 0.0,
            'scale': 1.0,
            'translate_z': -30.0
        }
        
        # Cores para os clusters (9 cores distintas)
        self.cores_clusters = [
            (1.0, 0.0, 0.0),  # Vermelho
            (0.0, 1.0, 0.0),  # Verde
            (0.0, 0.0, 1.0),  # Azul
            (1.0, 1.0, 0.0),  # Amarelo
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Ciano
            (1.0, 0.5, 0.0),  # Laranja
            (0.5, 0.0, 1.0),  # Roxo
            (0.0, 1.0, 0.5)   # Verde-água
        ]
        
        # Inicialização dos dados
        self.processar_dados()
        self.init_visualizacao()

    def processar_dados(self):
        """Pipeline de processamento de dados"""
        self.gerar_dataset()
        self.processar_embeddings()
        self.criar_clusters()
        self.preparar_visualizacao()
        self.analisar_clusters()
        self.salvar_resultados()

    def gerar_dataset(self):
        """Geração do dataset com ~9800 palavras"""
        console.print("\n📚 Gerando dataset extenso de palavras...", style="bold yellow")
        
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

    def preparar_visualizacao(self):
        """Prepara os dados para visualização 3D"""
        console.print("\n🎨 Preparando dados para visualização 3D...", style="bold yellow")
        
        # Reduz dimensionalidade para 3D usando PCA
        pca = PCA(n_components=3)
        self.coords_3d = pca.fit_transform(self.embeddings_norm)
        
        # Normaliza coordenadas para melhor visualização
        self.coords_3d = (self.coords_3d - self.coords_3d.mean()) / self.coords_3d.std() * 5
        
        console.print("✅ Dados preparados para visualização!", style="bold green")

    def init_visualizacao(self):
        """Inicializa PyGame e OpenGL"""
        console.print("\n🎮 Iniciando sistema de visualização...", style="bold yellow")
        
        pygame.init()
        pygame.display.set_caption("Visualizador 3D de Clusters - Elias Andrade")
        
        # Configuração do display
        flags = DOUBLEBUF | OPENGL | HWSURFACE
        self.screen = pygame.display.set_mode(self.display_size, flags)
        
        # Configuração OpenGL
        self.setup_opengl()
        
        console.print("✅ Sistema de visualização iniciado!", style="bold green")

    def setup_opengl(self):
        """Configuração do OpenGL"""
        # Configuração básica
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Configuração da luz
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Configuração da perspectiva
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display_size[0]/self.display_size[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def renderizar(self):
        """Renderiza a cena"""
        # Limpa buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Aplica transformações da câmera
        glTranslatef(0, 0, self.camera_config['translate_z'])
        glRotatef(self.camera_config['rotation_x'], 1, 0, 0)
        glRotatef(self.camera_config['rotation_y'], 0, 1, 0)
        glRotatef(self.camera_config['rotation_z'], 0, 0, 1)
        glScalef(self.camera_config['scale'], self.camera_config['scale'], self.camera_config['scale'])
        
        # Renderiza pontos
        glPointSize(3.0)
        glBegin(GL_POINTS)
        for i, (x, y, z) in enumerate(self.coords_3d):
            cluster = self.labels[i]
            glColor3fv(self.cores_clusters[cluster])
            glVertex3f(x, y, z)
        glEnd()
        
        # Renderiza conexões (opcional, para pontos próximos do mesmo cluster)
        if self.camera_config['scale'] > 1.5:  # Só mostra conexões quando zoom in
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for i, (x1, y1, z1) in enumerate(self.coords_3d):
                cluster1 = self.labels[i]
                for j in range(i+1, min(i+10, len(self.coords_3d))):
                    if self.labels[j] == cluster1:
                        x2, y2, z2 = self.coords_3d[j]
                        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        if dist < 2.0:
                            glColor4f(*self.cores_clusters[cluster1], 0.3)
                            glVertex3f(x1, y1, z1)
                            glVertex3f(x2, y2, z2)
            glEnd()
        
        pygame.display.flip()

    def processar_eventos(self):
        """Processa eventos do PyGame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Rotação com mouse
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    self.camera_config['rotation_y'] += event.rel[0] * 0.5
                    self.camera_config['rotation_x'] += event.rel[1] * 0.5
            
            # Zoom com roda do mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Zoom in
                    self.camera_config['scale'] *= 1.1
                elif event.button == 5:  # Zoom out
                    self.camera_config['scale'] /= 1.1
            
            # Reset de visualização com ESPAÇO
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.camera_config = {
                        'rotation_x': 0.0,
                        'rotation_y': 0.0,
                        'rotation_z': 0.0,
                        'scale': 1.0,
                        'translate_z': -30.0
                    }
        
        return True

    def executar(self):
        """Loop principal"""
        console.print("\n🎮 Iniciando visualização 3D...", style="bold green")
        running = True
        
        while running:
            running = self.processar_eventos()
            self.renderizar()
            pygame.time.wait(10)
        
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador 3D de Clusters de Palavras[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title=" Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        visualizador = ClusterizadorVisualizador()
        
        tempo_processamento = time.time() - start_time
        console.print(f"\n⏱️ Tempo de processamento: {tempo_processamento:.2f} segundos", style="bold blue")
        
        visualizador.executar()
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise

"""
Assinatura: Elias Andrade
Arquiteto de Soluções
Replika AI - Maringá, PR
"""
