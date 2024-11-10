"""
Visualizador Orbital de Palavras BERT 3D
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 2.0.0 (Micro-revisão 000000001)
Data: 2024-03-27

Sistema de visualização avançada de palavras com:
- Geração via BERT
- Física orbital
- Clusters dinâmicos
- Interação em tempo real
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import time
from rich.console import Console
from rich.panel import Panel
import threading
import queue
from datetime import datetime
import random
from dataclasses import dataclass
import pymunk
import pygame.freetype
import psutil
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import ctypes

console = Console()

@dataclass
class Palavra:
    """Estrutura de dados para palavras"""
    texto: str
    embedding: np.ndarray
    posicao: np.ndarray
    velocidade: np.ndarray
    cor: list
    cluster: int
    tema: str
    corpo: pymunk.Body  # Corpo físico
    raio: float
    massa: float
    z: float  # Nova propriedade para profundidade
    
@dataclass
class Cluster:
    """Estrutura de dados para clusters"""
    centro: np.ndarray
    raio: float
    cor: list
    palavras: list
    tema: str
    nivel: int
    massa: float
    corpo: pymunk.Body

class BertOrbitalVisualizer:
    def __init__(self):
        self.window_size = (1920, 1080)
        self.max_palavras = 55000
        self.palavras = []
        self.clusters = []
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # Gravidade 2D
        
        # Estado do programa
        self.running = True
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # Conversão 3D -> 2D
        self.z_scale = 0.1  # Fator de escala para profundidade
        
        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.model.eval()
        
        # Clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=12,
            batch_size=1000,
            random_state=42
        )
        
        # Temas e cores
        self.temas = {
            'tecnologia': {'cor': [1.0, 0.0, 0.0], 'palavras': []},
            'negócios': {'cor': [0.0, 1.0, 0.0], 'palavras': []},
            'ciência': {'cor': [0.0, 0.0, 1.0], 'palavras': []},
            'saúde': {'cor': [1.0, 1.0, 0.0], 'palavras': []},
            'educação': {'cor': [1.0, 0.0, 1.0], 'palavras': []},
            'finanças': {'cor': [0.0, 1.0, 1.0], 'palavras': []}
        }
        
        # Interface
        self.font = None
        self.info_palavra = None
        self.mouse_pos_3d = np.zeros(3)
        self.camera = {
            'distancia': -50,
            'rotacao_x': 0,
            'rotacao_y': 0,
            'zoom': 1.0
        }
        
        # Estatísticas
        self.stats = {
            'fps': 0,
            'palavras_processadas': 0,
            'clusters_ativos': 0,
            'memoria_usada': 0,
            'tempo_execucao': 0
        }
        
        # Tela
        self.screen = None
        
        # Física
        self.configurar_fisica()
        
        # VBOs para renderização acelerada
        self.vertices = np.zeros((self.max_palavras, 3), dtype=np.float32)
        self.cores = np.zeros((self.max_palavras, 4), dtype=np.float32)
        self.vertex_buffer = vbo.VBO(self.vertices)
        self.color_buffer = vbo.VBO(self.cores)
        
    def configurar_fisica(self):
        """Configura o sistema de física"""
        # Configurações do espaço físico
        self.space.damping = 0.8  # Amortecimento
        
        # Força gravitacional personalizada
        def gravidade_orbital(bodies, gravity, damping, dt):
            for i, body1 in enumerate(bodies):
                for body2 in bodies[i+1:]:
                    delta = body2.position - body1.position
                    dist = np.linalg.norm(delta)
                    if dist > 0:
                        F = (body1.mass * body2.mass * 0.1) / (dist * dist)
                        n = delta / dist
                        body1.apply_force_at_local_point(F * n)
                        body2.apply_force_at_local_point(-F * n)
                        
        self.space.add_default_collision_handler()
        self.space.gravity_func = gravidade_orbital

    def processar_bert(self, texto):
        """Processa texto através do BERT e retorna embedding"""
        with torch.no_grad():
            inputs = self.tokenizer(texto, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(1).numpy()[0]
            
    def gerar_palavra(self):
        """Gera nova palavra e processa"""
        temas = list(self.temas.keys())
        tema = random.choice(temas)
        
        # Palavras por tema
        palavras_tema = {
            'tecnologia': ['IA', 'blockchain', 'cloud', 'dados', 'algoritmo'],
            'negócios': ['estratégia', 'inovação', 'gestão', 'processo', 'resultado'],
            'ciência': ['pesquisa', 'método', 'análise', 'teoria', 'experimento'],
            'saúde': ['medicina', 'tratamento', 'prevenção', 'diagnóstico', 'terapia'],
            'educação': ['aprendizagem', 'ensino', 'conhecimento', 'didática', 'pedagogia'],
            'finanças': ['investimento', 'mercado', 'capital', 'risco', 'retorno']
        }
        
        texto = random.choice(palavras_tema[tema])
        embedding = self.processar_bert(texto)
        
        # Posição inicial aleatória (3D)
        pos_3d = np.random.normal(0, 10, 3)
        vel_3d = np.random.normal(0, 0.1, 3)
        z = pos_3d[2]  # Guarda a profundidade
        
        # Convertendo para 2D para física
        pos_2d = (float(pos_3d[0]), float(pos_3d[1]))
        vel_2d = (float(vel_3d[0]), float(vel_3d[1]))
        
        # Cria corpo físico
        massa = random.uniform(1.0, 2.0)
        momento = pymunk.moment_for_circle(massa, 0, 1.0)
        corpo = pymunk.Body(massa, momento)
        corpo.position = pos_2d  # Posição 2D
        corpo.velocity = vel_2d  # Velocidade 2D
        
        # Adiciona forma circular
        shape = pymunk.Circle(corpo, 1.0)
        shape.elasticity = 0.95
        shape.friction = 0.9
        
        # Adiciona ao espaço físico
        self.space.add(corpo, shape)
        
        return Palavra(
            texto=texto,
            embedding=embedding,
            posicao=np.array(pos_2d),  # Agora é 2D
            velocidade=np.array(vel_2d),  # Agora é 2D
            cor=self.temas[tema]['cor'],
            cluster=-1,
            tema=tema,
            corpo=corpo,
            raio=1.0,
            massa=massa,
            z=z  # Armazena Z separadamente
        )
        
    def atualizar_clusters(self):
        """Atualiza clusters baseado nos embeddings"""
        if len(self.palavras) < 100:
            return
            
        # Coleta embeddings
        embeddings = np.array([p.embedding for p in self.palavras])
        
        # Clustering
        labels = self.kmeans.fit_predict(embeddings)
        
        # Atualiza clusters
        self.clusters = []
        for i in range(self.kmeans.n_clusters):
            mask = labels == i
            if not np.any(mask):
                continue
                
            palavras_cluster = [p for p, m in zip(self.palavras, mask) if m]
            centro = np.mean([p.posicao for p in palavras_cluster], axis=0)
            
            # Determina tema dominante
            temas = [p.tema for p in palavras_cluster]
            tema = max(set(temas), key=temas.count)
            
            # Cria corpo físico para cluster
            massa = len(palavras_cluster) * 2.0
            momento = pymunk.moment_for_circle(massa, 0, 5.0)
            corpo = pymunk.Body(massa, momento)
            corpo.position = tuple(centro)
            
            self.space.add(corpo)
            
            self.clusters.append(Cluster(
                centro=centro,
                raio=len(palavras_cluster) ** 0.5,
                cor=self.temas[tema]['cor'],
                palavras=palavras_cluster,
                tema=tema,
                nivel=0,
                massa=massa,
                corpo=corpo
            ))
            
    def renderizar_info(self):
        """Renderiza informações na tela"""
        # Salva estado OpenGL
        glPushMatrix()
        glLoadIdentity()
        
        # Muda para modo 2D
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_size[0], self.window_size[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_DEPTH_TEST)
        
        # Data e hora
        agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Renderiza textos
        info_texts = [
            f"Data: {agora}",
            f"FPS: {self.stats['fps']:.1f}",
            f"Palavras: {len(self.palavras)}/{self.max_palavras}",
            f"Clusters: {len(self.clusters)}",
            f"Memória: {self.stats['memoria_usada']:.1f} MB",
            f"Tempo: {self.stats['tempo_execucao']:.1f}s"
        ]
        
        for i, text in enumerate(info_texts):
            surface, rect = self.font.render(text, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i*30))
            
        # Info da palavra sob o mouse
        if self.info_palavra:
            info = [
                f"Palavra: {self.info_palavra.texto}",
                f"Tema: {self.info_palavra.tema}",
                f"Cluster: {self.info_palavra.cluster}",
                f"Posição: {tuple(self.info_palavra.posicao)}"
            ]
            
            for i, text in enumerate(info):
                surface, rect = self.font.render(text, (255, 255, 255))
                self.screen.blit(surface, (self.window_size[0] - 300, 10 + i*30))
                
        # Restaura estado OpenGL
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
    def renderizar_cena(self):
        """Renderiza cena usando VBOs"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Posiciona câmera
        glTranslatef(0, 0, self.camera['distancia'] * self.camera['zoom'])
        glRotatef(self.camera['rotacao_x'], 1, 0, 0)
        glRotatef(self.camera['rotacao_y'], 0, 1, 0)

        # Atualiza e renderiza VBOs
        if len(self.palavras) > 0:
            # Atualiza arrays
            for i, palavra in enumerate(self.palavras):
                pos_2d = palavra.corpo.position
                self.vertices[i] = [pos_2d[0], pos_2d[1], palavra.z]
                self.cores[i] = [*palavra.cor, 0.8]

            # Renderiza pontos
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            # Atualiza e liga VBOs
            self.vertex_buffer.bind()
            self.vertex_buffer.set_array(self.vertices[:len(self.palavras)])
            glVertexPointer(3, GL_FLOAT, 0, None)

            self.color_buffer.bind()
            self.color_buffer.set_array(self.cores[:len(self.palavras)])
            glColorPointer(4, GL_FLOAT, 0, None)

            # Desenha
            glDrawArrays(GL_POINTS, 0, len(self.palavras))

            # Desliga VBOs
            self.vertex_buffer.unbind()
            self.color_buffer.unbind()

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

    def executar(self):
        """Loop principal otimizado"""
        self.inicializar_pygame()
        clock = pygame.time.Clock()
        start_time = time.time()
        
        while self.running and len(self.palavras) < self.max_palavras:
            self.processar_eventos()
            
            # Batch generation
            if len(self.palavras) < self.max_palavras:
                batch_size = min(100, self.max_palavras - len(self.palavras))
                new_palavras = [self.gerar_palavra() for _ in range(batch_size)]
                self.palavras.extend(new_palavras)
            
            # Física em batch
            self.space.step(1.0/60.0)
            
            # Renderização
            self.renderizar_cena()
            self.renderizar_info()
            
            # Atualiza estatísticas
            self.stats.update({
                'fps': clock.get_fps(),
                'palavras_processadas': len(self.palavras),
                'clusters_ativos': len(self.clusters),
                'memoria_usada': psutil.Process().memory_info().rss / 1024 / 1024,
                'tempo_execucao': time.time() - start_time
            })
            
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()

    def inicializar_pygame(self):
        """Inicializa Pygame e OpenGL"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Visualizador Orbital de Palavras BERT 3D")
        
        # Inicializa fonte
        pygame.freetype.init()
        self.font = pygame.freetype.SysFont('Arial', 24)
        
        # Configurações OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Configurações de ponto
        glPointSize(2.0)
        glEnable(GL_POINT_SPRITE)
        
        # Configuração da projeção
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.window_size[0]/self.window_size[1]), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Cor de fundo
        glClearColor(0.0, 0.0, 0.05, 1.0)

    def renderizar_palavra(self, palavra):
        """Renderiza uma palavra individual"""
        glPushMatrix()
        
        # Usa posição 2D do corpo físico + Z armazenado
        pos_2d = palavra.corpo.position
        pos_3d = (pos_2d[0], pos_2d[1], palavra.z)
        
        # Configura ponto
        glPointSize(2.0)  # Tamanho reduzido do ponto
        
        # Define cor com brilho
        glColor4f(*palavra.cor, 0.8)
        
        # Desenha ponto
        glBegin(GL_POINTS)
        glVertex3f(pos_3d[0], pos_3d[1], pos_3d[2])
        glEnd()
        
        glPopMatrix()
        
    def renderizar_cluster(self, cluster):
        """Renderiza um cluster de forma sutil"""
        glPushMatrix()
        
        # Converte posição 2D para 3D
        pos_2d = cluster.corpo.position
        pos_3d = (pos_2d[0], pos_2d[1], 0)
        
        # Define cor muito transparente para cluster
        glColor4f(*cluster.cor, 0.1)  # Alpha muito baixo
        
        # Desenha pontos para representar área do cluster
        glPointSize(1.0)
        glBegin(GL_POINTS)
        for _ in range(20):  # 20 pontos por cluster
            offset = np.random.normal(0, cluster.raio/2, 3)
            glVertex3f(
                pos_3d[0] + offset[0],
                pos_3d[1] + offset[1],
                pos_3d[2] + offset[2]
            )
        glEnd()
        
        glPopMatrix()
        
    def processar_eventos(self):
        """Processa eventos do Pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Botão esquerdo
                    self.mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    self.camera['zoom'] *= 0.9
                elif event.button == 5:  # Scroll down
                    self.camera['zoom'] *= 1.1
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    x, y = pygame.mouse.get_pos()
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]
                    self.camera['rotacao_y'] += dx * 0.5
                    self.camera['rotacao_x'] += dy * 0.5
                    self.last_mouse_pos = (x, y)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.resetar_camera()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador Orbital de Palavras BERT 3D[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Sistema",
            border_style="green"
        ))
        
        visualizer = BertOrbitalVisualizer()
        visualizer.executar()
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise
