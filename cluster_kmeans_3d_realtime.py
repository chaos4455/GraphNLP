"""
Visualizador de Clusters K-means 3D em Tempo Real
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Visualização 3D interativa de clusters K-means em tempo real.
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import time
from rich.console import Console
from rich.panel import Panel
import threading
import queue
from datetime import datetime
import random
import math

console = Console()

class ClusterKMeans3DRealtime:
    def __init__(self):
        self.window_size = (1200, 800)
        self.running = True
        
        # Parâmetros do clustering
        self.n_clusters = 8
        self.batch_size = 100
        self.max_points = 10000
        self.current_points = 0
        
        # Dados e clusters
        self.points = np.zeros((self.max_points, 3))
        self.colors = np.zeros((self.max_points, 3))
        self.centroids = np.zeros((self.n_clusters, 3))
        self.labels = np.zeros(self.max_points, dtype=int)
        
        # Cores para clusters
        self.cluster_colors = [
            [1.0, 0.0, 0.0],  # Vermelho
            [0.0, 1.0, 0.0],  # Verde
            [0.0, 0.0, 1.0],  # Azul
            [1.0, 1.0, 0.0],  # Amarelo
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Ciano
            [1.0, 0.5, 0.0],  # Laranja
            [0.5, 0.0, 1.0]   # Roxo
        ]
        
        # Rotação da visualização
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        
        # Modelo K-means
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=42
        )
        
        # Scaler para normalização
        self.scaler = StandardScaler()
        
        # Fila de processamento
        self.point_queue = queue.Queue()
        
    def gerar_ponto(self):
        """Gera um novo ponto 3D com distribuição específica"""
        # Escolhe um cluster aleatório
        cluster = random.randint(0, self.n_clusters-1)
        
        # Gera ponto baseado no cluster
        center = np.array([
            random.gauss(cluster * 2, 0.5),
            random.gauss(cluster * 1.5, 0.5),
            random.gauss(cluster * 1.2, 0.5)
        ])
        
        # Adiciona ruído
        noise = np.random.normal(0, 0.2, 3)
        return center + noise
        
    def processar_clusters(self):
        """Processa clusters em background"""
        while self.running and self.current_points < self.max_points:
            try:
                # Gera novo ponto
                new_point = self.gerar_ponto()
                
                # Adiciona à fila
                self.point_queue.put(new_point)
                
                # Atualiza clusters se tiver pontos suficientes
                if self.current_points >= self.batch_size:
                    # Pega apenas os pontos atuais
                    current_data = self.points[:self.current_points]
                    
                    # Normaliza dados
                    points_norm = self.scaler.fit_transform(current_data)
                    
                    # Atualiza clusters
                    self.kmeans.partial_fit(points_norm)
                    
                    # Pega as labels apenas para os pontos atuais
                    current_labels = self.kmeans.predict(points_norm)
                    
                    # Atualiza labels e cores
                    self.labels[:self.current_points] = current_labels
                    for i in range(self.current_points):
                        self.colors[i] = self.cluster_colors[self.labels[i]]
                    
                    # Atualiza centroids
                    self.centroids = self.scaler.inverse_transform(
                        self.kmeans.cluster_centers_
                    )
                else:
                    # Se não tiver pontos suficientes, usa cores padrão
                    self.colors[self.current_points] = self.cluster_colors[0]
                
                time.sleep(0.01)  # Pequena pausa
                
            except Exception as e:
                console.print(f"[bold red]❌ Erro no processamento: {str(e)}[/]")
                continue  # Continua mesmo com erro
        
    def inicializar_pygame(self):
        """Inicializa PyGame e OpenGL"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Clusters K-means 3D")
        
        # Configuração OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Configuração da câmera
        gluPerspective(45, (self.window_size[0]/self.window_size[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -30)
        
    def desenhar_ponto(self, point, color, size=0.1):
        """Desenha um ponto 3D"""
        glColor3fv(color)
        glPointSize(size * 10)
        glBegin(GL_POINTS)
        glVertex3fv(point)
        glEnd()
        
    def desenhar_centroid(self, point, color, size=0.3):
        """Desenha um centroid"""
        glColor3fv(color)
        glPointSize(size * 20)
        glBegin(GL_POINTS)
        glVertex3fv(point)
        glEnd()
        
        # Desenha linhas conectando ao centro
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3fv(point)
        glEnd()
        
    def renderizar(self):
        """Renderiza a visualização 3D"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Aplica rotações
        glTranslatef(0.0, 0.0, -30)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)
        
        # Desenha eixos
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(10, 0, 0)  # X
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 10, 0)  # Y
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 10)  # Z
        glEnd()
        
        # Desenha pontos
        for i in range(self.current_points):
            self.desenhar_ponto(self.points[i], self.colors[i])
            
        # Desenha centroids
        for i in range(self.n_clusters):
            self.desenhar_centroid(self.centroids[i], self.cluster_colors[i])
        
        pygame.display.flip()
        
    def executar(self):
        """Loop principal"""
        # Inicia thread de processamento
        thread_processo = threading.Thread(target=self.processar_clusters)
        thread_processo.daemon = True
        thread_processo.start()
        
        fps_clock = pygame.time.Clock()
        mouse_pressed = False
        last_mouse_pos = (0, 0)
        
        while self.running and self.current_points < self.max_points:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pressed = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION and mouse_pressed:
                    new_mouse_pos = pygame.mouse.get_pos()
                    dx = new_mouse_pos[0] - last_mouse_pos[0]
                    dy = new_mouse_pos[1] - last_mouse_pos[1]
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                    last_mouse_pos = new_mouse_pos
            
            # Processa novos pontos da fila
            while not self.point_queue.empty() and self.current_points < self.max_points:
                new_point = self.point_queue.get()
                self.points[self.current_points] = new_point
                self.current_points += 1
            
            # Renderiza
            self.renderizar()
            
            # Atualiza título
            progresso = (self.current_points / self.max_points) * 100
            pygame.display.set_caption(
                f"Clusters K-means 3D - {progresso:.1f}% Completo - "
                f"{self.current_points}/{self.max_points} pontos"
            )
            
            fps_clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador de Clusters K-means 3D em Tempo Real[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualizador
        visualizador = ClusterKMeans3DRealtime()
        visualizador.inicializar_pygame()
        visualizador.executar()
        
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
