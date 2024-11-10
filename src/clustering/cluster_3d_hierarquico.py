"""
Visualizador de Clusters Hierárquicos 3D Interativo
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.cluster import MiniBatchKMeans
import time
from rich.console import Console
import threading
import queue
from dataclasses import dataclass
import random

console = Console()

@dataclass
class Cluster3D:
    """Estrutura de dados para clusters"""
    center: np.ndarray
    points: list
    color: list
    radius: float
    level: int  # 0=macro, 1=meso, 2=micro
    
class ClusterVisualizer3D:
    def __init__(self):
        # Configurações da janela
        self.window_size = (1600, 900)
        self.fov = 45
        self.near = 0.1
        self.far = 100.0
        
        # Controles de câmera
        self.camera_distance = -30
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 1.0
        
        # Configurações de clusters
        self.macro_clusters = 5
        self.meso_clusters = 3
        self.micro_clusters = 2
        self.max_points = 5000
        
        # Dados
        self.points = []
        self.clusters = []
        self.point_queue = queue.Queue()
        
        # Estado
        self.running = True
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # Cores base para níveis
        self.colors = {
            'macro': [
                [1.0, 0.0, 0.0],  # Vermelho
                [0.0, 1.0, 0.0],  # Verde
                [0.0, 0.0, 1.0],  # Azul
                [1.0, 1.0, 0.0],  # Amarelo
                [1.0, 0.0, 1.0]   # Magenta
            ],
            'meso': [
                [0.8, 0.2, 0.2],
                [0.2, 0.8, 0.2],
                [0.2, 0.2, 0.8]
            ],
            'micro': [
                [0.6, 0.4, 0.4],
                [0.4, 0.6, 0.4]
            ]
        }
        
    def inicializar_opengl(self):
        """Configura OpenGL"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Clusters Hierárquicos 3D")
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Luz ambiente
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        self.resetar_camera()
        
    def resetar_camera(self):
        """Reseta posição da câmera"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.window_size[0]/self.window_size[1], self.near, self.far)
        glMatrixMode(GL_MODELVIEW)
        
    def gerar_ponto(self):
        """Gera ponto 3D aleatório"""
        cluster = random.randint(0, self.macro_clusters-1)
        center = np.array([
            random.gauss(cluster * 3, 0.5),
            random.gauss(cluster * 2, 0.5),
            random.gauss(cluster * 2, 0.5)
        ])
        return center + np.random.normal(0, 0.3, 3)
        
    def atualizar_clusters(self):
        """Atualiza hierarquia de clusters"""
        if len(self.points) < 100:
            return
            
        # Macro clusters
        kmeans_macro = MiniBatchKMeans(
            n_clusters=self.macro_clusters,
            random_state=42
        ).fit(self.points)
        
        self.clusters = []
        
        # Para cada macro cluster
        for i in range(self.macro_clusters):
            mask = kmeans_macro.labels_ == i
            points_macro = np.array(self.points)[mask]
            
            if len(points_macro) < 10:
                continue
                
            # Cria macro cluster
            self.clusters.append(Cluster3D(
                center=kmeans_macro.cluster_centers_[i],
                points=points_macro.tolist(),
                color=self.colors['macro'][i % len(self.colors['macro'])],
                radius=2.0,
                level=0
            ))
            
            # Meso clusters
            kmeans_meso = MiniBatchKMeans(
                n_clusters=self.meso_clusters,
                random_state=42
            ).fit(points_macro)
            
            # Para cada meso cluster
            for j in range(self.meso_clusters):
                mask_meso = kmeans_meso.labels_ == j
                points_meso = points_macro[mask_meso]
                
                if len(points_meso) < 5:
                    continue
                    
                # Cria meso cluster
                self.clusters.append(Cluster3D(
                    center=kmeans_meso.cluster_centers_[j],
                    points=points_meso.tolist(),
                    color=self.colors['meso'][j % len(self.colors['meso'])],
                    radius=1.0,
                    level=1
                ))
                
                # Micro clusters
                kmeans_micro = MiniBatchKMeans(
                    n_clusters=self.micro_clusters,
                    random_state=42
                ).fit(points_meso)
                
                # Para cada micro cluster
                for k in range(self.micro_clusters):
                    mask_micro = kmeans_micro.labels_ == k
                    points_micro = points_meso[mask_micro]
                    
                    if len(points_micro) < 3:
                        continue
                        
                    # Cria micro cluster
                    self.clusters.append(Cluster3D(
                        center=kmeans_micro.cluster_centers_[k],
                        points=points_micro.tolist(),
                        color=self.colors['micro'][k % len(self.colors['micro'])],
                        radius=0.5,
                        level=2
                    ))
                    
    def desenhar_cluster(self, cluster):
        """Desenha um cluster"""
        glColor4f(*cluster.color, 0.7)  # Alpha para transparência
        
        # Desenha esfera do cluster
        quad = gluNewQuadric()
        glPushMatrix()
        glTranslatef(*cluster.center)
        gluSphere(quad, cluster.radius, 32, 32)
        glPopMatrix()
        
        # Desenha pontos
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for point in cluster.points:
            glVertex3fv(point)
        glEnd()
        
        # Desenha conexões se for macro cluster
        if cluster.level == 0:
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for point in cluster.points:
                glVertex3fv(cluster.center)
                glVertex3fv(point)
            glEnd()
            
    def renderizar(self):
        """Renderiza a cena"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Posiciona câmera
        glTranslatef(0, 0, self.camera_distance * self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Desenha eixos
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(10,0,0)  # X
        glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0,10,0)  # Y
        glColor3f(0, 0, 1); glVertex3f(0,0,0); glVertex3f(0,0,10)  # Z
        glEnd()
        
        # Desenha clusters
        for cluster in self.clusters:
            self.desenhar_cluster(cluster)
            
        pygame.display.flip()
        
    def processar_eventos(self):
        """Processa eventos de mouse e teclado"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Botão esquerdo
                    self.mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    self.zoom *= 0.9
                elif event.button == 5:  # Scroll down
                    self.zoom *= 1.1
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_pressed:
                    x, y = pygame.mouse.get_pos()
                    dx = x - self.last_mouse_pos[0]
                    dy = y - self.last_mouse_pos[1]
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                    self.last_mouse_pos = (x, y)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.resetar_camera()
                    
    def executar(self):
        """Loop principal"""
        self.inicializar_opengl()
        clock = pygame.time.Clock()
        
        while self.running and len(self.points) < self.max_points:
            self.processar_eventos()
            
            # Gera novos pontos
            if len(self.points) < self.max_points:
                new_point = self.gerar_ponto()
                self.points.append(new_point)
                
                # Atualiza clusters a cada 100 pontos
                if len(self.points) % 100 == 0:
                    self.atualizar_clusters()
                    
            self.renderizar()
            clock.tick(60)
            
            # Atualiza título
            progresso = (len(self.points) / self.max_points) * 100
            pygame.display.set_caption(
                f"Clusters 3D - {progresso:.1f}% - "
                f"{len(self.points)}/{self.max_points} pontos"
            )
            
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print("[bold green]Iniciando Visualizador de Clusters 3D[/]")
        visualizer = ClusterVisualizer3D()
        visualizer.executar()
    except Exception as e:
        console.print(f"[bold red]Erro: {str(e)}[/]")
