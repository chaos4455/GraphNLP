"""
Nuvem de Palavras 3D Expansiva
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from rich.console import Console
from rich.panel import Panel
import time
import threading
import queue
from datetime import datetime

console = Console()

class NuvemPalavras3D:
    def __init__(self):
        self.window_size = (1600, 900)
        self.max_points = 5000
        self.points = []
        self.colors = []
        self.words = []
        self.point_queue = queue.Queue()
        
        # Controles de câmera
        self.camera_distance = -30
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 1.0
        
        # Estado
        self.running = True
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # Palavras exemplo (pode ser substituído por seu próprio gerador)
        self.palavras_exemplo = [
            "inovação", "tecnologia", "desenvolvimento", "estratégia",
            "gestão", "processos", "qualidade", "performance", "eficiência",
            "resultados", "analytics", "digital", "transformação", "dados",
            "inteligência", "artificial", "machine", "learning", "cloud",
            "segurança", "automação", "integração", "sistema", "plataforma"
        ]
        
    def inicializar_opengl(self):
        """Inicializa OpenGL com configurações otimizadas"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Nuvem de Palavras 3D")
        
        # Configurações OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Luz ambiente suave
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        self.resetar_camera()
        
    def resetar_camera(self):
        """Reseta posição da câmera"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.window_size[0]/self.window_size[1], 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def gerar_ponto(self):
        """Gera ponto 3D com distribuição esférica expansiva"""
        r = random.gauss(5.0, 2.0)
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        return np.array([x, y, z])
        
    def gerar_cor(self):
        """Gera cor com tendência para certas cores base"""
        cores_base = [
            [1.0, 0.0, 0.0],  # Vermelho
            [0.0, 1.0, 0.0],  # Verde
            [0.0, 0.0, 1.0],  # Azul
            [1.0, 0.5, 0.0],  # Laranja
            [0.0, 1.0, 1.0],  # Ciano
            [1.0, 0.0, 1.0],  # Magenta
            [0.5, 0.5, 1.0],  # Azul claro
            [1.0, 1.0, 0.0]   # Amarelo
        ]
        
        cor_base = random.choice(cores_base)
        variacao = np.random.normal(0, 0.1, 3)
        cor = np.clip(cor_base + variacao, 0, 1)
        return cor.tolist()
        
    def desenhar_ponto(self, point, color, size=2.0):
        """Desenha um ponto com efeito de brilho"""
        glPointSize(size)
        
        # Ponto principal
        glBegin(GL_POINTS)
        glColor3fv(color)
        glVertex3fv(point)
        glEnd()
        
        # Halo suave
        glPointSize(size * 1.5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBegin(GL_POINTS)
        glColor4f(color[0], color[1], color[2], 0.3)
        glVertex3fv(point)
        glEnd()
        glDisable(GL_BLEND)
        
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
                    
    def renderizar(self):
        """Renderiza a cena"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Posiciona câmera
        glTranslatef(0, 0, self.camera_distance * self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Desenha pontos
        for point, color in zip(self.points, self.colors):
            self.desenhar_ponto(point, color)
            
        pygame.display.flip()
        
    def adicionar_palavra(self):
        """Adiciona nova palavra à nuvem"""
        if len(self.points) < self.max_points:
            palavra = random.choice(self.palavras_exemplo)
            ponto = self.gerar_ponto()
            cor = self.gerar_cor()
            
            self.points.append(ponto)
            self.colors.append(cor)
            self.words.append(palavra)
            
    def executar(self):
        """Loop principal"""
        self.inicializar_opengl()
        clock = pygame.time.Clock()
        
        while self.running and len(self.points) < self.max_points:
            self.processar_eventos()
            
            # Adiciona novas palavras
            for _ in range(5):  # 5 palavras por frame
                self.adicionar_palavra()
                
            self.renderizar()
            clock.tick(60)
            
            # Atualiza título
            progresso = (len(self.points) / self.max_points) * 100
            pygame.display.set_caption(
                f"Nuvem de Palavras 3D - {progresso:.1f}% - "
                f"{len(self.points)}/{self.max_points} palavras"
            )
            
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Nuvem de Palavras 3D Expansiva[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualizador
        nuvem = NuvemPalavras3D()
        nuvem.executar()
        
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
