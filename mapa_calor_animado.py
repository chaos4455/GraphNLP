"""
Mapa de Calor Animado com PyGame e OpenGL
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Visualização de mapa de calor 256x256 animado usando OpenGL.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
from rich.console import Console
from rich.panel import Panel
import math

console = Console()

class MapaCalorAnimado:
    def __init__(self):
        self.width = 256
        self.height = 256
        self.window_size = (800, 800)
        self.running = True
        self.time_start = time.time()
        
        # Parâmetros de animação
        self.wave_speed = 2.0
        self.wave_scale = 0.05
        self.color_speed = 1.0
        
        # Buffer para o mapa de calor
        self.heat_map = np.zeros((self.width, self.height, 3), dtype=np.float32)
        
        # Textura OpenGL
        self.texture_id = None
        
    def inicializar_pygame(self):
        """Inicializa PyGame e OpenGL"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Mapa de Calor Animado")
        
        # Configuração OpenGL
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, *self.window_size)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(-1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Cria textura
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
    def atualizar_mapa_calor(self):
        """Atualiza o mapa de calor com padrões animados"""
        current_time = time.time() - self.time_start
        
        # Cria grades de coordenadas
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        # Gera padrões de onda
        wave1 = np.sin(2 * np.pi * (X + Y + current_time * self.wave_speed))
        wave2 = np.cos(2 * np.pi * (X - Y + current_time * self.wave_speed * 0.7))
        wave3 = np.sin(4 * np.pi * np.sqrt(X**2 + Y**2) - current_time * self.wave_speed)
        
        # Combina ondas
        combined = (wave1 + wave2 + wave3) / 3
        normalized = (combined + 1) / 2  # Normaliza para [0, 1]
        
        # Cria gradiente de cores
        color_factor = (np.sin(current_time * self.color_speed) + 1) / 2
        
        # Atualiza canais de cor
        self.heat_map[:, :, 0] = normalized * (1 - color_factor)  # Vermelho
        self.heat_map[:, :, 1] = normalized * color_factor        # Verde
        self.heat_map[:, :, 2] = 0.0                             # Azul
        
    def renderizar(self):
        """Renderiza o mapa de calor"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # Atualiza textura
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0,
            GL_RGB, GL_FLOAT, self.heat_map
        )
        
        # Desenha quad com textura
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        
        pygame.display.flip()
        
    def executar(self):
        """Loop principal"""
        fps_clock = pygame.time.Clock()
        frames = 0
        last_time = time.time()
        
        while self.running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Atualiza e renderiza
            self.atualizar_mapa_calor()
            self.renderizar()
            
            # FPS counter
            frames += 1
            if time.time() - last_time > 1.0:
                fps = frames
                pygame.display.set_caption(f"Mapa de Calor Animado - FPS: {fps}")
                frames = 0
                last_time = time.time()
            
            fps_clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Mapa de Calor Animado com PyGame e OpenGL[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualização
        visualizador = MapaCalorAnimado()
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
