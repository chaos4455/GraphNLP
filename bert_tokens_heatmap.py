"""
Visualizador de Tokens BERT com Mapa de Calor
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Visualização em tempo real da ativação de tokens BERT com mapa de calor.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
from rich.console import Console
from rich.panel import Panel
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import softmax
import threading
import queue
import random
from scipy.ndimage import zoom

console = Console()

class BertTokenHeatmap:
    def __init__(self):
        self.width = 256
        self.height = 256
        self.window_size = (1024, 800)
        self.running = True
        self.time_start = time.time()
        
        # Inicializa BERT com configuração específica
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased',
            add_pooling_layer=False,
            attn_implementation="eager"  # Evita o warning
        )
        self.model.eval()
        
        # Buffer para o mapa de calor
        self.heat_map = np.zeros((self.width, self.height, 3), dtype=np.float32)
        
        # Fila de processamento
        self.token_queue = queue.Queue()
        
        # Dados de tokens
        self.current_tokens = []
        self.attention_weights = np.zeros((12, 12))  # 12 camadas x 12 cabeças
        self.token_embeddings = np.zeros((512, 768))  # Max 512 tokens x 768 dim
        
        # Fonte PyGame
        self.font = None
        
        # Palavras para processamento
        self.tech_words = [
            "python", "javascript", "docker", "kubernetes", "machine learning",
            "artificial intelligence", "deep learning", "neural networks",
            "cloud computing", "microservices", "devops", "data science",
            "blockchain", "cybersecurity", "big data", "automation"
        ]
        
    def inicializar_pygame(self):
        """Inicializa PyGame e OpenGL"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("BERT Token Heatmap")
        
        # Inicializa fonte
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        
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
        
    def processar_tokens(self):
        """Processa tokens em background"""
        while self.running:
            try:
                # Gera texto aleatório combinando palavras tech
                text = " ".join(random.sample(self.tech_words, 3))
                
                # Tokeniza e processa
                with torch.no_grad():
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                    outputs = self.model(**inputs, output_attentions=True)
                    
                    # Extrai atenções e embeddings
                    attentions = outputs.attentions
                    last_hidden = outputs.last_hidden_state
                    
                    # Processa atenções - Corrigido para garantir matriz 2D
                    att_layers = [att[0].mean(0) for att in attentions]  # Média das cabeças
                    att_matrix = torch.mean(torch.stack(att_layers), dim=0)  # Média das camadas
                    self.attention_weights = att_matrix.numpy()
                    
                    # Processa embeddings
                    self.token_embeddings = last_hidden[0].numpy()
                    
                    # Atualiza tokens atuais
                    self.current_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    
                time.sleep(0.1)  # Pequena pausa
                
            except Exception as e:
                console.print(f"[bold red]❌ Erro no processamento: {str(e)}[/]")
                
    def atualizar_mapa_calor(self):
        """Atualiza o mapa de calor baseado nas atenções e embeddings"""
        try:
            if self.attention_weights.ndim < 2:
                return  # Evita processamento se não tivermos uma matriz 2D
            
            # Normaliza atenções para o mapa de calor
            att_min = np.min(self.attention_weights)
            att_max = np.max(self.attention_weights)
            if att_max > att_min:
                att_normalized = (self.attention_weights - att_min) / (att_max - att_min)
            else:
                att_normalized = np.zeros_like(self.attention_weights)
            
            # Redimensiona para 256x256 usando interpolação
            from scipy.ndimage import zoom
            
            # Calcula fatores de zoom para cada dimensão
            zoom_x = self.width / att_normalized.shape[0]
            zoom_y = self.height / att_normalized.shape[1]
            
            # Aplica zoom
            att_resized = zoom(att_normalized, (zoom_x, zoom_y))
            
            # Garante dimensões corretas
            att_resized = att_resized[:self.width, :self.height]
            
            # Atualiza mapa de calor
            self.heat_map[:, :, 0] = att_resized  # Vermelho para alta atenção
            self.heat_map[:, :, 1] = 1 - att_resized  # Verde para baixa atenção
            self.heat_map[:, :, 2] = 0.0  # Sem azul
            
        except Exception as e:
            console.print(f"[bold red]❌ Erro na atualização do mapa: {str(e)}[/]")
            
    def renderizar(self):
        """Renderiza o mapa de calor e informações"""
        try:
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
            
        except Exception as e:
            console.print(f"[bold red]❌ Erro na renderização: {str(e)}[/]")
        
    def executar(self):
        """Loop principal"""
        # Inicia thread de processamento
        process_thread = threading.Thread(target=self.processar_tokens)
        process_thread.daemon = True
        process_thread.start()
        
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
                pygame.display.set_caption(
                    f"BERT Token Heatmap - FPS: {fps} - Tokens: {len(self.current_tokens)}"
                )
                frames = 0
                last_time = time.time()
            
            fps_clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador de Tokens BERT com Mapa de Calor[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualização
        visualizador = BertTokenHeatmap()
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
