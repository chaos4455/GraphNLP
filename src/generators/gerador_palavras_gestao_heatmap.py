"""
Gerador de Palavras de Gestão com Mapa de Calor
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera 14 mil palavras sobre gestão empresarial com visualização em tempo real.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
from rich.console import Console
from rich.panel import Panel
import threading
import queue
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
from datetime import datetime
import os

console = Console()

class GeradorPalavrasGestao:
    def __init__(self):
        self.width = 256
        self.height = 256
        self.window_size = (1200, 800)
        self.running = True
        self.palavras_geradas = 0
        self.total_palavras = 14000
        self.output_file = f"palavras_gestao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Temas e contextos
        self.temas = {
            'gestao': [
                "gestão empresarial", "administração", "liderança", "estratégia",
                "planejamento", "eficiência", "produtividade", "performance",
                "resultados", "metas", "objetivos", "indicadores"
            ],
            'financeiro': [
                "receita", "custos", "despesas", "lucro", "investimento",
                "orçamento", "fluxo de caixa", "rentabilidade", "roi",
                "faturamento", "margem", "capital"
            ],
            'processos': [
                "otimização", "automação", "padronização", "qualidade",
                "melhoria contínua", "lean", "six sigma", "processos",
                "workflow", "metodologia", "framework", "boas práticas"
            ]
        }
        
        # Prompts base
        self.prompts = [
            "Como melhorar a {} da empresa através de {}",
            "Implementando {} para aumentar a {}",
            "Estratégias de {} focadas em {}",
            "Boas práticas de {} para otimizar {}",
            "Metodologia {} aplicada à {}",
            "Técnicas de {} para maximizar {}"
        ]
        
        # Buffer para o mapa de calor
        self.heat_map = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.palavras_buffer = []
        
        # Inicializa modelo
        self.tokenizer = GPT2Tokenizer.from_pretrained('pierreguillou/gpt2-small-portuguese')
        self.model = GPT2LMHeadModel.from_pretrained('pierreguillou/gpt2-small-portuguese')
        self.model.eval()
        
    def gerar_prompt(self):
        """Gera um prompt aleatório combinando temas"""
        tema1 = random.choice(list(self.temas.keys()))
        tema2 = random.choice(list(self.temas.keys()))
        palavra1 = random.choice(self.temas[tema1])
        palavra2 = random.choice(self.temas[tema2])
        prompt = random.choice(self.prompts).format(palavra1, palavra2)
        return prompt
        
    def gerar_texto(self, prompt):
        """Gera texto baseado no prompt"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            texto = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return texto
        except Exception as e:
            console.print(f"[bold red]❌ Erro na geração: {str(e)}[/]")
            return ""
            
    def atualizar_mapa_calor(self):
        """Atualiza o mapa de calor baseado nas palavras geradas"""
        try:
            # Calcula progresso
            progresso = self.palavras_geradas / self.total_palavras
            
            # Atualiza mapa gradualmente
            for i in range(self.width):
                for j in range(self.height):
                    if random.random() < progresso:
                        # Verde aumenta com o progresso
                        self.heat_map[i, j, 0] = 1 - progresso  # Vermelho diminui
                        self.heat_map[i, j, 1] = progresso      # Verde aumenta
                        self.heat_map[i, j, 2] = 0.0            # Sem azul
                        
        except Exception as e:
            console.print(f"[bold red]❌ Erro na atualização do mapa: {str(e)}[/]")
            
    def salvar_palavras(self, texto):
        """Salva palavras geradas no arquivo"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(texto + "\n\n")
        except Exception as e:
            console.print(f"[bold red]❌ Erro ao salvar: {str(e)}[/]")
            
    def processar_geracoes(self):
        """Processa geração de palavras em background"""
        while self.running and self.palavras_geradas < self.total_palavras:
            try:
                prompt = self.gerar_prompt()
                texto = self.gerar_texto(prompt)
                
                if texto:
                    self.palavras_geradas += len(texto.split())
                    self.salvar_palavras(texto)
                    console.print(f"[green]✓ Geradas {self.palavras_geradas} palavras[/]")
                    
            except Exception as e:
                console.print(f"[bold red]❌ Erro no processamento: {str(e)}[/]")
                
    def inicializar_pygame(self):
        """Inicializa PyGame e OpenGL"""
        pygame.init()
        pygame.display.set_mode(self.window_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Gerador de Palavras - Mapa de Calor")
        
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
        # Inicia thread de geração
        thread_geracao = threading.Thread(target=self.processar_geracoes)
        thread_geracao.daemon = True
        thread_geracao.start()
        
        fps_clock = pygame.time.Clock()
        
        while self.running and self.palavras_geradas < self.total_palavras:
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
            
            # Atualiza título
            progresso = (self.palavras_geradas / self.total_palavras) * 100
            pygame.display.set_caption(
                f"Gerador de Palavras - {progresso:.1f}% Completo - "
                f"{self.palavras_geradas}/{self.total_palavras} palavras"
            )
            
            fps_clock.tick(30)
        
        pygame.quit()

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Gerador de Palavras de Gestão com Mapa de Calor[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa gerador
        gerador = GeradorPalavrasGestao()
        gerador.inicializar_pygame()
        gerador.executar()
        
        tempo_total = time.time() - start_time
        console.print(f"\n⏱️ Tempo total de execução: {tempo_total:.2f} segundos", style="bold blue")
        console.print(f"\n📝 Arquivo gerado: {gerador.output_file}", style="bold green")
        
    except Exception as e:
        console.print(f"[bold red]❌ Erro: {str(e)}[/]")
        raise

"""
Assinatura: Elias Andrade
Arquiteto de Soluções
Replika AI - Maringá, PR
"""
