"""
Visualizador 3D de Perceptron Multicamadas
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Visualização 3D interativa da arquitetura de um Perceptron Multicamadas.
"""

import numpy as np
import plotly.graph_objects as go
from rich.console import Console
from rich.panel import Panel
import time
from datetime import datetime
import hashlib
import plotly.io as pio
import math

console = Console()

class VisualizadorPerceptron3D:
    def __init__(self):
        # Arquitetura do Perceptron
        self.camadas = [
            64,     # Camada de entrada
            128,    # Primeira camada oculta
            256,    # Segunda camada oculta
            128,    # Terceira camada oculta
            32,     # Quarta camada oculta
            1       # Camada de saída
        ]
        
        # Cores para diferentes elementos
        self.colors = {
            'input': '#FF9999',      # Rosa para entrada
            'hidden': '#99FF99',     # Verde para camadas ocultas
            'output': '#9999FF',     # Azul para saída
            'connection': '#CCCCCC',  # Cinza para conexões
            'activation': '#FFFF99'   # Amarelo para ativações
        }
        
        # Configurações de visualização
        self.neuron_size = 0.3
        self.layer_spacing = 10
        self.neuron_spacing = 1.5
        
    def criar_neuronio(self, x, y, z, cor, nome):
        """Cria visualização de um neurônio"""
        theta = np.linspace(0, 2*np.pi, 20)
        phi = np.linspace(0, np.pi, 20)
        
        # Cria esfera para o neurônio
        x_sphere = x + self.neuron_size * np.outer(np.cos(theta), np.sin(phi))
        y_sphere = y + self.neuron_size * np.outer(np.sin(theta), np.sin(phi))
        z_sphere = z + self.neuron_size * np.outer(np.ones(20), np.cos(phi))
        
        return go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            colorscale=[[0, cor], [1, cor]],
            showscale=False,
            name=nome,
            opacity=0.8
        )
        
    def criar_conexoes(self, x1, y1, z1, x2, y2, z2):
        """Cria conexões entre neurônios"""
        return go.Scatter3d(
            x=[x1, x2],
            y=[y1, y2],
            z=[z1, z2],
            mode='lines',
            line=dict(color=self.colors['connection'], width=1),
            showlegend=False,
            opacity=0.3
        )
        
    def criar_camada_ativacao(self, z_pos, largura, altura):
        """Cria visualização da função de ativação"""
        x = np.linspace(-largura/2, largura/2, 20)
        y = np.linspace(-altura/2, altura/2, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_pos)
        
        return go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, self.colors['activation']], [1, self.colors['activation']]],
            showscale=False,
            name='Activation Function',
            opacity=0.2
        )
        
    def visualizar_perceptron_3d(self):
        """Cria visualização 3D completa do Perceptron"""
        console.print("\n🎨 Gerando visualização 3D do Perceptron...", style="bold yellow")
        
        fig = go.Figure()
        
        # Para cada camada
        for i, neurons in enumerate(self.camadas):
            z_pos = i * self.layer_spacing
            
            # Determina a cor da camada
            if i == 0:
                cor = self.colors['input']
                tipo = 'Input'
            elif i == len(self.camadas) - 1:
                cor = self.colors['output']
                tipo = 'Output'
            else:
                cor = self.colors['hidden']
                tipo = 'Hidden'
            
            # Calcula posições dos neurônios
            altura_total = (neurons - 1) * self.neuron_spacing
            for j in range(neurons):
                y_pos = -altura_total/2 + j * self.neuron_spacing
                
                # Adiciona neurônio
                fig.add_trace(self.criar_neuronio(
                    0, y_pos, z_pos, 
                    cor, 
                    f'{tipo} Neuron {j+1}'
                ))
                
                # Adiciona conexões com a próxima camada
                if i < len(self.camadas) - 1:
                    next_neurons = self.camadas[i + 1]
                    next_altura = (next_neurons - 1) * self.neuron_spacing
                    
                    for k in range(next_neurons):
                        next_y = -next_altura/2 + k * self.neuron_spacing
                        fig.add_trace(self.criar_conexoes(
                            0, y_pos, z_pos,
                            0, next_y, z_pos + self.layer_spacing
                        ))
            
            # Adiciona camada de ativação
            if i < len(self.camadas) - 1:
                fig.add_trace(self.criar_camada_ativacao(
                    z_pos + self.layer_spacing/2,
                    2,
                    altura_total + 2
                ))
        
        # Configuração do layout
        fig.update_layout(
            title="Arquitetura Perceptron Multicamadas 3D",
            width=1920,
            height=1080,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Camadas",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=2.5, y=0.5, z=0.5)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=2)
            ),
            showlegend=True
        )
        
        # Salva visualização
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        filename = f"perceptron_3d_{timestamp}_{hash_id}.html"
        png_filename = f"perceptron_3d_{timestamp}_{hash_id}.png"
        
        fig.write_html(filename)
        pio.write_image(fig, png_filename)
        
        console.print(f"\n✅ Visualizações salvas:", style="bold green")
        console.print(f"  • HTML: {filename}")
        console.print(f"  • PNG: {png_filename}")
        
        return filename, png_filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador 3D de Perceptron Multicamadas[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualização
        visualizador = VisualizadorPerceptron3D()
        html_file, png_file = visualizador.visualizar_perceptron_3d()
        
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
