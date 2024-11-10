"""
Visualizador 3D de Arquitetura Transformer
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Visualização 3D interativa da arquitetura completa de um Transformer.
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

class VisualizadorTransformer3D:
    def __init__(self):
        self.num_layers = 12        # Número de camadas do transformer
        self.num_heads = 8          # Número de cabeças de atenção
        self.d_model = 768          # Dimensão do modelo
        self.d_ff = 3072           # Dimensão do feed-forward
        self.vocab_size = 30000     # Tamanho do vocabulário
        
        # Cores
        self.colors = {
            'embedding': '#FF9999',      # Rosa claro
            'attention': '#99FF99',      # Verde claro
            'ffn': '#9999FF',           # Azul claro
            'layernorm': '#FFFF99',     # Amarelo claro
            'connection': '#CCCCCC',     # Cinza
            'output': '#FF99FF'         # Roxo claro
        }
        
    def criar_camada_embedding(self, z_pos):
        """Cria visualização da camada de embedding"""
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_pos)
        
        return go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, self.colors['embedding']], [1, self.colors['embedding']]],
            showscale=False,
            name='Embedding Layer',
            opacity=0.8
        )
        
    def criar_atencao_multihead(self, z_pos, layer_idx):
        """Cria visualização da atenção multi-head"""
        traces = []
        
        # Cria esferas para cada cabeça de atenção
        for head in range(self.num_heads):
            theta = np.linspace(0, 2*np.pi, 20)
            phi = np.linspace(0, np.pi, 20)
            
            # Posiciona as cabeças em círculo
            center_x = 4 * np.cos(2 * np.pi * head / self.num_heads)
            center_y = 4 * np.sin(2 * np.pi * head / self.num_heads)
            
            x = center_x + np.outer(np.cos(theta), np.sin(phi))
            y = center_y + np.outer(np.sin(theta), np.sin(phi))
            z = z_pos + np.outer(np.ones(20), np.cos(phi))
            
            traces.append(go.Surface(
                x=x, y=y, z=z,
                colorscale=[[0, self.colors['attention']], [1, self.colors['attention']]],
                showscale=False,
                name=f'Attention Head {head+1} (Layer {layer_idx+1})',
                opacity=0.7
            ))
            
        return traces
        
    def criar_ffn(self, z_pos, layer_idx):
        """Cria visualização da rede feed-forward"""
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        
        # Cria uma superfície ondulada para representar FFN
        Z = z_pos + 0.5 * np.sin(np.sqrt(X**2 + Y**2))
        
        return go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, self.colors['ffn']], [1, self.colors['ffn']]],
            showscale=False,
            name=f'FFN Layer {layer_idx+1}',
            opacity=0.7
        )
        
    def criar_layer_norm(self, z_pos, layer_idx):
        """Cria visualização da normalização de camada"""
        x = np.linspace(-4, 4, 20)
        y = np.linspace(-4, 4, 20)
        X, Y = np.meshgrid(x, y)
        Z = z_pos + 0.2 * np.exp(-(X**2 + Y**2)/8)
        
        return go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, self.colors['layernorm']], [1, self.colors['layernorm']]],
            showscale=False,
            name=f'Layer Norm {layer_idx+1}',
            opacity=0.6
        )
        
    def criar_conexoes(self, z_start, z_end):
        """Cria conexões entre camadas"""
        traces = []
        
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = 3 * np.cos(angle)
            y = 3 * np.sin(angle)
            
            traces.append(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z_start, z_end],
                mode='lines',
                line=dict(color=self.colors['connection'], width=2),
                showlegend=False
            ))
            
        return traces
        
    def visualizar_transformer_3d(self):
        """Cria visualização 3D completa do Transformer"""
        console.print("\n🎨 Gerando visualização 3D do Transformer...", style="bold yellow")
        
        fig = go.Figure()
        
        # Camada de Embedding
        fig.add_trace(self.criar_camada_embedding(0))
        
        # Camadas do Transformer
        z_spacing = 4  # Espaçamento vertical entre camadas
        
        for layer in range(self.num_layers):
            z_pos = (layer + 1) * z_spacing
            
            # Adiciona camadas de atenção
            attention_traces = self.criar_atencao_multihead(z_pos, layer)
            for trace in attention_traces:
                fig.add_trace(trace)
            
            # Adiciona FFN
            fig.add_trace(self.criar_ffn(z_pos + 1, layer))
            
            # Adiciona Layer Norm
            fig.add_trace(self.criar_layer_norm(z_pos + 2, layer))
            
            # Adiciona conexões
            connection_traces = self.criar_conexoes(z_pos, z_pos + 1)
            for trace in connection_traces:
                fig.add_trace(trace)
        
        # Configuração do layout
        fig.update_layout(
            title="Arquitetura Transformer 3D",
            width=1920,
            height=1080,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Camadas",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=2)
            ),
            showlegend=True
        )
        
        # Salva visualização
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_id = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        
        filename = f"transformer_3d_{timestamp}_{hash_id}.html"
        png_filename = f"transformer_3d_{timestamp}_{hash_id}.png"
        
        fig.write_html(filename)
        pio.write_image(fig, png_filename)
        
        console.print(f"\n✅ Visualizações salvas:", style="bold green")
        console.print(f"  • HTML: {filename}")
        console.print(f"  • PNG: {png_filename}")
        
        return filename, png_filename

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador 3D de Arquitetura Transformer[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Cria e executa visualização
        visualizador = VisualizadorTransformer3D()
        html_file, png_file = visualizador.visualizar_transformer_3d()
        
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
