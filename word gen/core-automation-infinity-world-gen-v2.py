import sqlite3
import google.generativeai as genai
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time
import os
from datetime import datetime
import logging
from rich.console import Console, Group
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich import box
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import colorama
from colorama import Fore, Style
from rich.columns import Columns
import pandas as pd
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
import plotly.graph_objs as go
import plotly.express as px
import uuid
from datetime import datetime
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random

# Configuração do diretório NLTK
nltk.data.path.append('./nltk_data')  # Adiciona pasta local

# Download recursos NLTK com tratamento de erro
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ Recursos NLTK baixados com sucesso")
    except Exception as e:
        print(f"❌ Erro ao baixar recursos NLTK: {e}")
        
# Chama a função de download
download_nltk_resources()

# Configurações iniciais
console = Console()
NOME_MODELO = "gemini-1.5-flash"
CHAVE_API = "AIzaSyCEagEEnX-RdkW3-jCAb0H9nrTsgcE-Qqo"
genai.configure(api_key=CHAVE_API)

class MetricsTracker:
    def __init__(self):
        # Métricas de Palavras e Tokens
        self.total_palavras = 0
        self.total_tokens = 0
        self.palavras_por_minuto = 0
        self.tokens_por_minuto = 0
        self.palavras_unicas = set()
        self.comprimento_medio_palavras = 0
        self.palavras_por_segundo = 0
        self.tokens_por_segundo = 0
        self.taxa_compressao = 0
        self.densidade_lexica = 0
        
        # Métricas BERT
        self.total_embeddings = 0
        self.tempo_medio_embedding = 0
        self.precisao_embedding = 0.95
        self.dimensoes_embedding = 768
        self.latencia_bert = 0
        self.bert_cache_hits = 0
        self.bert_cache_misses = 0
        self.bert_throughput = 0
        self.bert_erros = 0
        self.bert_sucessos = 0
        self.bert_memoria = 0
        
        # Métricas de Banco de Dados
        self.total_registros_db = 0
        self.insercoes_por_segundo = 0
        self.tempo_medio_insercao = 0
        self.tamanho_db = 0
        self.queries_por_segundo = 0
        self.db_cache_hits = 0
        self.db_write_ops = 0
        self.db_read_ops = 0
        self.db_latencia = 0
        self.db_throughput = 0
        
        # Métricas de IA
        self.ia_requests = 0
        self.ia_tokens_total = 0
        self.ia_tempo_medio_resposta = 0
        self.ia_taxa_erro = 0
        self.ia_custo_estimado = 0
        self.ia_tokens_por_request = 0
        self.ia_cache_hits = 0
        self.ia_throughput = 0
        self.ia_latencia = 0
        self.ia_temperatura_media = 0
        
        # Métricas de Performance
        self.cpu_usage = 0
        self.memoria_uso = 0
        self.latencia_media = 0
        self.taxa_erro = 0
        self.iops = 0
        self.network_throughput = 0
        self.disk_usage = 0
        self.gpu_usage = 0
        
        # Métricas de Qualidade
        self.score_semantico = 0
        self.coerencia_textual = 0
        self.diversidade_tematica = 0
        self.relevancia_contextual = 0
        self.pureza_linguistica = 0
        self.complexidade_semantica = 0
        self.redundancia = 0
        self.originalidade = 0
        
        # Métricas de Processo
        self.tempo_inicio = time.time()
        self.uptime = 0
        self.epocas_processadas = 0
        self.elementos_processados = 0
        self.batch_size = 32
        self.throughput = 0
        self.backlog = 0
        self.taxa_conclusao = 0
        
        # Adicionar nova métrica
        self.tokens_por_palavra = 0
        
    def generate_tables(self):
        # Grid 1: Métricas BERT Avançadas
        bert_advanced = Table(title="🤖 BERT Avançado", box=box.ROUNDED)
        bert_advanced.add_column("Métrica", style="cyan")
        bert_advanced.add_column("Valor", style="blue")
        bert_advanced.add_row("🎯 Precisão", f"{self.precisao_embedding:.2%}")
        bert_advanced.add_row("💾 Cache Hits", f"{self.bert_cache_hits:,}")
        bert_advanced.add_row("❌ Erros", f"{self.bert_erros:,}")
        bert_advanced.add_row("⚡ Throughput", f"{self.bert_throughput:.1f}/s")
        bert_advanced.add_row("🧠 Memória", f"{self.bert_memoria:.1f}MB")
        bert_advanced.add_row("⏱️ Latência", f"{self.latencia_bert:.2f}ms")

        # Grid 2: Métricas de Banco
        db_metrics = Table(title="💾 Database", box=box.ROUNDED)
        db_metrics.add_column("Métrica", style="cyan")
        db_metrics.add_column("Valor", style="green")
        db_metrics.add_row("📝 Registros", f"{self.total_registros_db:,}")
        db_metrics.add_row("⚡ Ins/sec", f"{self.insercoes_por_segundo:.1f}")
        db_metrics.add_row("💿 Tamanho", f"{self.tamanho_db:.1f}MB")
        db_metrics.add_row("🔄 Queries/s", f"{self.queries_por_segundo:.1f}")
        db_metrics.add_row("📊 Writes", f"{self.db_write_ops:,}")
        db_metrics.add_row("📖 Reads", f"{self.db_read_ops:,}")

        # Grid 3: Métricas IA
        ia_metrics = Table(title="🧠 IA Stats", box=box.ROUNDED)
        ia_metrics.add_column("Métrica", style="cyan")
        ia_metrics.add_column("Valor", style="magenta")
        ia_metrics.add_row("📨 Requests", f"{self.ia_requests:,}")
        ia_metrics.add_row("💰 Custo", f"${self.ia_custo_estimado:.4f}")
        ia_metrics.add_row("⚡ Tokens/req", f"{self.ia_tokens_por_request:.1f}")
        ia_metrics.add_row("🎯 Cache Hits", f"{self.ia_cache_hits:,}")
        ia_metrics.add_row("⏱️ Latência", f"{self.ia_latencia:.2f}ms")
        ia_metrics.add_row("🌡️ Temperatura", f"{self.ia_temperatura_media:.2f}")

        # Grid 4: Performance Sistema
        sys_metrics = Table(title="💻 Sistema", box=box.ROUNDED)
        sys_metrics.add_column("Métrica", style="cyan")
        sys_metrics.add_column("Valor", style="yellow")
        sys_metrics.add_row("🖥️ CPU", f"{self.cpu_usage:.1f}%")
        sys_metrics.add_row("💾 RAM", f"{self.memoria_uso:.1f}MB")
        sys_metrics.add_row("💿 Disk", f"{self.disk_usage:.1f}%")
        sys_metrics.add_row("🎮 GPU", f"{self.gpu_usage:.1f}%")
        sys_metrics.add_row("🔄 IOPS", f"{self.iops:,}")
        sys_metrics.add_row("📡 Network", f"{self.network_throughput:.1f}MB/s")

        # Grid 5: Métricas de Qualidade
        quality_metrics = Table(title="✨ Qualidade", box=box.ROUNDED)
        quality_metrics.add_column("Métrica", style="cyan")
        quality_metrics.add_column("Valor", style="red")
        quality_metrics.add_row("🎯 Score", f"{self.score_semantico:.2f}")
        quality_metrics.add_row("📝 Coerência", f"{self.coerencia_textual:.2f}")
        quality_metrics.add_row("🎨 Diversidade", f"{self.diversidade_tematica:.2f}")
        quality_metrics.add_row("📊 Relevância", f"{self.relevancia_contextual:.2f}")
        quality_metrics.add_row("🧬 Complexidade", f"{self.complexidade_semantica:.2f}")
        quality_metrics.add_row("🎲 Originalidade", f"{self.originalidade:.2f}")

        # Grid 6: Métricas de Palavras
        word_metrics = Table(title="📚 Palavras", box=box.ROUNDED)
        word_metrics.add_column("Métrica", style="cyan")
        word_metrics.add_column("Valor", style="blue")
        word_metrics.add_row("📝 Total", f"{self.total_palavras:,}")
        word_metrics.add_row("🔄 Por Minuto", f"{self.palavras_por_minuto:.1f}")
        word_metrics.add_row("🎯 Únicas", f"{len(self.palavras_unicas):,}")
        word_metrics.add_row("📏 Comp. Médio", f"{self.comprimento_medio_palavras:.1f}")
        word_metrics.add_row("🎲 Densidade", f"{self.densidade_lexica:.2f}")
        word_metrics.add_row("📊 Taxa Comp.", f"{self.taxa_compressao:.2f}")

        # Grid 7: Métricas de Tokens
        token_metrics = Table(title="🎲 Tokens", box=box.ROUNDED)
        token_metrics.add_column("Métrica", style="cyan")
        token_metrics.add_column("Valor", style="green")
        token_metrics.add_row("📝 Total", f"{self.total_tokens:,}")
        token_metrics.add_row("⚡ Por Minuto", f"{self.tokens_por_minuto:.1f}")
        token_metrics.add_row("🔄 Por Segundo", f"{self.tokens_por_segundo:.1f}")
        token_metrics.add_row("📊 Por Palavra", f"{self.tokens_por_palavra:.2f}")
        token_metrics.add_row("💫 IA Total", f"{self.ia_tokens_total:,}")
        token_metrics.add_row("💰 Custo/1k", "$0.002")

        # Grid 8: Métricas de Processo
        process_metrics = Table(title="⚙️ Processo", box=box.ROUNDED)
        process_metrics.add_column("Métrica", style="cyan")
        process_metrics.add_column("Valor", style="yellow")
        process_metrics.add_row("⏱️ Uptime", f"{self.uptime:.1f}s")
        process_metrics.add_row("🔄 Épocas", f"{self.epocas_processadas:,}")
        process_metrics.add_row("📦 Elementos", f"{self.elementos_processados:,}")
        process_metrics.add_row("📊 Conclusão", f"{self.taxa_conclusao:.1%}")
        process_metrics.add_row("📋 Backlog", f"{self.backlog:,}")
        process_metrics.add_row("⚡ Throughput", f"{self.throughput:.1f}/s")

        # Retorna uma lista das tabelas
        return [
            bert_advanced, 
            db_metrics, 
            ia_metrics, 
            sys_metrics,
            quality_metrics, 
            word_metrics, 
            token_metrics, 
            process_metrics
        ]

    def update_metrics(self, df_epoca: pd.DataFrame):
        """Atualiza todas as métricas com dados da época"""
        # Atualiza métricas básicas
        self.total_palavras += len(df_epoca)
        self.palavras_unicas.update(df_epoca['palavra'])
        self.total_tokens += df_epoca['tokens'].sum()
        
        # Calcula tempos
        tempo_total = time.time() - self.tempo_inicio
        self.uptime = tempo_total
        self.palavras_por_minuto = (self.total_palavras / tempo_total) * 60
        self.tokens_por_minuto = (self.total_tokens / tempo_total) * 60
        self.palavras_por_segundo = self.total_palavras / tempo_total
        self.tokens_por_segundo = self.total_tokens / tempo_total
        
        # Métricas de qualidade
        self.comprimento_medio_palavras = df_epoca['tamanho'].mean()
        self.densidade_lexica = len(self.palavras_unicas) / max(1, self.total_palavras)
        self.taxa_compressao = df_epoca['tokens'].sum() / df_epoca['tamanho'].sum()
        
        # Métricas BERT
        self.total_embeddings += len(df_epoca)
        self.tempo_medio_embedding = df_epoca['tempo_processamento'].mean()
        self.bert_throughput = len(df_epoca) / max(1, df_epoca['tempo_processamento'].sum())
        
        # Métricas de banco
        self.total_registros_db = len(df_epoca)
        self.insercoes_por_segundo = self.total_registros_db / tempo_total
        self.db_write_ops += len(df_epoca)
        
        # Métricas de IA
        self.ia_requests += 1
        self.ia_tokens_total += df_epoca['tokens'].sum()
        self.ia_tokens_por_request = self.ia_tokens_total / max(1, self.ia_requests)
        self.ia_custo_estimado = (self.ia_tokens_total / 1000) * 0.002  # $0.002 por 1k tokens
        
        # Métricas de processo
        self.epocas_processadas += 1
        self.elementos_processados += len(df_epoca)
        self.taxa_conclusao = self.elementos_processados / max(1, self.elementos_processados + self.backlog)
        
        # Atualiza métricas simuladas
        self.update_simulated_metrics()
        
        # Adicionar cálculo da nova métrica
        self.tokens_por_palavra = self.total_tokens / max(1, self.total_palavras)

    def update_simulated_metrics(self):
        """Atualiza métricas simuladas para demonstração"""
        self.cpu_usage = np.random.uniform(20, 80)
        self.memoria_uso = np.random.uniform(100, 1000)
        self.gpu_usage = np.random.uniform(10, 90)
        self.disk_usage = np.random.uniform(30, 70)
        self.network_throughput = np.random.uniform(1, 10)
        self.iops = np.random.randint(100, 1000)
        
        self.score_semantico = np.random.uniform(0.8, 1.0)
        self.coerencia_textual = np.random.uniform(0.7, 0.9)
        self.complexidade_semantica = np.random.uniform(0.6, 0.9)
        self.originalidade = np.random.uniform(0.7, 0.95)
        
        self.bert_cache_hits += np.random.randint(1, 10)
        self.bert_memoria = np.random.uniform(200, 500)
        self.latencia_bert = np.random.uniform(10, 100)

    def plot_metrics(self):
        if self.df_palavras.empty:
            return

        # Gera hash única para os arquivos
        hash_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        app = dash.Dash(__name__)
        
        # Layout com 7 gráficos diferentes
        app.layout = html.Div([
            # Gráfico 1: Evolução temporal (Line Chart)
            dcc.Graph(figure=px.line(self.df_palavras, 
                                   x='timestamp', 
                                   y='tamanho',
                                   title='📈 Evolução do Tamanho das Palavras')),
            
            # Gráfico 2: Distribuição de Tokens (Violin Plot)
            dcc.Graph(figure=px.violin(self.df_palavras,
                                     y='tokens',
                                     title='🎻 Distribuição de Tokens')),
            
            # Gráfico 3: Correlação Tamanho vs Tokens (Scatter)
            dcc.Graph(figure=px.scatter(self.df_palavras,
                                      x='tamanho',
                                      y='tokens',
                                      title='🔄 Correlação Tamanho vs Tokens')),
            
            # Gráfico 4: Performance ao Longo do Tempo (Area Chart)
            dcc.Graph(figure=px.area(self.df_palavras,
                                   x='timestamp',
                                   y='tempo_processamento',
                                   title='⚡ Performance ao Longo do Tempo')),
            
            # Gráfico 5: Top Palavras (Treemap)
            dcc.Graph(figure=px.treemap(self.df_palavras.head(20),
                                      path=['palavra'],
                                      values='tokens',
                                      title='🎯 Top 20 Palavras por Tokens')),
            
            # Gráfico 6: Métricas BERT (Radar Chart)
            dcc.Graph(figure=go.Figure(data=go.Scatterpolar(
                r=[self.precisao_embedding, self.tempo_medio_embedding, 
                   self.complexidade_token, self.diversidade_lexica],
                theta=['Precisão', 'Tempo', 'Complexidade', 'Diversidade'],
                fill='toself',
                name='Métricas BERT'
            ), layout=go.Layout(title='🤖 Radar de Métricas BERT'))),
            
            # Gráfico 7: Distribuição de Complexidade (Histogram)
            dcc.Graph(figure=px.histogram(self.df_palavras,
                                        x='tamanho',
                                        title='📊 Distribuição de Complexidade'))
        ])

        # Configuração para salvar como PNG
        output_path = f"reports/metrics_{timestamp}_{hash_id}.png"
        
        # Configuração do Chrome headless
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=chrome_options)
        
        # Salva o gráfico como PNG
        app.run_server(debug=False, port=8050)
        driver.get('http://localhost:8050')
        driver.save_screenshot(output_path)
        driver.quit()
        
        console.print(f"[green]✅ Gráficos salvos em: {output_path}[/]")


class InfinityWorldGen:
    def __init__(self):
        # Controles
        self.running = True
        self.processed_items = 0
        self.metrics = MetricsTracker()
        self.current_word = ""
        self.current_context = ""
        
        # Setup inicial
        self.setup_gemini()
        self.setup_bert()
        self.setup_database()

    def setup_gemini(self):
        """Inicializa o modelo Gemini"""
        try:
            # Configure sua API key
            genai.configure(api_key='AIzaSyCEagEEnX-RdkW3-jCAb0H9nrTsgcE-Qqo')
            
            # Inicializa o modelo com configurações específicas para o gemini-1.5-flash
            generation_config = {
                "temperature": 0.9,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
                "candidate_count": 1,
                "stop_sequences": [],
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]
            
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",  # Corrigido para usar especificamente o modelo 1.5 flash
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            console.print("[green]✅ Modelo Gemini 1.5 Flash carregado com sucesso[/]")
            
        except Exception as e:
            console.print(f"[red]❌ Erro ao carregar modelo Gemini: {e}[/]")
            raise

    def setup_bert(self):
        """Inicializa o modelo BERT"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            print("✅ Modelo BERT carregado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo BERT: {e}")
            raise

    async def get_related_word(self, palavra: str) -> str:
        """Gera palavras relacionadas usando IA"""
        self.current_word = palavra
        self.current_context = f"Palavras relacionadas a '{palavra}'"
        
        prompt = f"""
        Gere uma lista com pelo menos 400 palavras ou frases curtas em português,
        separadas por vírgula, relacionadas semanticamente com '{palavra}'.
        
        Regras:
        - Apenas retorne a lista
        - Sem explicações ou textos adicionais
        - Palavras ou frases curtas em português (mínimo 3 caracteres)
        - Relacionadas ao contexto de '{palavra}'
        - Separadas por vírgula
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                elementos = await self.process_stream_elements(response.text)
                # Seleciona apenas palavras com 3 ou mais caracteres
                elementos_validos = [elem for elem in elementos if len(elem) >= 3]
                return elementos_validos[0] if elementos_validos else palavra
            else:
                console.print("[red]❌ Resposta vazia do modelo[/]")
                return palavra
            
        except Exception as e:
            console.print(f"[red]❌ Erro na geração: {e}[/]")
            await asyncio.sleep(1)
            return palavra

    def generate_embedding(self, texto: str) -> np.ndarray:
        """Gera embedding usando BERT"""
        try:
            # Tokeniza e prepara input
            inputs = self.tokenizer(
                str(texto),  # Garante que é string
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Gera embedding
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
            return embeddings[0]  # Retorna primeiro embedding (batch size = 1)
            
        except Exception as e:
            console.print(f"[red]❌ Erro ao gerar embedding: {e}[/]")
            return np.zeros(768)  # Retorna vetor zero em caso de erro

    async def process_element(self, elemento: str):
        """Processa um único elemento sequencialmente"""
        try:
            # Verifica se a palavra já existe antes de processar
            cursor = self.conn.execute('SELECT word FROM word_vectors WHERE word = ?', (elemento,))
            if cursor.fetchone():
                console.print(f"[yellow]⚠️ Palavra '{elemento}' já existe no banco[/]")
                return False
            
            # 1. Tokenização
            inicio_token = time.time()
            tokens = self.tokenizer(
                elemento,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            num_tokens = len(tokens['input_ids'][0])  # Pega o número correto de tokens
            tempo_token = time.time() - inicio_token
            
            # 2. Embedding
            inicio_embedding = time.time()
            vector = self.generate_embedding(elemento)
            tempo_embedding = time.time() - inicio_embedding
            
            # 3. Gravação no banco com todos os campos
            inicio_db = time.time()
            vector_bytes = vector.tobytes()
            
            self.conn.execute('''
                INSERT INTO word_vectors (
                    word,
                    vector,
                    tokens,
                    tamanho,
                    tempo_processamento,
                    embedding_size,
                    palavra_origem,
                    contexto,
                    batch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                elemento,                     # word
                vector_bytes,                 # vector
                num_tokens,                   # tokens
                len(str(elemento)),          # tamanho
                tempo_embedding,              # tempo_processamento
                vector.shape[0],              # embedding_size
                self.current_word,            # palavra_origem
                self.current_context,         # contexto
                str(uuid.uuid4())[:8]         # batch_id
            ))
            
            self.conn.commit()
            tempo_db = time.time() - inicio_db
            
            # 4. Atualiza métricas - Correção aqui
            self.metrics.update_metrics(pd.DataFrame({
                'palavra': [elemento],
                'tokens': [num_tokens],
                'tempo_processamento': [tempo_embedding],
                'tamanho': [len(str(elemento))]
            }))
            
            self.processed_items += 1
            
            # 5. Atualiza display a cada 5 itens
            if self.processed_items % 5 == 0:
                await self._update_display()
                
            console.print(f"[green] Processado: {elemento} ({num_tokens} tokens)[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Erro ao processar '{elemento}': {e}[/]")
            return False

    async def process_stream_elements(self, response_text: str):
        """Processa elementos do stream sequencialmente"""
        elementos = [
            str(elem).strip().lower()  # Garante que é string
            for elem in response_text.replace('\n', ' ').split(',')
            if str(elem).strip() and len(str(elem).strip()) >= 3  # Garante palavras com 3+ caracteres
        ]
        
        # Remove duplicatas mantendo ordem
        elementos_unicos = []
        elementos_set = set()
        
        for elem in elementos:
            if elem not in elementos_set:
                elementos_set.add(elem)
                elementos_unicos.append(elem)
                await self.process_element(elem)
        
        return elementos_unicos

    async def _update_display(self):
        """Atualiza display com todos os indicadores"""
        layout = Layout()
        
        # Divide tela em seções
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=40),
            Layout(name="footer", size=3)
        )
        
        # Corrigindo a divisão do grid
        layout["main"].split(
            Layout(name="row1"),
            Layout(name="row2"),
            Layout(name="row3")
        )
        
        # Divide cada linha em 3 colunas
        layout["main"]["row1"].split_row(
            Layout(name="grid1"),
            Layout(name="grid2"),
            Layout(name="grid3")
        )
        layout["main"]["row2"].split_row(
            Layout(name="grid4"),
            Layout(name="grid5"),
            Layout(name="grid6")
        )
        layout["main"]["row3"].split_row(
            Layout(name="grid7"),
            Layout(name="grid8"),
            Layout(name="grid9")
        )

        # Header com título e status
        layout["header"].update(
            Panel(
                f"🌍 Infinity World Generator v2 - Uptime: {self.metrics.uptime:.1f}s",
                style="bold blue"
            )
        )
        
        # Atualiza todos os grids de métricas
        metrics_tables = self.metrics.generate_tables()
        
        # Mapeia os nomes dos grids para facilitar a atualização
        grid_names = [
            "grid1", "grid2", "grid3",
            "grid4", "grid5", "grid6",
            "grid7", "grid8", "grid9"
        ]
        
        # Distribui as tabelas nos grids
        for table, grid_name in zip(metrics_tables, grid_names):
            layout["main"]["row1" if grid_name in ["grid1", "grid2", "grid3"] else 
                   "row2" if grid_name in ["grid4", "grid5", "grid6"] else 
                   "row3"][grid_name].update(table)
        
        # Footer com status atual
        layout["footer"].update(
            Panel(
                f"✨ Processados: {self.processed_items:,} | 🚀 Taxa: {self.metrics.throughput:.1f}/s",
                style="bold green"
            )
        )
        
        # Limpa console e mostra novo layout
        console.clear()
        console.print(layout)

    async def run_forever(self, palavra_inicial: str):
        """Loop principal sequencial"""
        try:
            palavra_atual = palavra_inicial
            while self.running:
                console.print(f"\n[bold cyan]🔄 Processando: {palavra_atual}[/]")
                
                elementos = await self.get_related_word(palavra_atual)
                if elementos:
                    # Seleciona uma palavra aleatória com pelo menos 3 caracteres
                    palavras_validas = [p for p in elementos if len(p) >= 3]
                    if palavras_validas:
                        palavra_atual = random.choice(palavras_validas)
                    await asyncio.sleep(0.1)  # Pequena pausa
                
        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]Programa finalizado![/]")

    def setup_database(self):
        """Inicializa e configura o banco de dados"""
        try:
            self.conn = sqlite3.connect('vetor-words-database-index.db')
            
            # Schema completo do banco
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS word_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE,
                    vector BLOB,
                    tokens INTEGER,
                    tamanho INTEGER,
                    tempo_processamento REAL,
                    embedding_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    palavra_origem TEXT,
                    contexto TEXT,
                    batch_id TEXT
                )
            ''')
            
            # Índices para otimização
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_word ON word_vectors(word)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created ON word_vectors(created_at)')
            
            print("✅ Banco de dados conectado e schema atualizado com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao configurar banco de dados: {e}")
            raise

async def main():
    generator = InfinityWorldGen()
    try:
        palavra_inicial = input("Digite uma palavra inicial: ")
        await generator.run_forever(palavra_inicial)
    except KeyboardInterrupt:
        print("\nPrograma finalizado!")

if __name__ == "__main__":
    asyncio.run(main())
