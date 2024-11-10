import sqlite3
import google.generativeai as genai
import time
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich import box
import random
import asyncio

class SpeedMetrics:
    def __init__(self):
        # Métricas simplificadas focadas em velocidade
        self.palavras_total = 0
        self.caracteres_total = 0
        self.palavras_por_segundo = 0
        self.caracteres_por_segundo = 0
        self.palavras_por_minuto = 0
        self.caracteres_por_minuto = 0
        self.tempo_inicio = time.time()
        self.palavras_unicas = set()
        self.maior_palavra = ""
        self.menor_palavra = "a" * 1000
        self.media_caracteres = 0
        self.total_requests = 0
        self.requests_por_segundo = 0
        # Adicionar métricas de banco
        self.palavras_pendentes = 0
        self.ultimo_id = 0
        self.ultima_palavra = ""
        self.status_banco = "🟢 Livre"
        self.total_gravados = 0
        self.ultimas_palavras = []  # Lista para as últimas 10 palavras
        self.maiores_palavras = []  # Lista para as 10 maiores palavras
        
    def add_palavra(self, palavra):
        # Atualiza últimas palavras
        self.ultimas_palavras.insert(0, palavra)
        self.ultimas_palavras = self.ultimas_palavras[:10]
        
        # Atualiza maiores palavras
        self.maiores_palavras.append(palavra)
        self.maiores_palavras.sort(key=len, reverse=True)
        self.maiores_palavras = self.maiores_palavras[:10]

    def generate_speed_table(self):
        speed_metrics = Table(title="⚡ Métricas de Velocidade", box=box.ROUNDED)
        speed_metrics.add_column("Métrica", style="cyan")
        speed_metrics.add_column("Valor", style="green")
        
        # Calcula métricas em tempo real
        tempo_total = time.time() - self.tempo_inicio
        self.palavras_por_segundo = self.palavras_total / max(1, tempo_total)
        self.caracteres_por_segundo = self.caracteres_total / max(1, tempo_total)
        self.palavras_por_minuto = self.palavras_por_segundo * 60
        self.caracteres_por_minuto = self.caracteres_por_segundo * 60
        self.requests_por_segundo = self.total_requests / max(1, tempo_total)
        
        # Adiciona linhas com métricas
        speed_metrics.add_row("⚡ Palavras/seg", f"{self.palavras_por_segundo:.1f}")
        speed_metrics.add_row("🚀 Palavras/min", f"{self.palavras_por_minuto:.1f}")
        speed_metrics.add_row("📝 Total Palavras", f"{self.palavras_total:,}")
        speed_metrics.add_row("🎯 Palavras Únicas", f"{len(self.palavras_unicas):,}")
        speed_metrics.add_row("💫 Chars/seg", f"{self.caracteres_por_segundo:.1f}")
        speed_metrics.add_row("📊 Chars Total", f"{self.caracteres_total:,}")
        speed_metrics.add_row("⏱️ Uptime", f"{tempo_total:.1f}s")
        speed_metrics.add_row("📏 Maior Palavra", f"{self.maior_palavra} ({len(self.maior_palavra)})")
        speed_metrics.add_row("📏 Menor Palavra", f"{self.menor_palavra} ({len(self.menor_palavra)})")
        speed_metrics.add_row("🔄 Requests/seg", f"{self.requests_por_segundo:.1f}")
        
        return speed_metrics

class UltraSpeedWorldGen:
    def __init__(self):
        self.running = True
        self.metrics = SpeedMetrics()
        self.console = Console()
        self.setup_gemini()
        self.setup_database()

    def setup_gemini(self):
        """Configuração simplificada do Gemini"""
        genai.configure(api_key='AIzaSyCEagEEnX-RdkW3-jCAb0H9nrTsgcE-Qqo')
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def setup_database(self):
        """Schema simplificado do banco"""
        self.conn = sqlite3.connect('ultra-speed-words.db')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE,
                chars INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_word ON words(word)')

    async def get_related_words(self, palavra: str) -> list:
        """Versão simplificada de geração de palavras"""
        prompt = f" baseado na {palavra}', o usuario quer fazer um dataset de textos, palavras e embeeding pra enriquecer o contexto de uma IA, crie então baseado na {palavra}', uma lista infinita de termos, palavras e frases, relativas a {palavra}' e seu contexto e correção de verossimilhança, , Liste 500 palavras em português unicas, seja criativa, expanda, busque palavras e contexto simular, nunca repita, tente expandir e trazer amplo contexto de palavras e frases relacionadas a '{palavra}'. Apenas a lista, separada por vírgula."
        
        try:
            response = self.model.generate_content(prompt)
            self.metrics.total_requests += 1
            
            if response and response.text:
                palavras = [
                    p.strip().lower() 
                    for p in response.text.replace('\n', ' ').split(',')
                    if len(p.strip()) >= 3
                ]
                return palavras
            return []
            
        except Exception as e:
            self.console.print(f"[red]Erro: {e}[/]")
            return []

    async def process_word(self, palavra: str):
        """Processamento com atualização das listas"""
        try:
            self.metrics.status_banco = "🔴 Ocupado"
            self.metrics.palavras_pendentes += 1
            
            # Atualiza métricas e listas
            self.metrics.add_palavra(palavra)
            self.metrics.palavras_total += 1
            self.metrics.caracteres_total += len(palavra)
            self.metrics.palavras_unicas.add(palavra)
            
            if len(palavra) > len(self.metrics.maior_palavra):
                self.metrics.maior_palavra = palavra
            if len(palavra) < len(self.metrics.menor_palavra):
                self.metrics.menor_palavra = palavra
            
            # Grava no banco com retorno do ID
            cursor = self.conn.execute(
                'INSERT OR IGNORE INTO words (word, chars) VALUES (?, ?)',
                (palavra, len(palavra))
            )
            
            # Atualiza métricas do banco
            self.metrics.ultimo_id = cursor.lastrowid or self.metrics.ultimo_id
            self.metrics.ultima_palavra = palavra
            self.metrics.total_gravados += 1 if cursor.rowcount > 0 else 0
            self.metrics.palavras_pendentes -= 1
            
            if self.metrics.palavras_total % 10 == 0:
                self.conn.commit()
                self.metrics.status_banco = "🟢 Livre"
                await self._update_display()
                
        except Exception as e:
            self.metrics.status_banco = "🟡 Erro"
            self.console.print(f"[red]Erro ao processar '{palavra}': {e}[/]")
        finally:
            self.metrics.status_banco = "🟢 Livre"

    def generate_top_words_table(self):
        """Tabela com as maiores palavras"""
        table = Table(title="📏 Top 10 Maiores", box=box.ROUNDED)
        table.add_column("Palavra", style="cyan")
        table.add_column("Tamanho", style="green", justify="right")
        
        for palavra in self.metrics.maiores_palavras:
            table.add_row(palavra, str(len(palavra)))
        return table

    def generate_latest_words_table(self):
        """Tabela com as últimas palavras"""
        table = Table(title="🕒 Últimas 10", box=box.ROUNDED)
        table.add_column("Palavra", style="cyan")
        table.add_column("ID", style="green", justify="right")
        
        for palavra in self.metrics.ultimas_palavras:
            table.add_row(palavra, str(self.metrics.ultimo_id))
        return table

    def generate_unique_words_table(self):
        """Tabela com palavras únicas"""
        table = Table(title="🎯 Top 10 Únicas", box=box.ROUNDED)
        table.add_column("Palavra", style="cyan")
        
        for palavra in list(self.metrics.palavras_unicas)[:10]:
            table.add_row(palavra)
        return table

    def generate_status_table(self):
        """Tabela de status do banco"""
        table = Table(title="📊 Status do Banco", box=box.ROUNDED)
        table.add_column("Métrica", style="yellow")
        table.add_column("Valor", style="cyan")
        
        table.add_row("Status", self.metrics.status_banco)
        table.add_row("Último ID", str(self.metrics.ultimo_id))
        table.add_row("Última Palavra", self.metrics.ultima_palavra)
        table.add_row("Total Gravados", f"{self.metrics.total_gravados:,}")
        table.add_row("Pendentes", f"{self.metrics.palavras_pendentes:,}")
        
        return table

    async def _update_display(self):
        """Display com 4 grids lado a lado"""
        layout = Layout()
        
        # Divide em header e content
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content", size=15),
            Layout(name="footer", size=3)
        )
        
        # Divide o content em 4 colunas
        layout["content"].split_row(
            Layout(name="status"),
            Layout(name="maiores"),
            Layout(name="unicas"),
            Layout(name="ultimas")
        )
        
        # Header
        layout["header"].update(Panel(
            "🚀 Ultra Speed World Generator v3",
            style="bold blue"
        ))
        
        # Atualiza os 4 grids
        layout["status"].update(self.generate_status_table())
        layout["maiores"].update(self.generate_top_words_table())
        layout["unicas"].update(self.generate_unique_words_table())
        layout["ultimas"].update(self.generate_latest_words_table())
        
        # Footer
        layout["footer"].update(Panel(
            f"✨ Palavras: {self.metrics.palavras_total:,} | ⚡ {self.metrics.palavras_por_segundo:.1f}/s",
            style="bold green"
        ))
        
        self.console.clear()
        self.console.print(layout)

    async def run_forever(self, palavra_inicial: str):
        """Loop principal otimizado"""
        palavra_atual = palavra_inicial
        while self.running:
            palavras = await self.get_related_words(palavra_atual)
            if palavras:
                for palavra in palavras:
                    await self.process_word(palavra)
                palavra_atual = random.choice(palavras)
            await asyncio.sleep(0.1)

async def main():
    generator = UltraSpeedWorldGen()
    try:
        palavra_inicial = input("Palavra inicial: ")
        await generator.run_forever(palavra_inicial)
    except KeyboardInterrupt:
        print("\nFinalizado!")

if __name__ == "__main__":
    asyncio.run(main())
