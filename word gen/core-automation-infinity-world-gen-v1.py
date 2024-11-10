import sqlite3
import google.generativeai as genai
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time
import os
from datetime import datetime
import logging
from rich.console import Console
from rich.panel import Panel

# Configurações iniciais
console = Console()
NOME_MODELO = "gemini-1.5-flash"
CHAVE_API = "AIzaSyCEagEEnX-RdkW3-jCAb0H9nrTsgcE-Qqo"
genai.configure(api_key=CHAVE_API)

class InfinityWorldGen:
    def __init__(self):
        self.setup_model()
        self.setup_database()
        
    def setup_model(self):
        """Inicializa o modelo BERT e tokenizer"""
        try:
            model_name = "neuralmind/bert-base-portuguese-cased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            console.print(f"[red]Erro ao carregar modelo: {e}[/]")
            exit(1)

    def setup_database(self):
        """Configura conexão com banco de dados"""
        try:
            self.conn = sqlite3.connect('vectors_continuo.db')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS word_vectors (
                    id INTEGER PRIMARY KEY,
                    word TEXT UNIQUE,
                    vector BLOB,
                    palavra_origem TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        except Exception as e:
            console.print(f"[red]Erro ao configurar banco de dados: {e}[/]")
            exit(1)

    def generate_embedding(self, word: str) -> np.ndarray:
        """Gera embedding para uma palavra"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(word, return_tensors='pt', padding=True).to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                return embedding
        except Exception as e:
            print(f"Erro ao gerar embedding para '{word}': {e}")
            return np.zeros(768)

    def get_related_word(self, palavra_base: str) -> str:
        """Obtém uma palavra relacionada usando Gemini"""
        try:
            prompt = f"""
            Gere 400 palavra inspirada ou  relacionada a '{palavra_base}'bem criativo 
            Requisitos:
            - Apenas 400 palavra
            - Sem explicações
            - Sem pontuação
            - Sem formatação
            1 palavra por linha
            ou 1 frase por linha curta
            """
            
            model = genai.GenerativeModel(NOME_MODELO)
            response = model.generate_content(prompt)
            word = response.text.strip().lower()
            return word
        except Exception as e:
            console.print(f"[red]Erro ao obter palavra relacionada: {e}[/]")
            return None

    async def run_forever(self, palavra_inicial: str):
        """Executa o processo infinitamente"""
        console.print(Panel.fit(
            f"[cyan]Iniciando geração infinita a partir da palavra: [bold]{palavra_inicial}[/][/]",
            border_style="cyan"
        ))
        
        count = 0
        while True:
            try:
                # Obtém nova palavra
                nova_palavra = self.get_related_word(palavra_inicial)
                if not nova_palavra:
                    continue
                
                # Gera e salva embedding
                vector = self.generate_embedding(nova_palavra)
                vector_bytes = vector.tobytes()
                
                # Salva no banco
                self.conn.execute(
                    'INSERT OR IGNORE INTO word_vectors (word, vector, palavra_origem) VALUES (?, ?, ?)',
                    (nova_palavra, vector_bytes, palavra_inicial)
                )
                self.conn.commit()
                
                # Atualiza contador e exibe progresso
                count += 1
                console.print(f"[green]#{count} - Palavra processada: {nova_palavra}[/]")
                
                # Pequena pausa para não sobrecarregar a API
                time.sleep(1)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Programa interrompido pelo usuário[/]")
                break
            except Exception as e:
                console.print(f"[red]Erro: {e}[/]")
                time.sleep(5)  # Pausa maior em caso de erro

def main():
    try:
        console.print("[bold cyan]🌍 Gerador Infinito de Palavras e Vetores[/]")
        palavra_inicial = input("\nDigite a palavra inicial: ").strip().lower()
        
        if not palavra_inicial:
            console.print("[red]Palavra inválida![/]")
            return
            
        generator = InfinityWorldGen()
        import asyncio
        asyncio.run(generator.run_forever(palavra_inicial))
        
    except Exception as e:
        console.print(f"[red]Erro crítico: {e}[/]")
    finally:
        console.print("[cyan]Programa finalizado[/]")

if __name__ == "__main__":
    main()
