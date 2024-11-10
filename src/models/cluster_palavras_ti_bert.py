"""
Visualizador 3D de Clusters de Palavras em TI/DevOps usando BERT
Autor: Elias Andrade - Arquiteto de Soluções
Versão: 1.0.0 (Micro-revisão 000000001)
Data: 2024-03-27
Descrição: Gera ~54k palavras em 55 temas de TI usando BERT, clusteriza com FAISS e visualiza em 3D.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F
import faiss
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.table import Table
import random
from datetime import datetime
import time
import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from sklearn.decomposition import PCA

# Inicialização
console = Console()

class ClusterizadorTIBert:
    def __init__(self):
        """Inicialização do sistema"""
        console.print(Panel.fit("🚀 Iniciando Clusterizador TI com BERT", style="bold green"))
        
        # Configurações de visualização
        self.display_size = (1600, 900)
        self.camera_config = {
            'rotation_x': 0.0,
            'rotation_y': 0.0,
            'rotation_z': 0.0,
            'scale': 1.0,
            'translate_z': -30.0
        }
        
        # Inicialização do BERT
        self.init_bert()
        
        # Processamento inicial
        self.definir_temas_ti()
        self.processar_dados()
        self.init_visualizacao()

    def init_bert(self):
        """Inicialização do modelo BERT"""
        console.print("\n🤖 Inicializando BERT...", style="bold yellow")
        
        with console.status("[bold green]Carregando modelos BERT..."):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.bert_model.eval()  # Modo de avaliação
        
        console.print("✅ BERT inicializado com sucesso!", style="bold green")

    def gerar_palavras_bert(self, contexto, num_palavras=10):
        """Gera palavras relacionadas usando BERT"""
        # Adiciona token de máscara
        texto = f"{contexto} [MASK]"
        inputs = self.tokenizer(texto, return_tensors="pt")
        
        # Obtém predições
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            predictions = outputs.logits
        
        # Obtém probabilidades para o token mascarado
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        mask_token_logits = predictions[0, mask_token_index, :]
        probs = F.softmax(mask_token_logits, dim=-1)
        
        # Obtém top palavras
        top_k = torch.topk(probs, num_palavras, dim=-1)
        return [self.tokenizer.decode([token_id.item()]).strip() for token_id in top_k.indices[0]]

    def definir_temas_ti(self):
        """Define os 55 temas de TI com suas palavras base"""
        self.temas_ti = {
            'cloud_computing': {
                'palavras_base': ['aws', 'azure', 'gcp', 'cloud', 'iaas', 'paas', 'saas', 'serverless',
                                'container', 'kubernetes', 'docker', 'microservices'],
                'contextos': ['cloud platform', 'cloud service', 'cloud infrastructure']
            },
            'devops': {
                'palavras_base': ['pipeline', 'ci/cd', 'jenkins', 'gitlab', 'github', 'automation',
                                'deployment', 'monitoring', 'logging', 'terraform'],
                'contextos': ['devops tools', 'continuous integration', 'continuous deployment']
            },
            'security': {
                'palavras_base': ['firewall', 'encryption', 'authentication', 'authorization', 'ssl',
                                'vpn', 'penetration', 'vulnerability', 'compliance'],
                'contextos': ['cybersecurity', 'network security', 'data security']
            },
            'networking': {
                'palavras_base': ['tcp/ip', 'dns', 'dhcp', 'routing', 'switching', 'wan', 'lan',
                                'subnet', 'vlan', 'load balancer'],
                'contextos': ['network protocol', 'network infrastructure']
            },
            'databases': {
                'palavras_base': ['sql', 'nosql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                                'mysql', 'oracle', 'cassandra'],
                'contextos': ['database management', 'data storage']
            },
            'infrastructure': {
                'palavras_base': ['server', 'datacenter', 'virtualization', 'vmware', 'hypervisor',
                                'storage', 'backup', 'disaster recovery', 'high availability'],
                'contextos': ['it infrastructure', 'server management', 'infrastructure maintenance']
            },
            'development': {
                'palavras_base': ['python', 'java', 'javascript', 'golang', 'rust', 'react',
                                'angular', 'vue', 'node.js', 'typescript', 'api'],
                'contextos': ['software development', 'programming language', 'web development']
            },
            'agile': {
                'palavras_base': ['scrum', 'kanban', 'sprint', 'backlog', 'user story', 'epic',
                                'retrospective', 'standup', 'agile methodology'],
                'contextos': ['agile practice', 'scrum methodology', 'project management']
            },
            'data_science': {
                'palavras_base': ['machine learning', 'deep learning', 'neural network', 'tensorflow',
                                'pytorch', 'pandas', 'numpy', 'scikit-learn', 'jupyter'],
                'contextos': ['data analysis', 'machine learning model', 'artificial intelligence']
            },
            'itsm': {
                'palavras_base': ['itil', 'service desk', 'incident', 'problem', 'change management',
                                'sla', 'cmdb', 'asset management', 'service catalog'],
                'contextos': ['it service management', 'service delivery', 'support process']
            },
            'automation': {
                'palavras_base': ['ansible', 'puppet', 'chef', 'terraform', 'powershell', 'bash',
                                'scripting', 'automation tool', 'configuration management'],
                'contextos': ['infrastructure automation', 'configuration automation', 'process automation']
            },
            'monitoring': {
                'palavras_base': ['prometheus', 'grafana', 'nagios', 'zabbix', 'datadog', 'splunk',
                                'elk stack', 'apm', 'metrics', 'alerting'],
                'contextos': ['system monitoring', 'performance monitoring', 'log monitoring']
            },
            'cloud_native': {
                'palavras_base': ['kubernetes', 'istio', 'helm', 'prometheus', 'envoy', 'etcd',
                                'service mesh', 'container orchestration'],
                'contextos': ['cloud native architecture', 'microservices platform', 'container orchestration']
            },
            'devsecops': {
                'palavras_base': ['security scanning', 'vulnerability assessment', 'penetration testing',
                                'sonarqube', 'owasp', 'security automation'],
                'contextos': ['security integration', 'secure development', 'security automation']
            },
            'api_management': {
                'palavras_base': ['rest', 'graphql', 'swagger', 'openapi', 'api gateway',
                                'api security', 'api versioning', 'api documentation'],
                'contextos': ['api development', 'api design', 'api security']
            },
            'testing': {
                'palavras_base': ['unit testing', 'integration testing', 'e2e testing', 'selenium',
                                'jest', 'pytest', 'test automation', 'quality assurance'],
                'contextos': ['software testing', 'test automation', 'quality assurance']
            },
            'container_orchestration': {
                'palavras_base': ['kubernetes', 'docker swarm', 'openshift', 'rancher', 'eks',
                                'aks', 'gke', 'container registry', 'pod', 'deployment'],
                'contextos': ['container management', 'orchestration platform', 'container deployment']
            },
            'serverless': {
                'palavras_base': ['aws lambda', 'azure functions', 'google cloud functions',
                                'faas', 'event-driven', 'api gateway', 'step functions'],
                'contextos': ['serverless architecture', 'function as service', 'event processing']
            },
            'blockchain': {
                'palavras_base': ['ethereum', 'hyperledger', 'smart contract', 'web3', 'solidity',
                                'consensus', 'distributed ledger', 'cryptocurrency'],
                'contextos': ['blockchain technology', 'distributed systems', 'smart contracts']
            },
            'quantum_computing': {
                'palavras_base': ['qbit', 'quantum gate', 'quantum circuit', 'quantum algorithm',
                                'quantum supremacy', 'quantum entanglement'],
                'contextos': ['quantum computing', 'quantum technology', 'quantum systems']
            },
            'edge_computing': {
                'palavras_base': ['edge', 'fog computing', 'iot gateway', 'edge analytics', 'edge ai',
                                'local processing', 'distributed computing', 'edge security'],
                'contextos': ['edge architecture', 'edge deployment', 'edge infrastructure']
            },
            'iot': {
                'palavras_base': ['sensors', 'actuators', 'mqtt', 'iot platform', 'embedded systems',
                                'raspberry pi', 'arduino', 'zigbee', 'bluetooth le'],
                'contextos': ['iot devices', 'iot architecture', 'iot protocols']
            },
            'big_data': {
                'palavras_base': ['hadoop', 'spark', 'data lake', 'data warehouse', 'etl',
                                'data pipeline', 'data processing', 'data analytics'],
                'contextos': ['big data processing', 'data architecture', 'data platform']
            },
            'data_engineering': {
                'palavras_base': ['airflow', 'kafka', 'data modeling', 'data pipeline', 'etl',
                                'data quality', 'data governance', 'data catalog'],
                'contextos': ['data infrastructure', 'data platform', 'data processing']
            },
            'mlops': {
                'palavras_base': ['model deployment', 'model monitoring', 'feature store', 'ml pipeline',
                                'model versioning', 'experiment tracking', 'model registry'],
                'contextos': ['mlops platform', 'ml infrastructure', 'ml deployment']
            },
            'gitops': {
                'palavras_base': ['argocd', 'flux', 'git workflow', 'infrastructure as code',
                                'declarative configuration', 'kubernetes operator'],
                'contextos': ['gitops workflow', 'gitops tools', 'gitops practices']
            },
            'sre': {
                'palavras_base': ['sli', 'slo', 'error budget', 'reliability', 'observability',
                                'incident management', 'chaos engineering'],
                'contextos': ['site reliability', 'service reliability', 'platform reliability']
            }
        }
        return self.temas_ti

    def gerar_dataset_ti(self):
        """Gera dataset expandido de palavras TI usando BERT"""
        console.print("\n📚 Gerando dataset de palavras TI...", style="bold yellow")
        
        palavras = []
        labels = []
        
        # Para cada tema
        for tema, conteudo in track(self.temas_ti.items(), description="Processando temas"):
            palavras_base = conteudo['palavras_base']
            contextos = conteudo['contextos']
            
            # Para cada palavra base
            for palavra in palavras_base:
                # Adiciona palavra original
                palavras.append(palavra)
                labels.append(tema)
                
                # Gera variações com BERT para cada contexto
                for contexto in contextos:
                    prompt = f"{contexto} {palavra}"
                    
                    # Gera palavras relacionadas
                    palavras_bert = self.gerar_palavras_bert(prompt, num_palavras=15)
                    
                    for palavra_bert in palavras_bert:
                        if len(palavra_bert.split()) <= 3:  # Limita tamanho das frases
                            palavras.append(f"{palavra} {palavra_bert}")
                            labels.append(tema)
                
                # Combinações entre palavras do mesmo tema
                for palavra2 in random.sample(palavras_base, min(3, len(palavras_base))):
                    if palavra != palavra2:
                        palavras.append(f"{palavra} {palavra2}")
                        labels.append(tema)
                        
                        # Gera variações da combinação
                        prompt = f"technology {palavra} {palavra2}"
                        palavras_bert = self.gerar_palavras_bert(prompt, num_palavras=5)
                        
                        for palavra_bert in palavras_bert:
                            if len(palavra_bert.split()) <= 3:
                                palavras.append(f"{palavra} {palavra2} {palavra_bert}")
                                labels.append(tema)
        
        self.df = pd.DataFrame({'texto': palavras, 'categoria': labels})
        console.print(f"✅ Dataset gerado com {len(self.df)} palavras!", style="bold green")
        
        # Salva dataset
        self.df.to_csv('dataset_ti_bert.csv', index=False)
        console.print("💾 Dataset salvo em 'dataset_ti_bert.csv'", style="bold blue")

    def processar_embeddings(self):
        """Processamento dos embeddings com feedback detalhado"""
        console.print("\n🧠 Processando embeddings...", style="bold yellow")
        
        with console.status("[bold green]Carregando modelo de embeddings..."):
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Processa em batches para melhor performance
        batch_size = 32
        total_batches = len(self.df) // batch_size + 1
        embeddings = []
        
        with console.status("[bold green]Gerando embeddings...") as status:
            for i in track(range(0, len(self.df), batch_size), total=total_batches):
                batch = self.df['texto'].iloc[i:i+batch_size].tolist()
                batch_embeddings = self.sentence_transformer.encode(batch)
                embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Normalização
        console.print("📊 Normalizando embeddings...", style="bold yellow")
        scaler = StandardScaler()
        self.embeddings_norm = scaler.fit_transform(embeddings)
        
        console.print("✅ Embeddings processados com sucesso!", style="bold green")

    def criar_clusters(self):
        """Criação de clusters usando FAISS com monitoramento"""
        console.print("\n🎯 Criando clusters...", style="bold yellow")
        
        n_clusters = 55  # Um cluster para cada tema
        dimensao = self.embeddings_norm.shape[1]
        
        # Configuração do FAISS
        console.print("⚙️ Configurando FAISS...", style="yellow")
        kmeans = faiss.Kmeans(
            d=dimensao,          # dimensionalidade
            k=n_clusters,        # número de clusters
            niter=300,          # número de iterações
            nredo=5,            # número de reinicializações
            verbose=True,       # feedback detalhado
            gpu=False           # CPU only
        )
        
        # Treinamento
        with console.status("[bold green]Treinando clusters..."):
            kmeans.train(self.embeddings_norm.astype(np.float32))
        
        # Atribuição de clusters
        _, self.labels = kmeans.index.search(self.embeddings_norm.astype(np.float32), 1)
        self.labels = self.labels.flatten()
        
        # Adiciona labels ao DataFrame
        self.df['cluster'] = self.labels
        
        console.print(f"✅ {n_clusters} clusters criados com sucesso!", style="bold green")

    def preparar_visualizacao(self):
        """Prepara dados para visualização 3D"""
        console.print("\n🎨 Preparando visualização 3D...", style="bold yellow")
        
        # Redução de dimensionalidade
        pca = PCA(n_components=3)
        self.coords_3d = pca.fit_transform(self.embeddings_norm)
        
        # Normalização para visualização
        self.coords_3d = (self.coords_3d - self.coords_3d.mean()) / self.coords_3d.std() * 5
        
        console.print("✅ Dados preparados para visualização!", style="bold green")

    def init_visualizacao(self):
        """Inicialização do sistema de visualização"""
        console.print("\n🎮 Iniciando sistema de visualização...", style="bold yellow")
        
        # Inicializa PyGame
        pygame.init()
        pygame.display.set_caption("Visualizador de Clusters TI - Elias Andrade")
        
        # Configuração do display
        flags = DOUBLEBUF | OPENGL | HWSURFACE
        self.screen = pygame.display.set_mode(self.display_size, flags)
        
        # Configuração inicial do OpenGL
        self.setup_opengl()
        
        # Cores para os 55 clusters (geradas algoritmicamente)
        self.cores_clusters = self.gerar_cores(55)
        
        console.print("✅ Sistema de visualização inicializado!", style="bold green")

    def setup_opengl(self):
        """Configuração do OpenGL"""
        # Configuração da perspectiva
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Configuração da luz
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Configuração do material
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Configuração da perspectiva
        gluPerspective(45, (self.display_size[0]/self.display_size[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, self.camera_config['translate_z'])

    def gerar_cores(self, n_cores):
        """Gera cores distintas para os clusters"""
        cores = []
        for i in range(n_cores):
            # Usa HSV para gerar cores bem distribuídas
            h = i / n_cores
            s = 0.8 + random.random() * 0.2  # Saturação alta
            v = 0.8 + random.random() * 0.2  # Valor alto
            
            # Converte HSV para RGB
            h_i = int(h * 6)
            f = h * 6 - h_i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            cores.append((r, g, b))
        
        return cores

    def renderizar(self):
        """Renderização da cena"""
        # Limpa buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Aplica transformações da câmera
        glTranslatef(0.0, 0.0, self.camera_config['translate_z'])
        glRotatef(self.camera_config['rotation_x'], 1, 0, 0)
        glRotatef(self.camera_config['rotation_y'], 0, 1, 0)
        glRotatef(self.camera_config['rotation_z'], 0, 0, 1)
        glScalef(self.camera_config['scale'], self.camera_config['scale'], self.camera_config['scale'])
        
        # Renderiza pontos
        self.renderizar_clusters()
        
        # Atualiza display
        pygame.display.flip()

    def renderizar_clusters(self):
        """Renderização dos clusters"""
        # Renderiza pontos
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i, (x, y, z) in enumerate(self.coords_3d):
            cluster = self.labels[i]
            glColor3fv(self.cores_clusters[cluster])
            glVertex3f(x, y, z)
        glEnd()
        
        # Renderiza conexões (opcional)
        if self.camera_config['scale'] > 1.5:
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for i, (x1, y1, z1) in enumerate(self.coords_3d):
                cluster1 = self.labels[i]
                for j in range(i+1, min(i+5, len(self.coords_3d))):
                    if self.labels[j] == cluster1:
                        x2, y2, z2 = self.coords_3d[j]
                        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        if dist < 2.0:
                            glColor4f(*self.cores_clusters[cluster1], 0.3)
                            glVertex3f(x1, y1, z1)
                            glVertex3f(x2, y2, z2)
            glEnd()

    def processar_eventos(self):
        """Processamento de eventos"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Rotação com mouse
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    self.camera_config['rotation_y'] += event.rel[0] * 0.5
                    self.camera_config['rotation_x'] += event.rel[1] * 0.5
            
            # Zoom com roda do mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Zoom in
                    self.camera_config['scale'] *= 1.1
                elif event.button == 5:  # Zoom out
                    self.camera_config['scale'] /= 1.1
            
            # Reset com ESPAÇO
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.reset_camera()
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True

    def reset_camera(self):
        """Reset da câmera para posição inicial"""
        self.camera_config = {
            'rotation_x': 0.0,
            'rotation_y': 0.0,
            'rotation_z': 0.0,
            'scale': 1.0,
            'translate_z': -30.0
        }

    def executar(self):
        """Loop principal"""
        console.print("\n🎮 Iniciando visualização...", style="bold green")
        running = True
        
        while running:
            running = self.processar_eventos()
            self.renderizar()
            pygame.time.wait(10)  # Controle de FPS
        
        pygame.quit()

    def processar_dados(self):
        """Pipeline completo de processamento de dados"""
        console.print("\n🔄 Iniciando pipeline de processamento...", style="bold yellow")
        
        try:
            # 1. Geração do dataset
            self.gerar_dataset_ti()
            
            # 2. Processamento dos embeddings
            self.processar_embeddings()
            
            # 3. Criação dos clusters
            self.criar_clusters()
            
            # 4. Preparação para visualização
            self.preparar_visualizacao()
            
            console.print("✅ Pipeline de processamento concluído!", style="bold green")
            
        except Exception as e:
            console.print(f"[bold red]❌ Erro no processamento: {str(e)}[/]")
            raise

    def analisar_resultados(self):
        """Análise dos resultados do processamento"""
        console.print("\n📊 Analisando resultados...", style="bold yellow")
        
        # Estatísticas gerais
        stats = {
            'total_palavras': len(self.df),
            'palavras_por_cluster': self.df['cluster'].value_counts().describe(),
            'distribuicao_temas': self.df['categoria'].value_counts()
        }
        
        # Exibe estatísticas
        console.print("\n📈 Estatísticas Gerais:")
        console.print(f"Total de palavras: {stats['total_palavras']}")
        console.print("\nDistribuição por cluster:")
        console.print(stats['palavras_por_cluster'])
        
        # Salva estatísticas
        with open('estatisticas_clusters.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        console.print("✅ Análise concluída e salva!", style="bold green")
        return stats

if __name__ == "__main__":
    try:
        console.print(Panel.fit(
            "[bold green]Visualizador 3D de Clusters TI com BERT[/]\n"
            "Por: Elias Andrade - Arquiteto de Soluções\n"
            "Replika AI - Maringá, PR",
            title="🚀 Iniciando Aplicação",
            border_style="green"
        ))
        
        start_time = time.time()
        
        # Inicialização e execução
        visualizador = ClusterizadorTIBert()
        visualizador.executar()
        
        # Análise final
        visualizador.analisar_resultados()
        
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
