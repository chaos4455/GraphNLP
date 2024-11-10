# Script de Organização de Arquivos do Projeto
# Autor: Elias Andrade
# Data: 2024-03-27

# Criação das pastas principais
$pastas = @(
    "src/core",              # Núcleo do sistema
    "src/visualizers",       # Visualizadores 3D
    "src/generators",        # Geradores de palavras
    "src/clustering",        # Algoritmos de clustering
    "src/utils",             # Utilitários
    "src/models",            # Modelos e classes base
    "assets/images",         # Imagens geradas
    "assets/data",           # Dados e datasets
    "docs/scripts",          # Documentação dos scripts
    "tests"                  # Testes unitários
)

# Cria as pastas
foreach ($pasta in $pastas) {
    New-Item -Path $pasta -ItemType Directory -Force
    Write-Host "✅ Pasta criada: $pasta" -ForegroundColor Green
}

# Mapeamento de arquivos para pastas
$movimentos = @{
    "src/models" = @(
        "cluster_palavras_tech_faiss.py",
        "cluster_palavras_ti_bert.py"
    )
    "src/clustering" = @(
        "cluster_palavras_tech_faiss_v2.py",
        "cluster_palavras_tech_faiss_v3.py",
        "cluster_palavras_grande.py",
        "cluster_3d_hierarquico.py"
    )
    "src/visualizers" = @(
        "cluster_palavras_mesh3d.py",
        "cluster_palavras_visualizador.py",
        "cluster_words_3d_cloud.py",
        "visualizador_perceptron_3d.py",
        "visualizador_transformer_3d.py",
        "bert_orbital_words_3d.py"
    )
    "src/generators" = @(
        "gerador_palavras_gestao_heatmap.py",
        "word_generation_heatmap.py"
    )
    "assets/data" = @(
        "analise_clusters.json",
        "clusters_palavras.csv",
        "dataset_ti_bert.csv",
        "estatisticas_clusters.json",
        "palavras_gestao_20241109_220859.txt"
    )
    "assets/images" = @(
        "*.png",
        "*.html"
    )
    "docs/scripts" = @(
        "*.md"
    )
}

# Move os arquivos para suas respectivas pastas
foreach ($pasta in $movimentos.Keys) {
    foreach ($arquivo in $movimentos[$pasta]) {
        if ($arquivo -like "*.png" -or $arquivo -like "*.html") {
            Move-Item -Path $arquivo -Destination $pasta -Force -ErrorAction SilentlyContinue
        } else {
            if (Test-Path $arquivo) {
                Move-Item -Path $arquivo -Destination $pasta -Force
                Write-Host "📦 Arquivo movido: $arquivo -> $pasta" -ForegroundColor Yellow
            }
        }
    }
}

# Cria arquivo README.md na raiz
$readmeContent = @"
# Projeto de Visualização 3D de Clusters de Palavras

**Elias Andrade - Arquiteto de Soluções**
**Replika AI - Maringá, PR**

## Estrutura do Projeto

- 📁 src/
  - 📁 core/         - Núcleo do sistema
  - 📁 visualizers/  - Visualizadores 3D
  - 📁 generators/   - Geradores de palavras
  - 📁 clustering/   - Algoritmos de clustering
  - 📁 utils/        - Utilitários
  - 📁 models/       - Modelos e classes base
- 📁 assets/
  - 📁 images/       - Imagens geradas
  - 📁 data/         - Dados e datasets
- 📁 docs/
  - 📁 scripts/      - Documentação dos scripts
- 📁 tests/          - Testes unitários

## Instalação

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Execução

Consulte a documentação específica de cada módulo em docs/scripts/
"@

$readmeContent | Out-File -FilePath "README.md" -Encoding utf8
Write-Host "📝 README.md criado" -ForegroundColor Cyan

Write-Host "`n✨ Organização concluída!" -ForegroundColor Green