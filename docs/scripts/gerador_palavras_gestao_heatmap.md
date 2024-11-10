# Gerador de Palavras de Gestão com Mapa de Calor - Documentação

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**

**Versão:** 1.0.0 (Micro-revisão 000000001) - 2024-11-06

## Visão Geral

Este documento descreve o script `gerador_palavras_gestao_heatmap.py`, que gera aproximadamente 14.000 palavras relacionadas à gestão empresarial usando um modelo de linguagem GPT-2 e visualiza o progresso da geração em um mapa de calor em tempo real usando Pygame e OpenGL.  O sistema combina geração de texto com visualização gráfica para fornecer uma representação dinâmica do processo de geração de conteúdo.

## Arquitetura

O script é composto pelas seguintes partes principais:

1. **Inicialização:** Inicializa o modelo GPT-2 (`pierreguillou/gpt2-small-portuguese`), o tokenizer correspondente e configura o ambiente Pygame/OpenGL para a visualização do mapa de calor.

2. **Geração de Prompts:** Gera prompts aleatórios combinando palavras-chave de diferentes temas relacionados à gestão (gestão, financeiro, processos).

3. **Geração de Texto (GPT-2):**  Utiliza o modelo GPT-2 para gerar texto baseado nos prompts gerados.  Os parâmetros de geração (temperatura, top_k, top_p) são ajustados para controlar a criatividade e a coerência do texto gerado.

4. **Atualização do Mapa de Calor:**  Atualiza um mapa de calor em tempo real, representando o progresso da geração de palavras.  A cor do mapa varia gradualmente do vermelho para o verde, indicando o progresso da geração.

5. **Renderização (Pygame/OpenGL):**  Utiliza Pygame e OpenGL para renderizar o mapa de calor na tela.  A visualização é atualizada a cada frame, mostrando o progresso dinamicamente.

6. **Salvamento de Palavras:**  Salva as palavras geradas em um arquivo de texto.

## Funcionalidades Principais

- **Geração de Texto com GPT-2:** Gera texto criativo e coerente sobre gestão empresarial.
- **Visualização Dinâmica:** Mostra o progresso da geração em tempo real através de um mapa de calor.
- **Salvamento de Palavras:** Salva as palavras geradas em um arquivo de texto para posterior análise.

## Tecnologias Utilizadas

- **Python:** Linguagem de programação principal.
- **Transformers:** Biblioteca para o modelo GPT-2.
- **Pygame:** Biblioteca para desenvolvimento de jogos, usada para a interface gráfica.
- **OpenGL:** Biblioteca para gráficos 2D.
- **Rich:** Biblioteca para saída de console aprimorada.

## Exemplo de Uso

O script é executado diretamente: `python gerador_palavras_gestao_heatmap.py`

## Considerações Finais

Este script demonstra uma abordagem inovadora para a geração e visualização de texto, combinando geração de linguagem natural com visualização gráfica.  A visualização em tempo real fornece um feedback imediato sobre o progresso da geração, tornando o processo mais transparente e envolvente.


---

**Elias Andrade - Arquiteto de Soluções - Replika AI - Maringá, PR**
