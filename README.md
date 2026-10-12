# 🛣️ Auditoria de Calçadas com Visão Computacional — YOLOv8

**Checkpoint Integrado CP2 + CP3 | Inteligência Artificial e Visão Computacional**

Projeto acadêmico para identificação automatizada de problemas de infraestrutura e elementos de acessibilidade em calçadas, utilizando técnicas de visão computacional e detecção de objetos com YOLOv8

Alunos:

- Leticia Cristina Gandarez Resina RM98069
- João Rihan

## 📌 1. Sobre o projeto

A qualidade das calçadas influencia diretamente a mobilidade urbana, a segurança dos pedestres e a acessibilidade de pessoas com deficiência ou mobilidade reduzida.

Buracos, desníveis e problemas relacionados à infraestrutura podem representar riscos de acidentes e dificultar o deslocamento.

Este projeto propõe o desenvolvimento de um sistema de **auditoria visual de calçadas**, capaz de identificar automaticamente elementos de interesse a partir de imagens.

A solução utiliza um modelo de detecção de objetos da família YOLO (You Only Look Once), treinado com um dataset personalizado e preparado para futura integração com um aplicativo mobile.

### Objetivo geral

Desenvolver um protótipo de inteligência artificial capaz de identificar e localizar elementos de infraestrutura urbana em imagens de calçadas.

### Objetivos específicos

- Construir um dataset com pelo menos 60 imagens originais.
- Identificar e rotular pelo menos três classes de objetos.
- Preparar as anotações para treinamento de detecção de objetos.
- Treinar um modelo YOLOv8 com o dataset personalizado.
- Avaliar o desempenho utilizando métricas de detecção.
- Disponibilizar o modelo treinado, notebook, configurações e resultados no GitHub.
- Integrar o modelo a uma aplicação mobile para inferência utilizando a câmera traseira do dispositivo.

---

## 🏷️ 2. Classes de detecção

Foram definidas três classes:

| ID | Classe | Descrição |
|---|---|---|
| 0 | `buraco` | Cavidades ou danos na superfície da calçada |
| 1 | `desnivel` | Diferenças de altura ou irregularidades no pavimento |
| 2 | `piso_tatil` | Elementos de sinalização tátil utilizados na acessibilidade |

A escolha dessas classes considera tanto problemas que podem comprometer a segurança quanto elementos relacionados à acessibilidade urbana.

**Observação:** a detecção de piso tátil identifica sua presença visual, mas não determina, isoladamente, se sua instalação atende às normas de acessibilidade.

---

## 📸 3. Construção do dataset

### 3.1. Coleta e seleção

O dataset foi construído a partir de 60 imagens originais de calçadas, selecionadas para representar as classes de interesse.

As imagens foram organizadas e anotadas utilizando o Roboflow.

O notebook do projeto documenta as etapas de coleta, seleção e preparação dos dados.

### 3.2. Anotação

As anotações foram utilizadas para indicar a localização dos objetos nas imagens.

Parte das marcações foi realizada com polígonos, enquanto outra parte já estava no formato de caixas delimitadoras (*bounding boxes*).

Como o objetivo é treinar um detector de objetos, foi implementada uma etapa de conversão dos polígonos para caixas delimitadoras no padrão YOLO.

### 3.3. Pré-processamento e aumento de dados

A exportação do Roboflow utilizou:

- Redimensionamento das imagens para 512 × 512 pixels.
- Ajustes de brilho de até ±15%.
- Aumento de dados para ampliar a quantidade de exemplos de treinamento.
- Exportação no formato YOLOv8.

O aumento de dados gera variações das imagens originais e não equivale à coleta de novas fotografias independentes.

### 3.4. Distribuição do dataset

| Conjunto | Quantidade de imagens |
|---|---:|
| Treinamento | 129 |
| Validação | 11 |
| Teste | 6 |
| **Total exportado** | **146** |

### 3.5. Distribuição das anotações

| Classe | Treinamento | Validação | Teste |
|---|---:|---:|---:|
| `buraco` | 51 | 3 | 0 |
| `desnivel` | 21 | 0 | 2 |
| `piso_tatil` | 66 | 7 | 4 |
| **Total de objetos** | **138** | **10** | **6** |

As quantidades representam objetos anotados, não necessariamente imagens distintas.

### 3.6. Conversão das anotações

Durante a preparação do dataset, foram identificadas anotações poligonais incompatíveis com o formato de caixas delimitadoras esperado pelo treinamento de detecção.

Foi desenvolvido um procedimento em Python que:

1. Lê os arquivos de anotação exportados pelo Roboflow.
2. Identifica se cada anotação representa um polígono ou uma caixa delimitadora.
3. Calcula os limites mínimo e máximo das coordenadas dos polígonos.
4. Converte os limites para o formato YOLO: `class_id x_center y_center width height`.
5. Preserva as caixas delimitadoras já existentes.
6. Valida os IDs das classes e os intervalos das coordenadas.
7. Gera um novo dataset para treinamento.

**Resultados da conversão:**

- 106 polígonos convertidos.
- 48 caixas delimitadoras originais preservadas.
- 154 objetos anotados no dataset final.

---

## 🧠 4. Modelo de inteligência artificial

### 4.1. Arquitetura escolhida

Foi utilizado o **YOLOv8n (YOLOv8 Nano)**, da biblioteca Ultralytics.

Essa arquitetura foi escolhida por oferecer um modelo relativamente compacto, adequado à experimentação com detecção de objetos e com potencial de execução em dispositivos com recursos limitados.

O treinamento utilizou pesos pré-treinados, aplicando a técnica de *transfer learning* para adaptar o detector às três classes personalizadas.

### 4.2. Ambiente de desenvolvimento

- Google Colab.
- Python.
- PyTorch.
- Ultralytics YOLO.
- Roboflow.
- GPU NVIDIA Tesla T4.

### 4.3. Configurações do treinamento

| Parâmetro | Valor |
|---|---|
| Modelo base | YOLOv8n |
| Pesos iniciais | `yolov8n.pt` |
| Épocas | 50 |
| Resolução | 512 × 512 |
| Batch size | 8 |
| Otimizador | AdamW, selecionado automaticamente |
| Paciência para early stopping | 12 |
| Seed | 42 |
| Classes | 3 |
| GPU | Tesla T4 |

O treinamento foi concluído após 50 épocas.

O melhor checkpoint foi salvo como `best.pt`.

---

## 📊 5. Avaliação do modelo

A avaliação foi realizada utilizando métricas tradicionais de detecção de objetos.

### 5.1. Métricas utilizadas

**Precision:** proporção das detecções positivas que estão corretas.

**Recall:** proporção dos objetos reais que foram identificados.

**mAP@50:** média da precisão média por classe, considerando uma interseção sobre união (IoU) de 0,50.

**mAP@50-95:** média da precisão média em diferentes limites de IoU, de 0,50 a 0,95.

### 5.2. Resultados gerais

| Métrica | Validação | Teste |
|---|---:|---:|
| Precision | 95,7% | 29,5% |
| Recall | 81,8% | 37,5% |
| mAP@50 | 83,0% | 33,2% |
| mAP@50-95 | 54,6% | 24,6% |

Os resultados mostram desempenho superior na validação em comparação com o conjunto de teste.

Essa diferença evidencia a necessidade de ampliar a diversidade dos dados e realizar avaliações adicionais.

### 5.3. Resultados por classe no teste

| Classe | mAP@50 | mAP@50-95 |
|---|---:|---:|
| `desnivel` | 0,0% | 0,0% |
| `piso_tatil` | 66,4% | 49,2% |
| `buraco` | Não avaliado | Não avaliado |

A classe `buraco` não possui exemplos anotados no conjunto de teste, impossibilitando sua avaliação nesse conjunto.

A classe `desnivel` apresentou desempenho insuficiente nos exemplos de teste.

### 5.4. Limitações identificadas

O projeto apresenta algumas limitações relevantes:

- O dataset original possui apenas 60 imagens.
- O conjunto de teste contém somente seis imagens.
- A classe `desnivel` não possui exemplos na validação.
- A classe `buraco` não possui exemplos no teste.
- Existe desequilíbrio entre as quantidades de exemplos das classes.
- Os resultados de validação não garantem desempenho equivalente em situações reais.
- A capacidade de generalização para diferentes ambientes, condições de iluminação e tipos de calçada ainda precisa ser investigada.

Dessa forma, o modelo deve ser considerado **um protótipo acadêmico experimental**, não um sistema validado para auditoria urbana em produção.

---

## 📁 6. Organização do repositório

Estrutura prevista do projeto:

    auditoria-calcadas-yolov8/
    │
    ├── notebooks/
    │   └── auditoria_calcadas_yolov8.ipynb
    │
    ├── dataset/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── valid/
    │   │   ├── images/
    │   │   └── labels/
    │   └── test/
    │       ├── images/
    │       └── labels/
    │
    ├── models/
    │   └── best.pt
    │
    ├── results/
    │   ├── training/
    │   └── test/
    │
    ├── config/
    │   └── data.yaml
    │
    ├── README.md
    └── requirements.txt

O notebook documenta o processamento dos dados e o treinamento. Os diretórios de resultados contêm as métricas e os gráficos gerados pelo Ultralytics.

A estrutura poderá ser ampliada com os arquivos da aplicação mobile.

---

## ⚙️ 7. Como reproduzir o treinamento

### 7.1. Requisitos

- Python.
- PyTorch.
- Ultralytics.
- PyYAML.
- Dataset anotado no formato YOLO.
- GPU compatível, recomendada para treinamento.

### 7.2. Instalação

Clone o repositório:

`git clone https://github.com/letyresina/auditoria-calcadas-yolov8.git`

Entre na pasta:

`cd auditoria-calcadas-yolov8`

Instale as dependências:

`pip install -r requirements.txt`

### 7.3. Execução pelo Google Colab

1. Abra o notebook disponível no repositório.
2. Configure o ambiente para utilizar GPU.
3. Execute as células de preparação.
4. Forneça o ZIP original do Roboflow quando solicitado.
5. Execute a etapa de conversão das anotações.
6. Execute o treinamento do YOLOv8.
7. Consulte as métricas e os gráficos produzidos.

O notebook utiliza caminhos próprios do Google Colab, como `/content/auditoria_yolo`, e deve ser executado seguindo a sequência de células.

A configuração disponibilizada em `config/data.yaml` descreve o dataset organizado no repositório. Para executar o treinamento fora do Colab, pode ser necessário ajustar os caminhos conforme o ambiente local.

---

## 🔍 8. Inferência com o modelo treinado

O arquivo `models/best.pt` contém os pesos do melhor modelo obtido durante o treinamento.

Exemplo de utilização em Python:

    from ultralytics import YOLO

    modelo = YOLO("models/best.pt")

    resultados = modelo.predict(
        source="imagem_teste.jpg",
        imgsz=512,
        conf=0.25,
        save=True
    )

Esse exemplo realiza inferência em uma imagem fornecida pelo usuário e salva a visualização das detecções.

O arquivo `imagem_teste.jpg` deve ser substituído pelo caminho de uma imagem existente.

---

## 📱 9. Integração com aplicativo mobile

**Status: etapa pendente de implementação e validação.**

O objetivo desta etapa é disponibilizar o modelo de detecção em um aplicativo para dispositivos móveis.

O aplicativo deverá utilizar a câmera traseira do smartphone para identificar os elementos de interesse em tempo real.

### Requisitos previstos

- Acesso à câmera traseira.
- Captura contínua de quadros.
- Execução do modelo de detecção.
- Exibição de caixas delimitadoras.
- Identificação das classes detectadas.
- Apresentação das respectivas pontuações de confiança.
- Funcionamento em dispositivo físico para demonstração.

### Aplicação de referência

Repositório indicado como base para a integração:

https://github.com/Mjrovai/yoloinferApp

### Exportação do modelo

O treinamento gerou um checkpoint PyTorch (`best.pt`).

Para execução mobile, será necessário verificar o formato de modelo aceito pelo aplicativo de referência e realizar a exportação apropriada.

A compatibilidade, a velocidade de inferência e o funcionamento da câmera ainda deverão ser testados.

---

## 📋 10. Requisitos do Checkpoint CP2 + CP3

A atividade possui valor total de **20 pontos**, distribuídos entre as seguintes etapas:

| Critério | Pontuação | Situação |
|---|---:|---|
| Dataset e rotulagem | 4 pontos | Concluído |
| Treinamento, métricas e exportação | 6 pontos | Treinamento e avaliação concluídos; exportação mobile pendente |
| Deploy mobile | 6 pontos | Pendente |
| Organização, GitHub e apresentação | 4 pontos | GitHub em organização; apresentação pendente |
| **Total** | **20 pontos** | **Projeto em desenvolvimento** |

A entrega prevê apresentação presencial e demonstração do funcionamento da solução em dispositivo móvel.

**Data prevista de entrega e apresentação: 13/10/2026.**

---

## 🚀 11. Melhorias futuras

Como possíveis evoluções do projeto, destacam-se:

- Ampliar a quantidade de imagens originais.
- Coletar exemplos em diferentes bairros e condições ambientais.
- Melhorar o equilíbrio entre as classes.
- Revisar a separação entre treinamento, validação e teste, evitando vazamento de imagens relacionadas.
- Melhorar a detecção de desníveis.
- Avaliar outras arquiteturas e configurações do YOLO.
- Medir o tempo de inferência em dispositivos móveis.
- Investigar a execução offline.
- Adicionar registro de localização das ocorrências.
- Desenvolver uma interface para consulta de problemas identificados.

---

## 👥 12. Integrantes

**Instituição:** FIAP

**Curso:** Engenharia de Software

**Atividade:** Checkpoint Integrado CP2 + CP3

**Equipe:**

- Leticia Resina
- [Adicionar integrante]
- [Adicionar integrante]
- [Adicionar integrante]
- [Adicionar integrante]

**Professor:** [Adicionar nome do professor]

---

## 📚 13. Referências

- Ultralytics — Documentação oficial do YOLO: https://docs.ultralytics.com/
- Roboflow — Plataforma de preparação e anotação de datasets: https://roboflow.com/
- PyTorch — Framework de aprendizado profundo: https://pytorch.org/
- Google Colab — Ambiente de execução de notebooks: https://colab.research.google.com/
- Projeto de referência para inferência mobile: https://github.com/Mjrovai/yoloinferApp

---

## 📄 14. Considerações finais

O projeto demonstrou a viabilidade técnica de desenvolver um detector de objetos personalizado para análise visual de calçadas utilizando YOLOv8.

Foram realizadas a preparação do dataset, a conversão de anotações, o treinamento do modelo e a avaliação quantitativa.

Os resultados evidenciaram desempenho promissor em parte dos exemplos, mas também limitações importantes de generalização, especialmente na classe `desnivel`.

A continuidade do trabalho envolve a integração com uma aplicação mobile, a realização de testes em dispositivo físico e o aprimoramento do conjunto de dados.

**Este projeto tem finalidade acadêmica e experimental.**
