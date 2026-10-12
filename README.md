# Auditoria de Calçadas — YOLOv8

Projeto acadêmico de visão computacional para identificação
de elementos e problemas em calçadas.

## Classes
- buraco
- desnivel
- piso_tatil

## Dataset
- 60 imagens originais
- 129 imagens de treinamento após aumento de dados
- 11 imagens de validação
- 6 imagens de teste

## Treinamento
- YOLOv8n
- 50 épocas
- Resolução 512x512
- GPU Tesla T4

## Resultados
- mAP@50 validação: 0.830
- mAP@50 teste: 0.332
- mAP@50-95 teste: 0.246

## Limitações
- A classe desnivel não aparece na validação.
- A classe buraco não aparece no teste.
- O desempenho em desnivel no teste foi insuficiente.
- O conjunto de teste é pequeno.

## Arquivos
- notebooks/: código completo do Colab
- dataset/: imagens e anotações YOLO
- models/best.pt: modelo treinado
- results/: gráficos e avaliações
- config/data.yaml: configuração das classes

## Próxima etapa
Integração com aplicativo mobile para inferência.
