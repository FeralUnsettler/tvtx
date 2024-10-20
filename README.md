# **Tênis Vídeo Treino (TVTx)**

## **Resumo do Projeto**
O *TVTx* é uma plataforma de análise de desempenho em tênis usando visão computacional e aprendizado de máquina. A solução visa fornecer feedback em tempo real para jogadores de tênis ao capturar, analisar e avaliar golpes como Forehand, Backhand e Saque, utilizando tecnologias como OpenCV, PyTorch e Streamlit. Isso otimiza o treinamento esportivo, oferecendo aos usuários uma visão objetiva de sua performance.

## **Definição do Problema**
O treinamento de tênis muitas vezes é baseado em avaliações subjetivas e limitadas de treinadores. Falta uma ferramenta automatizada e precisa que permita a análise de desempenho técnico em tempo real. Projetos semelhantes já existem, mas apresentam limitações em termos de feedback imediato e personalização, além de não integrarem o uso de visão computacional com aprendizado profundo. 

Referências técnicas e bibliográficas sobre o estado da arte em tecnologias de análise de vídeo e aprendizado de máquina, como *YOLO* e *OpenCV*, foram consultadas. Comparações com soluções como o *Coach's Eye* revelaram a necessidade de melhorias em feedback automatizado e acessibilidade dos dados.

## **Objetivos**
- **Objetivo Geral:** Desenvolver uma plataforma que utiliza visão computacional e aprendizado de máquina para analisar, em tempo real, o desempenho técnico de jogadores de tênis.
  
- **Objetivos Específicos:**
  1. Implementar captura de vídeo em tempo real com feedback imediato.
  2. Utilizar PyTorch e YOLO para reconhecimento de golpes de tênis.
  3. Desenvolver uma interface simples e acessível para visualização dos resultados.

## **Stack Tecnológico**
O desenvolvimento do *TVTx* utiliza tecnologias que otimizam a análise de vídeo em tempo real:
- **OpenCV**: Utilizado para capturar e processar os vídeos dos jogadores.
- **PyTorch**: Framework de aprendizado de máquina, utilizado para aplicar modelos treinados no reconhecimento de golpes.
- **Streamlit**: Ferramenta para construção rápida de uma interface interativa e visualização dos resultados.
- **PostgreSQL**: Banco de dados relacional para armazenar usuários, vídeos e análises.
  
Essas tecnologias foram escolhidas pela eficiência e capacidade de escalabilidade, conforme documentado em suas páginas oficiais.

## **Descrição da Solução**
A solução captura vídeos em tempo real, processando-os para identificar golpes e fornecer feedback. O sistema emprega modelos de aprendizado profundo treinados para reconhecer padrões de movimento e classificar o tipo de golpe. O feedback é gerado instantaneamente, permitindo que o usuário veja a análise de sua performance no momento da execução do golpe.

A integração com Streamlit garante uma interface amigável, onde o jogador pode acompanhar sua evolução em gráficos e relatórios. Além disso, o sistema utiliza técnicas avançadas de detecção de objetos para fornecer feedback preciso sobre os aspectos técnicos de cada golpe.

## **Arquitetura**
A arquitetura do sistema segue uma abordagem modular, com camadas que separam as funcionalidades principais:
- **Captura de vídeo** com OpenCV.
- **Processamento de dados** através do modelo YOLO em PyTorch.
- **Visualização** em tempo real via Streamlit.
  
Diagrama de camadas pode ser encontrado no repositório do projeto, com artefatos como ER diagrams, wireframes e o Project Model Canvas.

## **Validação**
A validação será realizada através de testes com jogadores reais, que utilizarão a plataforma para avaliar sua performance durante as sessões de treino. Os resultados serão comparados com feedbacks de treinadores, além de entrevistas e questionários para medir a usabilidade e precisão do sistema.

---

Para mais detalhes e artefatos adicionais, como benchmarks, documentos de requisitos e diagramas, consulte os repositórios:

1. [Tênis Vídeo Treino - Plataforma de Análise de Desempenho no Tênis Utilizando Visão Computacional](https://github.com/FeralUnsettler/tvtx/blob/main/paper_senacLab.md)
2. [BMDS_Vision.md](https://github.com/FeralUnsettler/tvtx/blob/main/BMDS_Vision.md)
3. [TVTx Framework](https://github.com/FeralUnsettler/tvtx/blob/main/p2mvp_framework.md)
4. [Benchmark App](https://github.com/FeralUnsettler/tvtx/blob/main/benchmark.md)


