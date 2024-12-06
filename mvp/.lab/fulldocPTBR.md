# TVTxMindVision - Detecção de Pose e Reconhecimento de Golpes

## 1. Introdução

### 1.1 Objetivo
Esta documentação fornece uma visão geral completa do TVTxMindVision, uma solução de software projetada para detecção de pose em tempo real e classificação de golpes no tênis, utilizando MediaPipe e OpenCV. O aplicativo suporta tanto vídeos enviados quanto feeds ao vivo da webcam.

### 1.2 Escopo
O TVTxMindVision é voltado para aplicações de treinamento de tênis, permitindo que os usuários detectem e classifiquem golpes de tênis (Saque, Forehand, Backhand) com base na análise de pose corporal. O aplicativo também permite que os usuários baixem os dados de marcos e os vídeos gravados.

### 1.3 Definições, Acrônimos e Abreviações
- **CUDA**: Compute Unified Device Architecture (Arquitetura Unificada de Computação)
- **Marcos de Pose**: Pontos-chave do corpo usados para analisar o movimento.
- **Tipo de Golpe**: Classificação de golpe no tênis (Saque, Forehand, Backhand).

## 2. Requisitos Funcionais

### 2.1 Funcionalidades do Sistema
1. **Detecção de Pose**: Detecção de marcos de pose em tempo real.
2. **Reconhecimento de Golpes**: Classificação em tempo real de golpes de tênis.
3. **Exportação de Dados**: Exportação de dados de marcos em formato `.pkl` e vídeos gravados em formato `.avi`.

### 2.2 Requisitos do Usuário
1. Um computador com webcam para gravação ao vivo.
2. Opcionalmente, uma GPU compatível com CUDA para otimização do processamento.

## 3. Instruções para o Usuário

### 3.1 Como Começar
1. **Selecione a Fonte de Vídeo**:


   - **Webcam ao Vivo**: Grave um vídeo ao vivo com a sua webcam.
   - **Envio de Vídeo**: Envie um arquivo de vídeo previamente gravado.

2. **Processamento e Detecção**:
   - O aplicativo detectará os marcos de pose e classificará os golpes como Saque, Forehand ou Backhand.

3. **Exportação de Dados**:
   - Após o processamento, baixe os dados de marcos de pose e o vídeo gravado diretamente da barra lateral.

## 4. Testes

### 4.1 Testes Realizados
- **Teste de Detecção de Golpes**: A detecção de Saque, Forehand e Backhand foi validada com uma variedade de vídeos.
- **Testes de Interface**: Verificamos a funcionalidade de upload e gravação de vídeo, bem como a opção de download de arquivos.

## 5. Conclusão

TVTxMindVision oferece uma solução inovadora para treinadores e jogadores de tênis, permitindo a análise detalhada dos golpes com base na detecção de pose e no uso de técnicas de visão computacional.
