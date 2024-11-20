# Documentação Completa: **Tênis Vídeo Treino (TVTx)**

---

## **1. Introdução**
O **TVTx** é uma plataforma inovadora que utiliza visão computacional e aprendizado de máquina para análise de desempenho técnico no tênis. Com suporte para reconhecimento de golpes como *Forehand*, *Backhand* e *Saque*, o sistema oferece feedback em tempo real, otimizando o treinamento esportivo e elevando o nível técnico dos jogadores.

---

### **1.1. Objetivo**
O objetivo principal do **TVTx** é melhorar a experiência de aprendizado no tênis ao fornecer feedback automatizado e preciso sobre os golpes. Isso é alcançado com o uso de algoritmos avançados de aprendizado profundo e visão computacional, integrados a uma interface amigável.

### **1.2. Escopo**
A plataforma é destinada a:
- **Jogadores amadores e profissionais**: Para aprimorar técnicas com base em dados objetivos.
- **Treinadores**: Para obter uma análise técnica detalhada dos alunos.
- **Clubes e academias esportivas**: Para integrar tecnologia aos treinamentos.

### **1.3. Definições, Acrônimos e Abreviações**
- **TVTx**: Tênis Vídeo Treino.
- **GPU**: Unidade de Processamento Gráfico.
- **ML**: *Machine Learning* (Aprendizado de Máquina).
- **YOLO**: *You Only Look Once*, algoritmo de detecção de objetos.
- **API Pose Detection**: Sistema de identificação de marcos corporais.

---

## **2. Definição do Problema**
O treinamento esportivo no tênis ainda depende, em sua maioria, de avaliações subjetivas de treinadores. Essa abordagem pode resultar em erros ou inconsistências. Ferramentas existentes oferecem funcionalidades limitadas em feedback automatizado e personalização, criando uma lacuna para soluções mais avançadas, acessíveis e precisas.

---

## **3. Arquitetura do Sistema**
O **TVTx** foi projetado com uma arquitetura modular, garantindo escalabilidade e flexibilidade.

### **3.1. Camadas Principais**
1. **Captura de Dados**:
   - Usa **OpenCV** para captura de vídeos ao vivo ou pré-gravados.
   - Integra-se com câmeras e webcams.
   
2. **Processamento de Dados**:
   - **YOLO e PyTorch**: Modelos treinados para detecção de movimentos e classificação de golpes.
   - **MediaPipe**: Utilizado para análise detalhada da pose corporal.

3. **Interface e Feedback**:
   - **Streamlit**: Apresentação de resultados em tempo real.
   - Relatórios detalhados com gráficos e vídeos processados.

4. **Armazenamento**:
   - **PostgreSQL**: Banco de dados para armazenar perfis de usuários, estatísticas e vídeos processados.

---

## **4. Funcionalidades do Sistema**

### **4.1. Funcionalidades Principais**
- **Detecção de Pose**: Identificação de marcos corporais para análise de movimento.
- **Classificação de Golpes**:
  - Reconhecimento de *Forehand*, *Backhand* e *Saque*.
- **Feedback em Tempo Real**:
  - Relatórios instantâneos sobre a performance técnica.
- **Exportação de Dados**:
  - Salvamento de vídeos processados e estatísticas em formatos populares como `.avi` e `.pkl`.

### **4.2. Funcionalidades Adicionais**
- **Comparação com Referências Profissionais**:
  - Disponível na versão PREMIUM.
- **Dashboard Interativa**:
  - Histórico de treinos e evolução técnica.

---

## **5. Requisitos do Sistema**

### **5.1. Requisitos de Hardware**
- CPU com suporte para processamento em tempo real.
- GPU compatível com CUDA para otimização de desempenho (opcional).
- Webcam ou câmera externa.

### **5.2. Requisitos de Software**
- **Python 3.10+**
- Bibliotecas:
  - OpenCV
  - PyTorch
  - MediaPipe
  - Streamlit
  - PostgreSQL

---

## **6. Fluxo de Uso**

1. **Início do Sistema**:
   - O usuário escolhe entre gravação ao vivo ou upload de vídeo.
  ![tvtxmindvision1](https://github.com/user-attachments/assets/9e034996-7237-4b33-803d-82a1b8ffa900)

    ![tvtxmindvision2](https://github.com/user-attachments/assets/77810442-d3be-4f82-bc62-833427942f3a)
 
   
2. **Análise de Golpes**:
   - O sistema processa o vídeo, detecta a pose corporal e classifica os golpes.
  
   ![Screenshot from 2024-11-20 07-51-24](https://github.com/user-attachments/assets/0a527c83-7960-42f8-a4dc-f0957ac8caeb)
  

3. **Visualização de Resultados**:
   - Feedback em tempo real exibido na interface Streamlit.
   - Relatórios detalhados salvos para download.

4. **Histórico de Treinos**:
   - Usuários PREMIUM podem acessar e comparar resultados de sessões anteriores.

---

## **7. Testes e Validação**

### **7.1. Estratégia de Testes**
- **Testes Funcionais**:
  - Verificação de detecção de golpes em diferentes ângulos e velocidades.
- **Testes de Usabilidade**:
  - Validação com grupos de usuários para medir a eficácia da interface.
- **Testes de Desempenho**:
  - Avaliação da latência do feedback em tempo real.

### **7.2. Resultados Esperados**
- Precisão acima de 90% na detecção e classificação de golpes.
- Redução significativa no tempo de análise comparado a métodos manuais.

---

## **8. Conclusão**
O **TVTx** é uma solução poderosa e acessível para jogadores e treinadores de tênis que desejam elevar seu nível técnico com o auxílio de tecnologias modernas. Seu diferencial reside no feedback em tempo real e na integração de aprendizado de máquina com visão computacional.

---

## **9. Referências**
- Documentação oficial do OpenCV, PyTorch e MediaPipe.
- Artigos sobre visão computacional e aprendizado profundo aplicados ao esporte.

---

**Repositórios Relacionados**:
- [Repositório TVTx no GitHub](https://github.com/FeralUnsettler/tvtx)

**Contato**:
Luciano Martins Fagundes | [LinkedIn](https://www.linkedin.com/in/luxxmf/)
