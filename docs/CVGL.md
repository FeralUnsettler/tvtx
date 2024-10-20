## Visão computacional com OpenCV em Python

* ### [Leitura e processamento de imagens, vídeos e webcam](#2-Leitura-e-processamento-de-imagens-vídeos-e-webcam)
* ### [Transformações geométricas e filtragem de imagens](#3-Transformações-geométricas-e-filtragem-de-imagens)
* ### [Detecção de contornos](#4-Detecção-de-contornos)
* ### [Transformações em canais de cor](#5-Transformações-em-canais-de-cor)
* ### [Técnicas avançadas: blur, operações bitwise, masks, thresholding](#6-Técnicas-avançadas-blur-operações-bitwise-masks-thresholding)
---
### 1. Instalando OpenCV
Certifique-se de ter o OpenCV instalado. Você pode instalá-lo usando o pip:

```bash
pip install opencv-python
```

### 2. Leitura e processamento de imagens, vídeos e webcam
Vamos começar com a leitura e processamento de diferentes tipos de mídia.

#### Leitura de uma imagem:

```python
import cv2

# Carregar uma imagem
image = cv2.imread('image.jpg')

# Mostrar a imagem
cv2.imshow('Imagem', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Leitura de um vídeo:

```python
# Carregar um vídeo
video = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Mostrar o vídeo frame a frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
```

#### Utilizando a webcam:

```python
# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mostrar o vídeo da webcam
    cv2.imshow('Webcam', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
#### [Index](#Visão-computacional-com-OpenCV-em-Python)
### 3. Transformações geométricas e filtragem de imagens
Vamos agora realizar transformações geométricas e filtragem de imagens.

#### Redimensionamento de imagens:

```python
# Redimensionar a imagem
resized_image = cv2.resize(image, (new_width, new_height))
```

#### Transformações geométricas (rotação):

```python
# Rotação da imagem
rows, cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
rotated_image = cv2.warpAffine(image, M, (cols, rows))
```

#### Filtragem de imagens (blur):

```python
# Aplicar um filtro de blur
blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```
#### [Index](#Visão-computacional-com-OpenCV-em-Python)
### 4. Detecção de contornos
Vamos agora detectar contornos em uma imagem.

```python
# Converter a imagem para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar um filtro de Canny para detecção de bordas
edges = cv2.Canny(gray_image, threshold1, threshold2)

# Encontrar contornos na imagem
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar os contornos na imagem original
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness)
```
#### [Index](#Visão-computacional-com-OpenCV-em-Python)
### 5. Transformações em canais de cor
Vamos realizar algumas transformações nos canais de cor da imagem.

#### Separação de canais de cor:

```python
# Separar os canais de cor
b, g, r = cv2.split(image)
```

#### Mesclagem de canais de cor:

```python
# Mesclar os canais de cor
merged_image = cv2.merge((b, g, r))
```
#### [Index](#Visão-computacional-com-OpenCV-em-Python)
### 6. Técnicas avançadas: blur, operações bitwise, masks, thresholding
Agora, vamos explorar algumas técnicas avançadas.

#### Operações bitwise:

```python
# Criar uma máscara
mask = cv2.inRange(image, lower_bound, upper_bound)

# Aplicar uma operação bitwise
result = cv2.bitwise_and(image1, image2, mask=mask)
```

#### Thresholding:

```python
# Aplicar thresholding
ret, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
```

Este tutorial abrange os conceitos básicos e avançados de visão computacional com OpenCV em Python. Espero que isso seja útil para você! Se tiver mais dúvidas ou precisar de mais exemplos, não hesite em perguntar.
#### [Index](#Visão-computacional-com-OpenCV-em-Python)
