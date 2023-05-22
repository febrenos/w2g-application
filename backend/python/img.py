import cv2
import urllib.request
import numpy as np
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
# pip install azure-cognitiveservices-vision-computervision
# pip install opencv-python
# pip install opencv-python-headless
# pip install numpy

chave = 'xxxxxxxxx'
ponto_extremidade = 'https://know-image-rm94170.cognitiveservices.azure.com/'
url_imagem = 'https://media.istockphoto.com/id/512990032/pt/foto/lobos-na-neve.jpg?s=1024x1024&w=is&k=20&c=UNV20XdhvAmS7-WZKmyhGQHWwR1QPTRw2xICj_hmKps='

cliente = ComputerVisionClient(ponto_extremidade, CognitiveServicesCredentials(chave))

# Análise por visão computacional
cs_imagem = cliente.analyze_image(url_imagem, visual_features=[VisualFeatureTypes.tags, VisualFeatureTypes.description])

# Carregue a imagem usando urllib.request
response = urllib.request.urlopen(url_imagem)
array_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)
cv2_imagem = cv2.imdecode(array_bytes, cv2.IMREAD_COLOR)

# Extraia os canais de cores (B, G, R)
canal_azul, canal_verde, canal_vermelho = cv2.split(cv2_imagem)

# Calcule as médias dos canais de cores
media_azul = canal_azul.mean()
media_verde = canal_verde.mean()
media_vermelho = canal_vermelho.mean()

if media_vermelho > media_azul:
    temperatura = 'quente'
else:
    temperatura = 'fria'

tags = [tag.name for tag in cs_imagem.tags]
descricao = cs_imagem.description.captions[0].text


print('Tags: ', tags)
print('Descrição: ', descricao)
print('Temperatura: ', temperatura)
