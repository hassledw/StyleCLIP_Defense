from attacks import FGSM
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import pandas as pd
import os

device = 'cpu'
processor = AutoImageProcessor.from_pretrained("RickyIG/emotion_face_image_classification_v3")
model = ViTForImageClassification.from_pretrained("RickyIG/emotion_face_image_classification_v3")
model.to(device)

# creating a image object (main image)

def get_image_label(filename, show_image=False):
    image = Image.open(rf"{filename}") 
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    if show_image:
        img = mpimg.imread(f'{filename}')
        imgplot = plt.imshow(img)
        plt.show()

    predicted_label = logits.argmax(-1).item()

    return model.config.id2label[predicted_label], logits

def get_confidence(logits):
    min_value = torch.min(logits)
    max_value = torch.max(logits)
    normalized_tensor = (logits - min_value) / (max_value - min_value)
    sum_values = torch.sum(normalized_tensor)

    return torch.max(normalized_tensor / sum_values).item()


df = pd.DataFrame(columns=['image', 'expression', 'confidence'])
n_images = 40000

attack = Attack()

for num in range(n_images):
    image_name = f'{num:05d}.png'
    image_path = f'./FFHQ512/{image_name}'
    print(image_name)
    if not os.path.isfile(image_path):
        continue
    attack.FGSM(image_path)
    expression, logits = get_image_label("./FGSM.jpg")
    confidence = get_confidence(logits)

    entry = [image_name, expression, confidence]
    df_entry = pd.DataFrame(entry, index=["image", "expression", "confidence"]).T
    df = pd.concat((df, df_entry))

# print(df)
df.to_csv('./FFHQ-512-FGSM.csv')  