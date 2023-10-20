import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
from torchattacks.attack import Attack
import torchvision.transforms as transforms
from torchattacks import JSMA, PGD, FGSM, SPSA, RFGSM, Jitter, OnePixel, FAB, AutoAttack
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import art.attacks.evasion as evasion
from art.estimators.classification import PyTorchClassifier

from attacks import Attack
import attackstorch
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


def run_attack():
    print("Running Attack...")
    model = models.resnet18(pretrained=True)
    model.to("cpu")

    orig_df = pd.read_csv("/home/grads/hassledw/StyleCLIP_Defense/FFHQ512-Labeled/FFHQ-512-labeled.csv")
    # orig_df = orig_df[:10] # first 10
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(orig_df["expression"])
    orig_df["expression"] = encoded_labels

    labels = torch.tensor(orig_df["expression"].values)
    adv_images, file_names = attackstorch.generate_attack(FGSM(model, eps=0.05), orig_df, labels)
    # print(adv_images)
    
    
    attack_df = pd.DataFrame(columns=['image', 'expression', 'confidence'])
    for i, image in enumerate(adv_images):
        print(i)
        image.save("./FGSM.png")
        expression, logits = get_image_label("./FGSM.png")
        confidence = get_confidence(logits)
        entry = [file_names[i], expression, confidence]
        attack_df_entry = pd.DataFrame(entry, index=["image", "expression", "confidence"]).T
        attack_df = pd.concat((attack_df, attack_df_entry))
        
    attack_df.to_csv('./FFHQ512-Labeled/FFHQ-512-FGSM-05.csv') 
    
run_attack()