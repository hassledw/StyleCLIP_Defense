# pip install torchattacks==2.14.5
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

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

class PILToFloatTensor(object):
    def __call__(self, img):
        # Convert PIL image to a float tensor
        return transforms.functional.to_tensor(img).float()

def normalize_between_range(img_tensor):
    input_tensor = img_tensor
    min_value = 0.0
    max_value = 1.0

    # Normalize the tensor to the specified range
    normalized_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value
    return normalized_tensor

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    model.to("cpu")
    df = pd.read_csv("/home/grads/hassledw/StyleCLIP_Defense/FFHQ-512-labeled.csv")
    df = df[:10] # first 10
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df["expression"])
    df["expression"] = encoded_labels

    labels = torch.tensor(df["expression"].values)
    images = []

    for file_name in df["image"]: 
        img = Image.open(f"/home/grads/hassledw/StyleCLIP_Defense/FFHQ512/{file_name}")
        transform = transforms.Compose([
            PILToFloatTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        input_tensor = transform(img)
        images.append(normalize_between_range(input_tensor))

    # everything but JSMA and SPSA works so far.
    atk = AutoAttack(model)
    images = torch.stack(images)
    adv_images = atk.forward(images, labels)
    print(adv_images)
    
    