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

# from StyleCLIP_Defense.resources.attacks import Attack
import attackstorch
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import pandas as pd
import numpy as np
import os
import time

path = "/home/grads/hassledw"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoImageProcessor.from_pretrained("RickyIG/emotion_face_image_classification_v3")
model = ViTForImageClassification.from_pretrained("RickyIG/emotion_face_image_classification_v3")
model.to(device)

# creating a image object (main image)

def get_image_label(filename, show_image=False):
    '''
    Given the filename of the image, return the classified label
    and its logits (logits are used to determine confidence).
    '''
    image = Image.open(rf"{filename}") 
    inputs = processor(image, return_tensors="pt")
    inputs.to(device)

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
    '''
    Return the confidence value for the associated classification. 
    '''
    min_value = torch.min(logits)
    max_value = torch.max(logits)
    normalized_tensor = (logits - min_value) / (max_value - min_value)
    sum_values = torch.sum(normalized_tensor)

    return torch.max(normalized_tensor / sum_values).item()


def run_attack(attack, filename):
    '''
    Given an attack, run the attack on all images in the FFHQ-512-labeled.csv file. This
    will save the attack results into the filename (csv).
    '''
    print("Running Attack...")
    print(f"{filename}")

    orig_df = pd.read_csv(f"{path}/StyleCLIP_Defense/FFHQ512-Labeled/FFHQ-512-labeled.csv")
    label_encoder = LabelEncoder()
    
    attack_df = pd.DataFrame(columns=['image', 'expression', 'confidence'])
    # step = 50
    # n_images = 38000
    
    step = 5
    n_images = 100

    total_time = time.time()
    end_time = 0
    start_time = 0
    times = []
    # runs the attacks in batches for memory purposes.
    for x in range(0, n_images, step):
        if x != 0:
            times.append(end_time - start_time)
        print(f"Progress: {(x / n_images) * 100:.2f}% Done, Time Elapsed: {(end_time - total_time) / 60:.3f}m, Estimated Total: {(np.mean(np.array(times)) * (n_images / step)) / 60:.3f}m")
        start_time = time.time()
        batch_df = orig_df[x:x+step]
        encoded_labels = label_encoder.fit_transform(batch_df["expression"])
        batch_df["expression"] = encoded_labels

        labels = torch.tensor(batch_df["expression"].values)
        adv_images, file_names = attackstorch.generate_attack(attack, batch_df, labels)

        for i, image in enumerate(adv_images):
            image.save(f"{path}/StyleCLIP_Defense/images/FGSM.png")
            
            os.system("python3 /home/grads/hassledw/StyleCLIP/defense.py")
            
            expression, logits = get_image_label(f"{path}/StyleCLIP_Defense/images/generated.png")
            confidence = get_confidence(logits)
            entry = [file_names[i], expression, confidence]
            attack_df_entry = pd.DataFrame(entry, index=["image", "expression", "confidence"]).T
            attack_df = pd.concat((attack_df, attack_df_entry))
            
        end_time = time.time()
    
    attack_df.to_csv(f'{path}/StyleCLIP_Defense/FFHQ512-Labeled/{filename}')

def main():
    model = models.resnet18(pretrained=True)
    model.to(device)
    # PGD_attack1 = PGD(model, eps=0.1, alpha=0.1)
    # PGD_attack2 = PGD(model, eps=0.1, alpha=0.2)
    # PGD_attack3 = PGD(model, eps=0.2, alpha=0.1)
    # PGD_attack4 = PGD(model, eps=0.5, alpha=0.5)
    Jitter_attack = Jitter(model, eps=0.1, alpha=0.1)
    # run_attack(PGD_attack1, 'FFHQ-512-PGD-10-10-verify.csv')
    # run_attack(PGD_attack2, 'FFHQ-512-PGD-10-20.csv')
    # run_attack(PGD_attack3, 'FFHQ-512-PGD-20-10.csv')
    # run_attack(PGD_attack4, 'FFHQ-512-PGD-50-50.csv')
    # run_attack(FGSM_attack, 'FFHQ-512-FGSM-50.csv')
    # run_attack(Jitter_attack, 'FFHQ-512-Jitter-10-10.csv')
    run_attack(Jitter_attack, 'FFHQ-512-Jitter-10-10-def.csv')

if __name__ == "__main__":
    main()