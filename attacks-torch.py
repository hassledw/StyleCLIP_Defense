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
    '''
    Converts integer representation of PIL into float. Used in Transform.
    '''
    def __call__(self, img):
        # Convert PIL image to a float tensor
        return transforms.functional.to_tensor(img).float()

def normalize_between_range(img_tensor):
    '''
    Normalizes a tensor to the range [0,1], important
    for tensor input into the attack framework.
    '''
    input_tensor = img_tensor
    min_value = 0.0
    max_value = 1.0

    # Normalize the tensor to the specified range
    normalized_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value
    return normalized_tensor

def tensor_to_image(tensor_imgs):
    '''
    A helper function that converts a transformed adversarial tensor 
    into an image that visualizable.

    tensor_imgs: tensors of images
    '''
    images = []
    for tensor in tensor_imgs:
        # Step 1: Reverse normalization to [0, 1]
        min_value = 0.0
        max_value = 1.0
        tensor = (tensor - min_value) / (max_value - min_value)
        tensor *= 255.0

        tensor = tensor.to(torch.uint8)
        numpy_image = tensor.cpu().numpy()

        pil_image = Image.fromarray(numpy_image.transpose(1, 2, 0))  # Channels last

        images.append(pil_image)

    return images

def generate_attack(attack, df, labels):
    '''
    Uses torch.attacks attack method to create adversarial image given the
    specified attack, dataframe, and labels.

    attack: framework (FGSM, PGD, AutoAttack ...)
    df: image data in which to attack in the type of pd.dataframe with "image" column
    labels: an encoding of the labels (integer values)
    '''
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
    images = torch.stack(images)
    adv_tensors = attack.forward(images, labels)
    adv_images = tensor_to_image(adv_tensors)
    return adv_images

def main():
    model = models.resnet18(pretrained=True)
    model.to("cpu")

    df = pd.read_csv("/home/grads/hassledw/StyleCLIP_Defense/FFHQ-512-labeled.csv")
    df = df[:10] # first 10
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df["expression"])
    df["expression"] = encoded_labels

    labels = torch.tensor(df["expression"].values)
    adv_images = generate_attack(FGSM(model), df, labels)
    print(adv_images)

    adv_images[0].save("Test.jpg")

if __name__ == "__main__":
    main()