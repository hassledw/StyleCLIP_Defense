'''
FGSM Attacker with PyTorch

Author: Daniel Hassler

'''
import torch
import urllib
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":
    model = load_mobilenet_model()
    filename = download_sample_image()
    sample_execution(filename)