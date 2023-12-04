import torch
from torchvision import datasets, transforms
import numpy as np
import os
import pandas as pd
path = "/home/grads/hassledw"

def celebClassifier(data_dir, subdir, num_files, model, save=True):
    '''
    passes celebrity data into the classifier for classification.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, subdir), transforms_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_files, shuffle=False)
    
    # predictions = []
    # labels_arr = []
    confidences = []

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_dataloader):
            # print(f"{i}:\n INPUTS: {inputs}\n LABELS: {labels}\n")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # print("out:", out)
            # print("out: ", sum(outputs[0]))
            for x in range(len(outputs)):
                min_val = torch.min(outputs[x]).item()
                scale_list = outputs[x] + (-1 * min_val)
                max_val = torch.max(scale_list).item()
                confidence_val = max_val / sum(scale_list)
                confidences.append(confidence_val.item())

            labels_arr = np.array(labels.tolist())
            predictions = np.array(preds.tolist())

        # print("Labels: ", labels_arr)
        # print("Pred: ", predictions)
    if save:
        df = pd.DataFrame(columns=['image', 'image_path', 'actual_label', 'predicted_label', 'confidence'])
        for x, (imagepath, label) in enumerate(test_dataset.imgs):
            entry = [imagepath.split("/")[-1], imagepath, label, predictions[x], confidences[x]]
            df_entry = pd.DataFrame(entry, index=["image", "image_path", "actual_label", "predicted_label", "confidence"]).T
            df = pd.concat((df, df_entry))
        df.to_csv(f"/home/grads/hassledw/StyleCLIP_Defense/CelebA_HQ-Labeled/{subdir}.csv")
        print(f"Results saved to {subdir}.csv!")
        
    torch.cuda.empty_cache()
    
    return torch.tensor(labels_arr, dtype=torch.long)



    
