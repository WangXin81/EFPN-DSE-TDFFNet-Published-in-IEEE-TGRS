from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.utils.data as Data
from dataset.mydataset import MyDataLoader
from EFPN import new_resnet_fpn

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
model_dict_pth = "/content/drive/My Drive/One-label/model/UC82.pth"

model_name = "resnet"
num_classes = 21
num_epochs = 1
feature_extract = False

def predict(model, dataloaders, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    predict_labels = np.array([])
    true_labels = np.array([])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['val']:
            model.eval()   # Set model to evaluate mode

            running_corrects = 0
            for inputs, labels  in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)
                p = preds.cpu()
                predict_label = p.numpy()
                predict_labels = np.append(predict_labels,predict_label)
                q = labels.data.cpu()
                true_label = q.numpy()
                true_labels = np.append(true_labels,true_label)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{}  Acc: {:.4f}'.format(phase, epoch_acc))

    np.save("/content/drive/My Drive/One-label/UCpredict_labels.npy", predict_labels)
    np.save("/content/drive/My Drive/One-label/UCtrue_labels.npy", true_labels)
    
    print("save .npy done")
    print("save model...")
    model_save_path = "/content/drive/My Drive/One-label/model/"+ "predicted" + "_" + str(epoch+1)+".pth"
    torch.save(model, model_save_path)
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        model_ft = new_resnet_fpn(num_classes=num_classes,pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size



# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = torch.load(model_dict_pth)

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")
img_root_dir = "/content/drive/My Drive/data/UCMerced_Landuse/"
val_txt_path = "/content/drive/My Drive/data/UCMerced_Landuse/val.txt"

test_batch_size = 10
test_dataset = MyDataLoader(img_root=img_root_dir, txt_file=val_txt_path, transforms=data_transforms["val"])
test_dataloader = Data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
data_loader = {"train": train_dataloader, "val": test_dataloader}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

model_ft = predict(model_ft,data_loader,num_epochs=num_epochs)