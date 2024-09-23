import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from PIL import Image
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models
from torch.optim import lr_scheduler
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SaveBestModel:
   
    def __init__(
        self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'best_model.pth')
            
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim, dtype=torch.float32))

    def forward(self, query, key, value, mask=None):
        batch_size, channels, height, width = query.size()
        query = query.view(batch_size, channels, height * width)
        key = key.view(batch_size, channels, height * width)
        value = value.view(batch_size, channels, height * width)

        score = torch.bmm(query.transpose(1, 2), key) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value.transpose(1, 2))
        context = context.view(batch_size, channels, height, width)
        return context, attn

class MultiplicativeAttention(nn.Module):
    def __init__(self, dim):
        super(MultiplicativeAttention, self).__init__()
        self.dim = dim

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()
        query = query.view(batch_size, channels, height * width)
        key = key.view(batch_size, channels, height * width)
        value = value.view(batch_size, channels, height * width)

        score = torch.bmm(query.transpose(1, 2), key)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn, value.transpose(1, 2))
        context = context.view(batch_size, channels, height, width)
        return context, attn

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, 1)

    def forward(self, query, key, value):
        score = self.V(torch.tanh(self.W2(key) + self.W1(query).unsqueeze(1))).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value).squeeze(1)
        return context, attn

class RichardsSigmoid(nn.Module):
    def __init__(self, units=1):
        super(RichardsSigmoid, self).__init__()
        self.A = nn.Parameter(torch.rand(units))
        self.Q = nn.Parameter(torch.rand(units))
        self.mu = nn.Parameter(torch.rand(units))
        self.A.data.abs_()
        self.Q.data.abs_()
        self.mu.data.abs_()

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        A = self.A.view(1, -1, 1, 1)  
        Q = self.Q.view(1, -1, 1, 1)  
        mu = self.mu.view(1, -1, 1, 1)  
        return 1 / (1 + torch.exp(-A * torch.exp(-Q * (x -mu))))

   
class AttentionModel(nn.Module):
    def __init__(self, num_classes, k=0.1):
        super(AttentionModel, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.scaled_dot_product = ScaledDotProductAttention(dim=2048)
        self.multiplicative_attention = MultiplicativeAttention(dim=2048)
        self.additive_attention = AdditiveAttention(dim=2048)
        self.richards_sigmoid = RichardsSigmoid(units=2048)
        # self.global_attention = AdditiveAttention(dim=2048)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            # nn.Linear(2048 * 3, 512),  # concatenated channels * 3
            # nn.ReLU(),
            nn.Linear(612, 256),
            nn.BatchNorm1d(256),  
            nn.ReLU(),

            nn.Linear(256,64),
            nn.BatchNorm1d(64),  
            nn.ReLU(),

            # nn.Linear(64,32),
            # nn.BatchNorm1d(32),  
            # nn.ReLU(),
            # nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

        self.k = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

    def top_k_channel_selection(self, x):
        batch_size, channels, height, width = x.size()
        # print(f"Input Tensor Shape: {x.shape}")
        k = max(1, int(self.k * channels)) 
        x = self.richards_sigmoid(x)
        x_flat = x.view(batch_size, channels, -1)
        channel_means = x_flat.mean(dim=2)
        _, top_k_indices = channel_means.topk(k, dim=1)
        top_k_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        selected_channels = x.gather(1, top_k_indices)
        
        return selected_channels


    def forward(self, x):
        x = self.backbone(x)
        B, C, H, W = x.size()
        x_for_additive = x.view(B, C, H * W).permute(0, 2, 1)
        global_context = self.avg_pool(x).view(B, C)

        context1, _ = self.scaled_dot_product(x, x, x)
        # print(f"Scaled Dot-Product Attention output shape: {context1.shape}")
        context2, _ = self.multiplicative_attention(x, x, x)
        # print(f"multi Attention output shape: {context2.shape}")
        context3, _ = self.additive_attention(global_context, x_for_additive, x_for_additive)
        # print(f"additive  Attention output shape: {context3.shape}")
        context3 = context3.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
        # print(context1.shape, context2.shape, context3.shape)
        # context3 = context3.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions: [B, C, 1, 1]
        # context3 = F.adaptive_avg_pool2d(context3, (H, W))
        # self.dropout = nn.Dropout(0.4)
        
        context1 = self.top_k_channel_selection(context1)
        # context1 = self.dropout(context1)
        context2 = self.top_k_channel_selection(context2)
        # context2 = self.dropout(context2)
        context3 = self.top_k_channel_selection(context3)
        # context3 = self.dropout(context3)

        concatenated = torch.cat([context1, context2, context3], dim=1)
        b1,c1,h1,w1 = concatenated.size()
        # concatenated, _ = self.global_attention(concatenated, concatenated, concatenated)
        concatenated = self.avg_pool(concatenated).view(b1, c1)
        x = self.flatten(concatenated)
        # print(x.shape)
        # quit()
        x = self.fc(x)
        return x
 
def train_model(model, train_loader, criterion, optimizer, device,):
    train_loss, train_acc = 0.0, 0.0
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get the index of the max log-probability
        total += labels.size(0)
        train_running_correct += predicted.eq(labels).sum().item()
    # scheduler.step()
    train_loss = train_running_loss / len(train_loader)
    train_acc = 100 * train_running_correct / total
    return train_loss, train_acc

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            val_running_correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_running_correct / total
    return val_loss, val_acc

def create_image_label_dataframe(data_dir):
    """
    Create a DataFrame of image file paths and their corresponding labels from a given directory.
    
    Args:
        data_dir (str): Path to the root directory containing subdirectories for each label.
        
    Returns:
        pd.DataFrame: DataFrame with columns 'filepath' and 'labels'.
    """
    filepath = []
    labels = []

    # List of subdirectories (folds) in the data directory
    folds = os.listdir(data_dir)

    # Iterate through each subdirectory
    for fold in folds:
        file_path = os.path.join(data_dir, fold)
        fpath = os.listdir(file_path)
        
        # Iterate through each file in the subdirectory
        for f in fpath:
            fil_path = os.path.join(file_path, f)
            filepath.append(fil_path) 
            labels.append(fold)
    
    # Create pandas Series for file paths and labels
    F_series = pd.Series(filepath, name="filepath")
    L_series = pd.Series(labels, name="labels")

    # Concatenate Series into a DataFrame
    df = pd.concat([F_series, L_series], axis=1)
    return df
    
class CustomDataset(Dataset): 
    def __init__(self, img_dir, dataframe, transform):
        self.img_dir = img_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        # image = cv2.bilateralFilter(image, 15, 75, 75)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx, 1]
        # print(f"Index: {idx}")
        # print(f"Image Name: {img_name}")
        # print(f"Image Path: {img_path}")
        # print(f"Label: {label}")
        label = torch.tensor(label, dtype=torch.long)  
        return image, label
    
train_dir = '/workspace/dataset/OCT-C8/train'
val_dir = '/workspace/dataset/OCT-C8/val'
test_dir = '/workspace/dataset/OCT-C8/test'

df_train = create_image_label_dataframe(train_dir)
df_val = create_image_label_dataframe(val_dir)
df_test = create_image_label_dataframe(test_dir)

unique_labels = df_train['labels'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
# print("label mapping: ", label_mapping)
# label mapping:  {'NORMAL': 0, 'CSR': 1, 'AMD': 2, 'DME': 3, 'CNV': 4, 'DRUSEN': 5, 'DR': 6, 'MH': 7}
df_train['labels'] = df_train['labels'].map(label_mapping)
df_val['labels'] = df_val['labels'].map(label_mapping)
df_test['labels'] = df_test['labels'].map(label_mapping)

# print("train",df_train)
# print("val",df_val)
# print("test",df_test)
# print(df_train.value_counts())

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),   
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),            
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_dataset = CustomDataset(train_dir, df_train, transform['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_dir,df_val, transform['val'])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomDataset(test_dir,df_test, transform['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = 8

model = AttentionModel(num_classes, k=0.1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
save_best_model = SaveBestModel()

train_loss_list, valid_loss_list = [], []
train_acc_list, valid_acc_list = [],[]
num_epochs=70
# Training and Validation
for epoch in range(num_epochs):
    print(f"[INFO]: Epoch {epoch+1} of {num_epochs}")
    
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = validate_model(model, val_loader, criterion, device)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

    print(f"Training loss: {train_loss:.3f}, Training acc: {train_acc:.3f}")
    print(f"Validation loss: {valid_loss:.3f}, Validation acc: {valid_acc:.3f}")
    
    save_best_model(valid_loss, epoch, model, optimizer, criterion) #save best weights
    
    print('-'*50)

save_plots(train_acc_list, valid_acc_list, train_loss_list, valid_loss_list)
print('TRAINING COMPLETE')

#eval
# model.load_state_dict(torch.load("best_model.pth")['model_state_dict'])
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
criterion = checkpoint['loss']

model.eval()
test_loss = 0.0
correct = 0
total = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


def plot_confusion_matrix(labels, pred_labels, save_path):
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7']  
    print(classification_report(labels, pred_labels, target_names=target_names))
    ConfusionMatrix = confusion_matrix(labels, pred_labels)
    print("Confusion matrix")
    print(ConfusionMatrix)
    CM = ConfusionMatrix.astype('float') / ConfusionMatrix.sum(axis=1)[:, np.newaxis]
    print("Classwise accuracy on test set of model", CM.diagonal()*100)
    sensitivity = np.diag(ConfusionMatrix) / np.sum(ConfusionMatrix, axis=1)
    print("Sensitivity (Recall) for each class:", sensitivity)
    specificity = []
    for i in range(len(target_names)):
        true_negatives = np.sum(np.delete(np.delete(ConfusionMatrix, i, axis=0), i, axis=1))
        false_positives = np.sum(np.delete(ConfusionMatrix[i, :], i))
        specificity.append(true_negatives / (true_negatives + false_positives))
    specificity = np.array(specificity)
    print("Specificity for each class:", specificity)
    plt.figure(figsize=(10, 8))
    sns.heatmap(ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

confusion_matrix_save_path = 'confusion_matrix.png'
plot_confusion_matrix(all_labels, all_predictions, confusion_matrix_save_path)
print(f"Confusion matrix saved to {confusion_matrix_save_path}")
