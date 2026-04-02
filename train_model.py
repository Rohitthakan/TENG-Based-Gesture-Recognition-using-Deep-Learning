import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ===============================
# PATHS (CHANGE THESE)
# ===============================
TRAIN_DIR = r"G:\Rohit\Training And Results\Train"
VAL_DIR   = r"G:\Rohit\Training And Results\Validate"

OUTPUT_DIR = "training_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# SETTINGS
# ===============================
# CLASSES = ["ASCEND","DESCEND","DIRECTION","HELP","LOW AIR","NO Word","OK","OUT OF AIR","PROBLEM","STOP"]
CLASSES = ["ASCEND","DIRECTION","HELP","LOW AIR","NO Word","OK","OUT OF AIR","PROBLEM","SAVE","STOP"]

TIME_COLUMN = "Time_since_log_start (s)"
VOLT_COLUMN = "Voltage (V)"

TARGET_LENGTH = 16000

BATCH_SIZE = 16
EPOCHS = 70
LR = 0.0003


# ===============================
# DATASET
# ===============================
class TENGDataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []

        for idx, cls in enumerate(CLASSES):
            folder = os.path.join(root_dir, cls)

            for f in os.listdir(folder):
                if f.endswith(".csv"):
                    self.samples.append(
                        (os.path.join(folder,f), idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        df = pd.read_csv(path)

        signal = df[VOLT_COLUMN].values.astype(np.float32)

        signal = (signal - signal.mean())/(signal.std()+1e-6)

        if len(signal) > TARGET_LENGTH:
            signal = signal[:TARGET_LENGTH]
        else:
            pad = TARGET_LENGTH - len(signal)
            signal = np.pad(signal,(0,pad))

        signal = torch.tensor(signal).unsqueeze(0)

        return signal, label


# ===============================
# MODEL
# ===============================
class CNN1D(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1,64,9,padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,128,7,padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128,256,5,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256,512,3,padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(512,512,3,padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(8)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(512*8,512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.ReLU(),

            nn.Linear(256,len(CLASSES))
        )

    def forward(self,x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ===============================
# TEST TIME AUGMENTATION
# ===============================
def tta_predict(model, signal, device):

    signal = signal.squeeze().cpu().numpy()

    preds = []

    shift_step = 500
    num_shifts = 5

    for i in range(num_shifts):

        start = i * shift_step

        crop = signal[start:start+TARGET_LENGTH]

        if len(crop) < TARGET_LENGTH:
            crop = np.pad(crop,(0,TARGET_LENGTH-len(crop)))

        crop = torch.tensor(crop).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():

            out = model(crop)

            pred = out.argmax(1).item()

            preds.append(pred)

    final_pred = max(set(preds), key=preds.count)

    return final_pred


# ===============================
# LOAD DATA
# ===============================
train_dataset = TENGDataset(TRAIN_DIR)
val_dataset   = TENGDataset(VAL_DIR)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=1)


# ===============================
# TRAIN SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1D().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LR,
                              weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_val_acc = 0

train_acc_history=[]
val_acc_history=[]


# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0

    for x,y in train_loader:

        x,y = x.to(device),y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        preds = out.argmax(1)

        correct += (preds==y).sum().item()

        total += y.size(0)

    train_acc = correct/total


    # ===============================
    # VALIDATION WITH TTA
    # ===============================
    model.eval()

    v_correct = 0
    v_total = 0

    with torch.no_grad():

        for x,y in val_loader:

            pred = tta_predict(model, x[0], device)

            if pred == y.item():
                v_correct += 1

            v_total += 1

    val_acc = v_correct/v_total

    scheduler.step()

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"Train Acc:{train_acc:.3f} "
          f"Val Acc:{val_acc:.3f}")


    # ===============================
    # SAVE BEST MODEL
    # ===============================
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR,"best_model.pth"))

        # also save best COMPLETE model
        torch.save(model,
                   os.path.join(OUTPUT_DIR,"best_complete_model.pth"))

        print(f"New best model saved with Val Acc: {val_acc:.4f}")


# ===============================
# SAVE FINAL MODEL WEIGHTS
# ===============================
torch.save(model.state_dict(),
           os.path.join(OUTPUT_DIR,"final_model.pth"))


# ===============================
# SAVE COMPLETE TRAINED MODEL
# ===============================
torch.save(model,
           os.path.join(OUTPUT_DIR,"complete_model.pth"))

print("Complete trained model saved as complete_model.pth")


# ===============================
# SAVE LABEL MAP
# ===============================
with open(os.path.join(OUTPUT_DIR,"labels.json"),"w") as f:
    json.dump(CLASSES,f)


# ===============================
# PLOT ACCURACY
# ===============================
plt.plot(train_acc_history,label="train")
plt.plot(val_acc_history,label="val")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR,"accuracy_curve.png"))
plt.clf()


# ===============================
# CONFUSION MATRIX
# ===============================
y_true=[]
y_pred=[]

model.eval()

with torch.no_grad():

    for x,y in val_loader:

        pred = tta_predict(model, x[0], device)

        y_pred.append(pred)
        y_true.append(y.item())

cm = confusion_matrix(y_true,y_pred)

disp = ConfusionMatrixDisplay(cm,display_labels=CLASSES)

disp.plot(xticks_rotation=45)

plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"))


print("Training complete")
print("Best validation accuracy:",best_val_acc)