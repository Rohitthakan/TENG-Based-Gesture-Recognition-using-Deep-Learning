import os
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

MODEL_PATH = r"G:\Rohit\Training And Results\training_results/best_model.pth"
TEST_DIR   = r"G:\Rohit\Training And Results\Test"

OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# SETTINGS
# ===============================

# CLASSES = ["ASCEND","DESCEND","DIRECTION","HELP","LOW AIR","NO Word","OK","OUT OF AIR","PROBLEM","STOP"]
CLASSES = ["ASCEND","DIRECTION","HELP","LOW AIR","NO Word","OK","OUT OF AIR","PROBLEM","SAVE","STOP"]

VOLT_COLUMN = "Voltage (V)"

TARGET_LENGTH = 16000


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
# MODEL (same as training)
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
# LOAD MODEL
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1D().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

print("Model loaded successfully")


# ===============================
# LOAD TEST DATA
# ===============================

test_dataset = TENGDataset(TEST_DIR)

test_loader = DataLoader(test_dataset, batch_size=1)


# ===============================
# TESTING
# ===============================

y_true = []
y_pred = []

class_total = [0]*len(CLASSES)
class_correct = [0]*len(CLASSES)


for x,y in test_loader:

    pred = tta_predict(model, x[0], device)

    y_true.append(y.item())
    y_pred.append(pred)

    class_total[y.item()] += 1

    if pred == y.item():
        class_correct[y.item()] += 1


# ===============================
# PRINT RESULTS
# ===============================

print("\n===== TEST RESULTS =====\n")

total_correct = 0
total_samples = 0

for i, cls in enumerate(CLASSES):

    correct = class_correct[i]
    total = class_total[i]
    incorrect = total - correct

    total_correct += correct
    total_samples += total

    print(f"Class: {cls}")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {incorrect}")
    print("---------------------------")


accuracy = total_correct / total_samples

print("\nOverall Test Accuracy:", round(accuracy*100,2), "%")


# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true,y_pred)

disp = ConfusionMatrixDisplay(cm,display_labels=CLASSES)

disp.plot(xticks_rotation=45)

plt.savefig(os.path.join(OUTPUT_DIR,"test_confusion_matrix.png"))

print("\nConfusion matrix saved in test_results/")