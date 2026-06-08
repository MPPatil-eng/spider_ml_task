import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 64

CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ── Load Data ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = torchvision.datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Split 60000 training samples into 54000 train / 6000 val
train_data, val_data = random_split(full_train, [54000, 6000])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)


# ── Model ─────────────────────────────────────────────────────────────────────
class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.shared  = nn.Sequential(nn.Linear(784, 16), nn.ReLU())

        # Branch A with skip connection
        self.a1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.a2 = nn.Sequential(nn.Linear(8,  8), nn.ReLU())

        # Branch B
        self.b  = nn.Sequential(nn.Linear(16, 12), nn.ReLU(),
                                 nn.Linear(12,  8), nn.ReLU())

        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x  = self.shared(self.flatten(x))
        a1 = self.a1(x)
        a  = a1 + self.a2(a1)          # skip connection
        b  = self.b(x)
        return self.out(torch.cat([a, b], dim=1))


model     = FashionNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ── Train ─────────────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(1, EPOCHS + 1):
    # training
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
    train_losses.append(total_loss / len(train_data))
    train_accs.append(correct / len(train_data))

    # validation
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
    val_losses.append(total_loss / len(val_data))
    val_accs.append(correct / len(val_data))

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]*100:.1f}%  |  "
          f"Val Loss: {val_losses[-1]:.4f}  Acc: {val_accs[-1]*100:.1f}%")

print(f"\nFinal Validation Accuracy : {val_accs[-1]*100:.2f}%")
print(f"Final Validation Loss     : {val_losses[-1]:.4f}")


# ── Save weights ──────────────────────────────────────────────────────────────
with open("fashion_net_weights.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)
print("Saved fashion_net_weights.pkl")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label="Train"); axes[0].plot(val_losses, label="Val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot([a*100 for a in train_accs], label="Train"); axes[1].plot([a*100 for a in val_accs], label="Val")
axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()


# ── submission.csv ────────────────────────────────────────────────────────────
model.eval()
preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())

pd.DataFrame({"Id": range(len(preds)), "Label": preds}).to_csv("submission.csv", index=False)
print("Saved submission.csv")


# =============================================================================
# BONUS: Autoencoder
# =============================================================================
AE_EPOCHS  = 15
LATENT_DIM = 32

# No normalisation — Sigmoid output needs pixels in [0,1]
ae_transform = transforms.Compose([transforms.ToTensor()])
ae_train = torchvision.datasets.FashionMNIST(root="./data", train=True,  download=False, transform=ae_transform)
ae_test  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=ae_transform)
ae_train_loader = DataLoader(ae_train, batch_size=128, shuffle=True)
ae_test_loader  = DataLoader(ae_test,  batch_size=128, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, 784),        nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


ae        = Autoencoder().to(DEVICE)
ae_optim  = optim.Adam(ae.parameters(), lr=0.001)
ae_crit   = nn.BCELoss()

print("\n--- Autoencoder Training ---")
for epoch in range(1, AE_EPOCHS + 1):
    ae.train()
    total_loss = 0
    for imgs, _ in ae_train_loader:
        imgs = imgs.to(DEVICE)
        ae_optim.zero_grad()
        recon = ae(imgs)
        loss  = ae_crit(recon, imgs.view(imgs.size(0), -1))
        loss.backward()
        ae_optim.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"AE Epoch {epoch}/{AE_EPOCHS}  Loss: {total_loss/len(ae_train):.4f}")

with open("autoencoder_weights.pkl", "wb") as f:
    pickle.dump(ae.state_dict(), f)
print("Saved autoencoder_weights.pkl")

# Show original vs reconstructed
ae.eval()
sample_imgs, sample_labels = next(iter(ae_test_loader))
sample_imgs = sample_imgs[:10].to(DEVICE)
with torch.no_grad():
    recon_imgs = ae(sample_imgs).view(-1, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(16, 4))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].cpu().view(28, 28), cmap="gray"); axes[0, i].axis("off")
    axes[0, i].set_title(CLASS_NAMES[sample_labels[i]], fontsize=7)
    axes[1, i].imshow(recon_imgs[i], cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original");      axes[1, 0].set_ylabel("Reconstructed")
plt.suptitle("Autoencoder Reconstructions"); plt.tight_layout()
plt.savefig("ae_reconstructions.png", dpi=150); plt.show()
print("Saved ae_reconstructions.png")
