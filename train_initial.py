import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("utils", exist_ok=True)

# Import custom modules
from models.unet import UNet
from data.dataset import ReflectionDataset, get_transforms

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Combined BCE + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

# IoU Metric
def calculate_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    intersection = (pred & target).float().sum((2, 3))
    union = (pred | target).float().sum((2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# Test-Time Augmentation
def test_time_augmentation(model, image, device):
    model.eval()
    aug_preds = []

    with torch.no_grad():
        original_pred = model(image)
        aug_preds.append(original_pred)

        # Horizontal flip
        flipped = torch.flip(image, dims=[3])
        pred = model(flipped)
        aug_preds.append(torch.flip(pred, dims=[3]))

        # Vertical flip
        flipped = torch.flip(image, dims=[2])
        pred = model(flipped)
        aug_preds.append(torch.flip(pred, dims=[2]))

        # Both flips
        flipped = torch.flip(image, dims=[2, 3])
        pred = model(flipped)
        aug_preds.append(torch.flip(pred, dims=[2, 3]))

    avg_pred = torch.mean(torch.stack(aug_preds), dim=0)
    return avg_pred

# Training Loop
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    writer,
    save_path,
    patience=10,
):
    best_val_iou = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iou = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            images = batch["image"].to(device)
            masks = batch["mask"].float().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if outputs.shape != masks.shape:
                outputs = nn.functional.interpolate(
                    outputs, size=masks.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_iou = calculate_iou(outputs, masks)
            train_loss += loss.item()
            train_iou += batch_iou.item()

            progress_bar.set_postfix({
                "train_loss": f"{loss.item():.4f}",
                "train_iou": f"{batch_iou.item():.4f}",
            })

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)

        # Validation with TTA
        model.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].float().to(device)

                outputs = test_time_augmentation(model, images, device)

                if outputs.shape != masks.shape:
                    outputs = nn.functional.interpolate(
                        outputs, size=masks.shape[2:], mode="bilinear", align_corners=False
                    )

                loss = criterion(outputs, masks)
                batch_iou = calculate_iou(outputs, masks)

                val_loss += loss.item()
                val_iou += batch_iou.item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate", current_lr, epoch)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, "
            f"LR: {current_lr:.7f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            early_stop_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
            }, save_path.replace("\\", "/"))
            print(f"Saved new best model with IoU: {val_iou:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return best_val_iou

# Main Function
def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    train_dir = "D:/spec/Medical images.v1i.coco-segmentation/train"
    train_mask_dir = "D:/spec/Medical images.v1i.coco-segmentation/train/masks"
    test_dir = "D:/spec/Medical images.v1i.coco-segmentation/test"
    test_mask_dir = "D:/spec/Medical images.v1i.coco-segmentation/test/masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")

    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 1e-5
    img_size = (320, 320)

    train_transform = get_transforms(is_train=True, img_size=img_size)
    val_transform = get_transforms(is_train=False, img_size=img_size)

    print(f"Loading training dataset from {train_dir}")
    train_dataset = ReflectionDataset(train_dir, transform=train_transform, is_train=True, mask_dir=train_mask_dir)
    print(f"Loading validation dataset from {test_dir}")
    val_dataset = ReflectionDataset(test_dir, transform=val_transform, is_train=False, mask_dir=test_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    writer = SummaryWriter("runs/initial_training")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    print("\n=== Initial training with labeled data ===")
    model = UNet(n_classes=1).to(device)
    criterion = CombinedLoss(bce_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-7)

    initial_model_path = "models/initial_model.pth"
    initial_val_iou = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        writer,
        initial_model_path,
        patience=10,
    )

    print(f"Training completed with best validation IoU: {initial_val_iou:.4f}")
    print(f"Model saved to {initial_model_path}")
    writer.close()

if __name__ == "__main__":
    main()
