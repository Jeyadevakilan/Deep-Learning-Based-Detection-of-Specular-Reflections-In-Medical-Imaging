import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
import random

from models.unet import UNet
from data.dataset import ReflectionDataset, PseudoLabelDataset, UnlabeledDataset, get_transforms
from utils.visualization import overlay_mask

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def calculate_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    intersection = (pred & target).float().sum((2, 3))
    union = (pred | target).float().sum((2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def test_time_augmentation(model, image, device):
    model.eval()
    aug_preds = []
    with torch.no_grad():
        aug_preds.append(model(image))
        aug_preds.append(torch.flip(model(torch.flip(image, [3])), [3]))
        aug_preds.append(torch.flip(model(torch.flip(image, [2])), [2]))
        aug_preds.append(torch.flip(model(torch.flip(image, [2, 3])), [2, 3]))
    return torch.mean(torch.stack(aug_preds), dim=0)

def generate_pseudo_labels(model, data_loader, output_dir, device, confidence_threshold=0.75):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    total_images = 0
    confident_images = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating pseudo-labels"):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            outputs = test_time_augmentation(model, images, device)
            probabilities = torch.sigmoid(outputs)
            binary_masks = (probabilities > 0.5).float()
            confidence_maps = torch.max(probabilities, 1 - probabilities)
            for i in range(len(images)):
                total_images += 1
                avg_conf = confidence_maps[i].mean().item()
                if avg_conf >= confidence_threshold:
                    confident_images += 1
                    img_name = image_ids[i]
                    base_name = os.path.splitext(img_name)[0]
                    mask_path = os.path.join(output_dir, f"{base_name}_pseudo.png").replace("\\", "/")
                    mask_np = binary_masks[i, 0].cpu().numpy() * 255
                    cv2.imwrite(mask_path, mask_np.astype(np.uint8))

                    conf_path = os.path.join(output_dir, f"{base_name}_confidence.png").replace("\\", "/")
                    conf_np = confidence_maps[i, 0].cpu().numpy() * 255
                    cv2.imwrite(conf_path, conf_np.astype(np.uint8))

                    image_np = images[i].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = np.clip(std * image_np + mean, 0, 1) * 255
                    image_np = image_np.astype(np.uint8)

                    if mask_np.shape[:2] != image_np.shape[:2]:
                        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    overlay = overlay_mask(image_np, mask_np)
                    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png").replace("\\", "/")
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Generated {confident_images} pseudo masks from {total_images} images")
    return confident_images

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer, save_path, patience=10):
    best_val_iou = 0
    early_stop_counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_iou = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            images = batch["image"].to(device)
            masks = batch["mask"].float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape != masks.shape:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_iou = calculate_iou(outputs, masks)
            train_loss += loss.item()
            train_iou += batch_iou.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}", "train_iou": f"{batch_iou.item():.4f}"})
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)

        model.eval()
        val_loss = val_iou = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].float().to(device)
                outputs = test_time_augmentation(model, images, device)
                if outputs.shape != masks.shape:
                    outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
                loss = criterion(outputs, masks)
                batch_iou = calculate_iou(outputs, masks)
                val_loss += loss.item()
                val_iou += batch_iou.item()
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, LR: {current_lr:.7f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            early_stop_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss, "val_iou": val_iou}, save_path)
            print(f"Saved new best model with IoU: {val_iou:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    return best_val_iou

def main():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    os.makedirs("pseudo_labels", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    train_dir = "D:/spec/Medical images.v1i.coco-segmentation/train"
    train_mask_dir = f"{train_dir}/masks"
    test_dir = "D:/spec/Medical images.v1i.coco-segmentation/test"
    test_mask_dir = f"{test_dir}/masks"
    unlabeled_dir = "D:/spec/Medical images.v1i.coco-segmentation/unannotated"
    initial_model_path = "models/initial_model.pth"

    if not os.path.exists(initial_model_path):
        raise FileNotFoundError(f"Initial model not found at {initial_model_path}. Run train_initial.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")

    batch_size = 4
    semi_supervised_epochs = 30
    final_epochs = 30
    learning_rate = 8e-5
    weight_decay = 5e-6
    img_size = (320, 320)

    train_transform = get_transforms(is_train=True, img_size=img_size)
    val_transform = get_transforms(is_train=False, img_size=img_size)

    train_dataset = ReflectionDataset(train_dir, transform=train_transform, is_train=True, mask_dir=train_mask_dir)
    val_dataset = ReflectionDataset(test_dir, transform=val_transform, is_train=False, mask_dir=test_mask_dir)
    unlabeled_dataset = UnlabeledDataset(unlabeled_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
    writer = SummaryWriter("runs/semi_supervised")

    model = UNet(n_classes=1).to(device)
    checkpoint = torch.load(initial_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_val_iou = checkpoint["val_iou"]
    print(f"Loaded initial model with IoU: {initial_val_iou:.4f}")

    print("\n=== Phase 1: Pseudo Label Generation ===")
    pseudo_labels_dir = "pseudo_labels"
    num_pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, pseudo_labels_dir, device)

    if num_pseudo_labels > 0:
        print("\n=== Phase 2: Semi-Supervised Training ===")
        pseudo_dataset = PseudoLabelDataset(unlabeled_dir, pseudo_labels_dir, transform=train_transform)
        weighted_train_dataset = ConcatDataset([train_dataset] * 4)
        combined_dataset = ConcatDataset([weighted_train_dataset, pseudo_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-7)
        semi_supervised_model_path = "models/semi_supervised_model.pth"

        semi_supervised_val_iou = train_model(model, combined_loader, val_loader, CombinedLoss(0.4), optimizer, scheduler, device, semi_supervised_epochs, writer, semi_supervised_model_path)

        print("\n=== Phase 3: Fine-Tuning ===")
        checkpoint = torch.load(semi_supervised_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate / 2, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-7)
        final_model_path = "models/final_model.pth"

        final_val_iou = train_model(model, combined_loader, val_loader, CombinedLoss(0.4), optimizer, scheduler, device, final_epochs, writer, final_model_path, patience=15)
        print(f"\nModel Training Summary:\nInitial IoU: {initial_val_iou:.4f}\nSemi-Supervised IoU: {semi_supervised_val_iou:.4f}\nFinal IoU: {final_val_iou:.4f}\nImprovement: {final_val_iou - initial_val_iou:.4f}")
    else:
        print("No pseudo-labels generated. Skipping semi-supervised training.")
    writer.close()

if __name__ == "__main__":
    main()
