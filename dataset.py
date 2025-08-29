import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ReflectionDataset(Dataset):
    """Dataset for loading reflection images and masks"""

    def __init__(self, img_dir, transform=None, is_train=True, mask_dir=None):
        self.img_dir = img_dir.replace("\\", "/")
        self.transform = transform
        self.is_train = is_train

        # Get images
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(os.path.join(img_dir, f))
        ]

        # Check for masks directory
        if mask_dir is None:
            self.mask_dir = os.path.join(img_dir, "masks").replace("\\", "/")
        else:
            self.mask_dir = mask_dir.replace("\\", "/")

        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Masks directory {self.mask_dir} does not exist")

        # Match images with masks
        self.valid_images = []
        self.mask_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_file).replace("\\", "/")
            if os.path.exists(mask_path):
                self.valid_images.append(img_file)
                self.mask_files.append(mask_file)

        if not self.valid_images:
            raise ValueError(f"No valid image-mask pairs found in {img_dir}")
        print(f"Found {len(self.valid_images)} valid image-mask pairs")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.valid_images[idx]).replace("\\", "/")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, self.mask_files[idx]).replace("\\", "/")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
        mask = mask.astype(np.float32) / 255.0

        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return {
            "image": image,
            "mask": mask.unsqueeze(0) if len(mask.shape) == 2 else mask,
            "image_id": self.valid_images[idx],
        }


class PseudoLabelDataset(Dataset):
    """Dataset for pseudo-labeled images"""

    def __init__(self, img_dir, pseudo_labels_dir, transform=None):
        self.img_dir = img_dir.replace("\\", "/")
        self.pseudo_labels_dir = pseudo_labels_dir.replace("\\", "/")
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(os.path.join(img_dir, f))
        ]

        self.valid_images = []
        self.mask_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}_pseudo.png"
            mask_path = os.path.join(self.pseudo_labels_dir, mask_file).replace("\\", "/")
            if os.path.exists(mask_path):
                self.valid_images.append(img_file)
                self.mask_files.append(mask_file)

        print(f"Found {len(self.valid_images)} images with pseudo-labels")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.valid_images[idx]).replace("\\", "/")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.pseudo_labels_dir, self.mask_files[idx]).replace("\\", "/")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
        mask = mask.astype(np.float32) / 255.0

        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return {
            "image": image,
            "mask": mask.unsqueeze(0) if len(mask.shape) == 2 else mask,
            "image_id": self.valid_images[idx],
        }


class UnlabeledDataset(Dataset):
    """Dataset for unlabeled images"""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir.replace("\\", "/")
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(os.path.join(img_dir, f).replace("\\", "/"))
        ]
        print(f"Found {len(self.image_files)} unlabeled images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx]).replace("\\", "/")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            except Exception as e:
                print(f"Error applying transform to image {self.image_files[idx]}: {e}")
                image = cv2.resize(image, (320, 320))
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.zeros((1, 320, 320)).float()

        return {
            "image": image,
            "mask": mask.unsqueeze(0) if len(mask.shape) == 2 else mask,
            "image_id": self.image_files[idx],
        }


def get_transforms(is_train, img_size=(320, 320)):
    """Get transforms for training and validation"""
    if is_train:
        return A.Compose(
            [
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=3, p=0.5),
                        A.MedianBlur(blur_limit=3, p=0.5),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )
