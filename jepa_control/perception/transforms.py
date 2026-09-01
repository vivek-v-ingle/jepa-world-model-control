import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ImagePreprocessor:
    """Preprocesses raw RGB images into V-JEPA compatible tensors."""

    def __init__(
        self,
        crop_size: int = 256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.crop_size = crop_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Input: np.ndarray [H, W, 3] uint8 or [T, H, W, 3]
        Output: torch.Tensor [1, 3, 2, crop_size, crop_size] (V-JEPA tubelet format)
        """
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                # Resize and center crop
                h, w, _ = img.shape
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                cropped = img[top : top + min_dim, left : left + min_dim]
                resized = cv2.resize(cropped, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                
                # Convert to [1, C, H, W] in [0, 1]
                tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            elif img.ndim == 4:
                # Sequence of frames [T, H, W, 3]
                frames = []
                for frame in img:
                    h, w, _ = frame.shape
                    min_dim = min(h, w)
                    top = (h - min_dim) // 2
                    left = (w - min_dim) // 2
                    cropped = frame[top : top + min_dim, left : left + min_dim]
                    resized = cv2.resize(cropped, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                    frames.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)
                tensor = torch.stack(frames, dim=1) # [C, T, H, W]
                tensor = tensor.unsqueeze(0)        # [1, C, T, H, W]
        elif torch.is_tensor(img):
            tensor = img.float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0

        # Normalize with ImageNet stats
        self.mean = self.mean.to(tensor.device)
        self.std = self.std.to(tensor.device)
        
        if tensor.ndim == 4:
            tensor = (tensor - self.mean) / self.std
            # Duplicate along temporal tubelet dimension to match V-JEPA expected format [B, C, 2, H, W]
            tensor = tensor.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        elif tensor.ndim == 5:
            tensor = (tensor - self.mean.unsqueeze(2)) / self.std.unsqueeze(2)

        return tensor
