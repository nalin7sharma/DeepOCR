"""
Data Augmentation for OCR training.

Augmentations are applied only during training (not val/test).
Each augmentation is independently toggled and parameterised via config.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class OCRAugmentor:
    """Apply random augmentations to OCR line images.

    Args:
        config: Dict with augmentation parameters.  See keys below.

    Config keys (all optional, defaults shown):
        rotation_degrees  (5)      max absolute rotation angle
        brightness_range  ([0.7, 1.3])  PIL brightness factor range
        contrast_range    ([0.7, 1.3])  PIL contrast factor range
        noise_std         (0.02)   Gaussian noise std as fraction of 255
        blur_probability  (0.3)    probability of applying blur
        blur_kernel_max   (3)      max Gaussian blur kernel radius
        scale_range       ([0.8, 1.2])  random scale factor range
        perspective_distortion (0.2)   perspective warp strength
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.rotation_degrees = cfg.get("rotation_degrees", 5)
        self.brightness_range = tuple(cfg.get("brightness_range", [0.7, 1.3]))
        self.contrast_range = tuple(cfg.get("contrast_range", [0.7, 1.3]))
        self.noise_std = cfg.get("noise_std", 0.02)
        self.blur_probability = cfg.get("blur_probability", 0.3)
        self.blur_kernel_max = cfg.get("blur_kernel_max", 3)
        self.scale_range = tuple(cfg.get("scale_range", [0.8, 1.2]))
        self.perspective_distortion = cfg.get("perspective_distortion", 0.2)

    def __call__(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """Apply a random subset of augmentations.

        Args:
            image: Input image (numpy BGR/grey or PIL).

        Returns:
            Augmented image as uint8 numpy array.
        """
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        img = self._random_rotate(img)
        img = self._random_scale(img)
        img = self._add_gaussian_noise(img)
        img = self._random_blur(img)
        img = self._random_brightness_contrast(img)
        img = self._random_perspective(img)
        return img.astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_rotate(self, img: np.ndarray) -> np.ndarray:
        """Rotate inside an enlarged canvas so the ends of long lines are kept."""
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - w / 2
        M[1, 2] += new_h / 2 - h / 2
        background = int(np.median(img))
        border = (background,) * (img.shape[2] if img.ndim == 3 else 1)
        return cv2.warpAffine(
            img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=border
        )

    def _random_scale(self, img: np.ndarray) -> np.ndarray:
        """Rescale the text inside an unchanged canvas without ever cropping ink.

        Scale < 1 shrinks the text and pads with background; scale > 1 stretches
        it horizontally only (a wider-handwriting effect) so no characters are
        cut off — cropping would make the image disagree with its label.
        """
        scale = random.uniform(*self.scale_range)
        h, w = img.shape[:2]
        if scale >= 1.0:
            new_w = max(1, int(w * scale))
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        background = int(np.median(img))
        canvas = np.full(img.shape, background, dtype=img.dtype)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        std = self.noise_std * 255
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _random_blur(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.blur_probability:
            k = random.choice(range(1, self.blur_kernel_max + 1)) * 2 + 1  # odd kernel
            return cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _random_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img) if len(img.shape) == 2 else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Brightness
        factor = random.uniform(*self.brightness_range)
        pil = ImageEnhance.Brightness(pil).enhance(factor)
        # Contrast
        factor = random.uniform(*self.contrast_range)
        pil = ImageEnhance.Contrast(pil).enhance(factor)
        result = np.array(pil)
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result

    def _random_perspective(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        d = self.perspective_distortion
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = lambda: random.uniform(-d * min(w, h), d * min(w, h))
        pts2 = np.float32([
            [offset(), offset()], [w + offset(), offset()],
            [w + offset(), h + offset()], [offset(), h + offset()],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
