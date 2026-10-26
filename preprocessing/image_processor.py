"""
Image Preprocessing Module for DeepOCR.

Applies a configurable sequence of preprocessing operations to document images
before passing them to the detection and recognition pipeline.

Operations supported:
  - Grayscale conversion
  - Resize (with aspect-ratio preservation and padding)
  - Noise removal (Gaussian / non-local means)
  - Contrast enhancement (CLAHE)
  - Binarization (Otsu / adaptive threshold)
  - Deskewing (Hough-line-based skew correction)
  - Brightness normalisation
  - Blur correction (sharpening kernel)
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class ImageProcessor:
    """Apply a sequence of preprocessing steps to a document image.

    Args:
        config: Optional dict with boolean/numeric flags controlling which
                operations are applied.  Defaults make all steps active.

    Example:
        >>> proc = ImageProcessor()
        >>> result = proc.process(image_array)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.do_grayscale = cfg.get("grayscale", True)
        self.do_resize = cfg.get("resize", True)
        self.target_height = cfg.get("target_height", 64)
        self.target_width = cfg.get("target_width", 512)
        self.do_denoise = cfg.get("denoise", True)
        self.do_contrast = cfg.get("contrast_enhance", True)
        self.do_binarize = cfg.get("binarize", False)   # off by default; deep models prefer grey
        self.do_deskew = cfg.get("deskew", True)
        self.do_normalize = cfg.get("normalize", True)
        self.do_sharpen = cfg.get("sharpen", False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray | Image.Image, steps: list[str] | None = None) -> np.ndarray:
        """Apply the preprocessing pipeline to an image.

        Args:
            image: Input image as numpy array (BGR / RGB / grayscale) or PIL Image.
            steps: Optional list of step names to apply (overrides config flags).
                   Valid step names: grayscale, denoise, contrast, binarize,
                   deskew, normalize, sharpen.

        Returns:
            Preprocessed image as uint8 numpy array.
        """
        img = self._to_numpy(image)

        if steps is not None:
            # Explicit step list
            if "grayscale" in steps:
                img = self.to_grayscale(img)
            if "denoise" in steps:
                img = self.remove_noise(img)
            if "contrast" in steps:
                img = self.enhance_contrast(img)
            if "binarize" in steps:
                img = self.binarize(img)
            if "deskew" in steps:
                img = self.deskew(img)
            if "normalize" in steps:
                img = self.normalize_brightness(img)
            if "sharpen" in steps:
                img = self.sharpen(img)
        else:
            # Use config flags
            if self.do_grayscale:
                img = self.to_grayscale(img)
            if self.do_deskew:
                img = self.deskew(img)
            if self.do_denoise:
                img = self.remove_noise(img)
            if self.do_contrast:
                img = self.enhance_contrast(img)
            if self.do_binarize:
                img = self.binarize(img)
            if self.do_normalize:
                img = self.normalize_brightness(img)
            if self.do_sharpen:
                img = self.sharpen(img)

        if self.do_resize:
            img = self.resize_and_pad(img, self.target_height, self.target_width)

        return img

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert BGR / RGB image to grayscale."""
        if len(image.shape) == 2:
            return image  # already grayscale
        if image.shape[2] == 4:
            # RGBA: drop alpha
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Remove noise using Gaussian blur followed by median filter.

        Uses GaussianBlur + medianBlur as the portable alternative to
        fastNlMeansDenoising (which is not available in all OpenCV builds).
        """
        # Gentle Gaussian blur to reduce high-frequency noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Median blur to remove salt-and-pepper noise
        denoised = cv2.medianBlur(blurred, 3)
        return denoised

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(image)

    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        """Binarize a grayscale image using Otsu's thresholding."""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def deskew(image: np.ndarray, max_skew_deg: float = 15.0) -> np.ndarray:
        """Correct document skew by detecting dominant line orientation.

        Uses Hough line transform on an edge-detected version of the image.
        Angles beyond max_skew_deg are ignored (probably not skew).

        Args:
            image: Input image (grayscale or BGR).
            max_skew_deg: Maximum skew angle to correct (degrees).

        Returns:
            Deskewed image.
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return image

        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert theta to angle from horizontal
            angle = np.degrees(theta) - 90
            if abs(angle) < max_skew_deg:
                angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:  # negligible skew
            return image

        h, w = image.shape[:2]
        centre = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def normalize_brightness(image: np.ndarray) -> np.ndarray:
        """Normalise image brightness to [0, 255] using min-max stretching."""
        img_float = image.astype(np.float32)
        min_val, max_val = img_float.min(), img_float.max()
        if max_val - min_val < 1:
            return image
        normalised = (img_float - min_val) / (max_val - min_val) * 255
        return normalised.astype(np.uint8)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Apply an unsharp-masking sharpening kernel."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def resize_and_pad(
        image: np.ndarray,
        target_height: int = 64,
        target_width: int = 512,
    ) -> np.ndarray:
        """Resize image to target_height preserving aspect ratio, then pad width.

        Args:
            image: Input image.
            target_height: Fixed output height.
            target_width: Maximum output width; shorter images are zero-padded.

        Returns:
            Image of shape (target_height, target_width) or
            (target_height, target_width, C) for colour images.
        """
        h, w = image.shape[:2]
        scale = target_height / h
        new_w = min(int(w * scale), target_width)
        resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

        # Pad to target_width
        is_gray = len(resized.shape) == 2
        if is_gray:
            canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        else:
            canvas = np.zeros((target_height, target_width, resized.shape[2]), dtype=np.uint8)
        canvas[:, :new_w] = resized
        return canvas

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(image: np.ndarray | Image.Image) -> np.ndarray:
        """Convert PIL Image or numpy array to uint8 numpy array."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        return np.array(image, dtype=np.uint8)

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        return np.array(image)

    @staticmethod
    def numpy_to_pil(image: np.ndarray) -> Image.Image:
        if len(image.shape) == 2:
            return Image.fromarray(image, mode="L")
        # Assume BGR -> convert to RGB for PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
