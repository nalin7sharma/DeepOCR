"""Unit tests for model forward passes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from models.cnn_bilstm import CNNBiLSTM, build_cnn_bilstm
from models.crnn import CRNN, build_crnn
from models.transformer_ocr import CustomViTOCR


NUM_CLASSES = 80
BATCH_SIZE = 2
HEIGHT = 64
WIDTH = 128


@pytest.fixture
def dummy_batch():
    return torch.zeros(BATCH_SIZE, 1, HEIGHT, WIDTH)


class TestCNNBiLSTM:
    def test_forward_shape(self, dummy_batch):
        model = CNNBiLSTM(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.ndim == 3
        assert out.shape[1] == BATCH_SIZE
        assert out.shape[2] == NUM_CLASSES

    def test_log_probs_sum_to_one(self, dummy_batch):
        model = CNNBiLSTM(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        # Each timestep should be log-softmax (exp sums to ~1)
        probs = out[0].exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-4)

    def test_build_factory(self):
        model = build_cnn_bilstm(NUM_CLASSES)
        assert isinstance(model, CNNBiLSTM)


class TestCRNN:
    def test_forward_shape(self, dummy_batch):
        model = CRNN(num_classes=NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.ndim == 3
        assert out.shape[1] == BATCH_SIZE
        assert out.shape[2] == NUM_CLASSES

    def test_build_factory(self):
        model = build_crnn(NUM_CLASSES)
        assert isinstance(model, CRNN)


class TestCustomViTOCR:
    def test_encode_shape(self):
        model = CustomViTOCR(
            num_classes=NUM_CLASSES,
            image_height=HEIGHT, image_width=WIDTH,
            patch_size=16, embed_dim=128,
            num_heads=4, encoder_layers=2, decoder_layers=2,
        )
        model.eval()
        batch = torch.zeros(BATCH_SIZE, 1, HEIGHT, WIDTH)
        with torch.no_grad():
            memory = model.encode(batch)
        assert memory.ndim == 3
        assert memory.shape[0] == BATCH_SIZE

    def test_forward_shape(self):
        model = CustomViTOCR(
            num_classes=NUM_CLASSES,
            image_height=HEIGHT, image_width=WIDTH,
            patch_size=16, embed_dim=128,
            num_heads=4, encoder_layers=2, decoder_layers=2,
            max_target_length=20,
        )
        model.eval()
        batch = torch.zeros(BATCH_SIZE, 1, HEIGHT, WIDTH)
        tgt = torch.zeros(BATCH_SIZE, 10, dtype=torch.long)
        with torch.no_grad():
            out = model(batch, tgt)
        assert out.shape == (BATCH_SIZE, 10, NUM_CLASSES)
