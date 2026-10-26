"""Production web API for DeepOCR."""
import sys
from pathlib import Path

# The API reuses the research packages (models/, preprocessing/, detection/,
# postprocessing/); make the project root importable however uvicorn is launched.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# On Windows, PaddlePaddle and PyTorch ship conflicting native DLLs: if Paddle is
# loaded first, `import torch` later fails (WinError 127 loading shm.dll). Loading
# torch first makes both work, so import it before any engine touches Paddle.
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch is optional for the PaddleOCR-only setup
    pass
