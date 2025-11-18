from typing import Any
import numpy as np

def ensure_numpy(x: Any) -> np.ndarray:
    """
    Приводим выход модели к np.ndarray float32.
    Поддерживает np.ndarray, torch.Tensor и списки.
    """
    if isinstance(x, np.ndarray):
        arr = x
    else:
        # torch.Tensor обычно имеет .detach().cpu().numpy()
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)

    if arr.dtype != np.float32:
        arr = arr.astype("float32")
    return arr