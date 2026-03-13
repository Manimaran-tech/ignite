import torch
from typing import Callable, Iterable, Sequence


def get_sequence_transform(
    ignore_index: int | Iterable[int] | None = None,
    output_transform: Callable = lambda x: x,
) -> Callable:
    """
    Returns a callable to transform sequence model outputs for metric evaluation.
    It flattens the sequences and filters out the padding (`ignore_index`).
    
    Args:
        ignore_index: An integer or an iterable of integers representing padding or 
            special tokens to be masked out from the sequence evaluation.
        output_transform: A callable to transform the output into `(y_pred, y)`.
    
    Returns:
        Callable that flattens `y_pred` and `y` and removes `ignore_index` elements.
    """
    def wrapper(output: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        y_pred, y = output_transform(output)

        if y_pred.ndimension() == 3 and y.ndimension() == 2:
            if y_pred.shape[:2] == y.shape:
                # y_pred is (N, L, C), y is (N, L)
                y_pred = y_pred.reshape(-1, y_pred.size(-1))
                y = y.reshape(-1)
            elif y_pred.shape[0] == y.shape[0] and y_pred.shape[2] == y.shape[1]:
                # y_pred is (N, C, L), y is (N, L)
                y_pred = y_pred.transpose(1, 2).reshape(-1, y_pred.size(1))
                y = y.reshape(-1)
            else:
                raise ValueError(
                    f"y_pred and y have incompatible sequence shapes: "
                    f"y_pred={y_pred.shape} vs y={y.shape}"
                )
        elif y_pred.ndimension() in (2, 3) and y.ndimension() == y_pred.ndimension():
            # y_pred is (N, L) or (N, L, C)/(N, C, L) and y has identical shape
            if y_pred.shape == y.shape:
                y_pred = y_pred.reshape(-1)
                y = y.reshape(-1)
            else:
                raise ValueError(
                    f"y_pred and y have incompatible sequence shapes: "
                    f"y_pred={y_pred.shape} vs y={y.shape}"
                )
        else:
            raise ValueError(
                f"y_pred and y must be 3D/2D, 3D/3D, or 2D/2D arrays "
                f"for sequence transformation. Got {y_pred.ndimension()}D and {y.ndimension()}D."
            )

        if ignore_index is not None:
            if isinstance(ignore_index, Iterable):
                mask = torch.ones_like(y, dtype=torch.bool)
                for idx in ignore_index:
                    mask &= (y != idx)
            else:
                mask = y != ignore_index
            
            y_pred = y_pred[mask]
            y = y[mask]

        return y_pred, y

    return wrapper
