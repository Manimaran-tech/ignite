import pytest
import torch

from ignite.metrics.utils import get_sequence_transform

def test_get_sequence_transform_shapes():
    # test (N, L, C)
    y_pred = torch.tensor(
        [
            [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.5, 0.5]],
            [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]],
        ]
    )  # shape: (2, 4, 2)
    y = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])  # shape: (2, 4)

    transform = get_sequence_transform()
    y_pred_t, y_t = transform((y_pred, y))

    assert y_pred_t.shape == (8, 2)
    assert y_t.shape == (8,)

    # test (N, C, L)
    y_pred_ncl = y_pred.transpose(1, 2).contiguous()  # (2, 2, 4)
    y_pred_t2, y_t2 = transform((y_pred_ncl, y))
    assert y_pred_t2.shape == (8, 2)
    assert torch.all(y_pred_t2 == y_pred_t)
    assert torch.all(y_t2 == y_t)

    # test binary (N, L)
    y_pred_bin = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]])
    y_bin = torch.tensor([[1, 0, 1, 2], [0, 1, 0, 2]])
    transform_bin = get_sequence_transform()
    y_pred_bin_t, y_bin_t = transform_bin((y_pred_bin, y_bin))
    
    assert y_pred_bin_t.shape == (8,)
    assert y_bin_t.shape == (8,)

    # test 3D tensors matched shaping (N, C, L) with (N, C, L)
    y_pred_3d = torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.5, 0.5]]])
    y_3d = torch.tensor([[[1, 0], [1, 1]], [[0, 1], [0, 0]]])
    transform_3d = get_sequence_transform()
    y_pred_3d_t, y_3d_t = transform_3d((y_pred_3d, y_3d))
    
    assert y_pred_3d_t.shape == (8,)
    assert y_3d_t.shape == (8,)
    assert y_pred_3d_t.tolist() == pytest.approx([0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.5, 0.5])
    assert y_3d_t.tolist() == [1, 0, 1, 1, 0, 1, 0, 0]

    # test bad shapes
    y_bad = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="must be 3D/2D, 3D/3D, or 2D/2D arrays"):
        transform((y_pred_bin, y_bad))

    y_pred_bad = torch.tensor([[[1], [2]], [[3], [4]]])
    y_bad = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="incompatible sequence shapes"):
        transform((y_pred_bad, y_bad))

def test_get_sequence_transform_ignore_index():
    # test padding with integer
    y_pred = torch.tensor(
        [
            [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.5, 0.5]],
            [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]],
        ]
    )
    y = torch.tensor([[1, 0, 1, -1], [0, 1, 0, -1]])

    transform = get_sequence_transform(ignore_index=-1)
    y_pred_t, y_t = transform((y_pred, y))

    assert y_pred_t.shape == (6, 2)
    assert y_t.shape == (6,)
    assert y_t.tolist() == [1, 0, 1, 0, 1, 0]
    assert y_pred_t[:, 1].tolist() == pytest.approx([0.9, 0.2, 0.7, 0.1, 0.8, 0.6])

    # test binary with integer ignore_index
    y_pred_bin = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]])
    y_bin = torch.tensor([[1, 0, 1, 2], [0, 1, 0, 2]])
    transform_bin = get_sequence_transform(ignore_index=2)
    y_pred_bin_t, y_bin_t = transform_bin((y_pred_bin, y_bin))
    
    assert y_pred_bin_t.shape == (6,)
    assert y_bin_t.shape == (6,)
    assert y_bin_t.tolist() == [1, 0, 1, 0, 1, 0]
    assert y_pred_bin_t.tolist() == [1, 0, 1, 0, 1, 0]

    # test multiple ignore_index values
    y_pred_bin = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]])
    y_bin = torch.tensor([[1, -1, 1, 2], [0, 1, -1, 2]])
    transform_multi = get_sequence_transform(ignore_index=[-1, 2])
    y_pred_multi_t, y_multi_t = transform_multi((y_pred_bin, y_bin))
    assert y_pred_multi_t.shape == (4,)
    assert y_multi_t.shape == (4,)
    assert y_multi_t.tolist() == [1, 1, 0, 1]
