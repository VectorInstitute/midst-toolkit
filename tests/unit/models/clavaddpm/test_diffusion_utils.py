import torch

from midst_toolkit.models.clavaddpm.diffusion_utils import exists


def test_exists() -> None:
    assert exists(torch.tensor(0))
    assert not exists(None)
