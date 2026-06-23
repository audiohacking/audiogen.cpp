import torch


def create_mask_from_length(lengths: torch.Tensor, max_length: int | None = None):
    lengths = torch.as_tensor(lengths)
    if lengths.ndim == 0:
        lengths = lengths.unsqueeze(0)
    lengths = lengths.long()
    if max_length is None:
        if lengths.numel() == 0:
            max_length = 0
        else:
            max_length = int(lengths.max().item())
    idxs = torch.arange(max_length, device=lengths.device).reshape(1, -1)
    mask = idxs < lengths.view(-1, 1)
    return mask


def convert_pad_shape(pad_shape: list[list[int]]):
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


def create_alignment_path(duration: torch.Tensor, mask: torch.Tensor):
    device = duration.device
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = create_mask_from_length(cum_duration_flat, t_y).float()
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(
        path, convert_pad_shape([[0, 0], [1, 0], [0, 0]])
    )[:, :-1]
    path = path * mask
    return path


def trim_or_pad_length(x: torch.Tensor, target_length: int, length_dim: int):
    current_length = x.shape[length_dim]
    if current_length > target_length:
        slices = [slice(None)] * x.ndim
        slices[length_dim] = slice(0, target_length)
        return x[tuple(slices)]
    elif current_length < target_length:
        pad_shape = list(x.shape)
        pad_shape[length_dim] = target_length - current_length
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=length_dim)
    return x
