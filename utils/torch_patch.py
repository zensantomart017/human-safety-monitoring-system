import torch

# PyTorch 2.6+ safe globals / weights_only workaround
# Overrides torch.load to default weights_only=False so that YOLO models can be unpickled safely.
if not hasattr(torch.load, '__patched__'):
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    _patched_torch_load.__patched__ = True
    torch.load = _patched_torch_load
