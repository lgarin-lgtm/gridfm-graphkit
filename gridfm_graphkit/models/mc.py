"""MC-dropout building blocks.

These are drop-in subclasses of standard layers that add **zero parameters** —
only a ``mc_dropout`` bool flag and a ``training`` override. Because the
``state_dict`` is identical to the plain layers, a checkpoint trained with the
plain model loads into the MC model and vice-versa. That is why there is a
single architecture (always MC-capable) plus a runtime/config switch, rather
than two architectures: any checkpoint loads in either mode.
"""

from torch import nn
from torch_geometric.nn import GPSConv, TransformerConv


class _MCActiveAtInference:
    """Mixin: keep this layer's dropout active at inference when ``mc_dropout``.

    MC-dropout needs dropout — and *only* dropout — to fire during prediction.
    Flipping the whole model to ``train()`` would also switch BatchNorm to batch
    statistics and corrupt outputs, so we override ``training`` per layer instead:
    it reads True iff the layer is genuinely training OR ``mc_dropout`` is set.
    """

    @property
    def training(self):
        return self._training or getattr(self, "mc_dropout", False)

    @training.setter
    def training(self, value):
        self._training = value


class MCDropout(_MCActiveAtInference, nn.Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_dropout = False


class MCGPSConv(_MCActiveAtInference, GPSConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_dropout = False


class MCTransformerConv(_MCActiveAtInference, TransformerConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_dropout = False
