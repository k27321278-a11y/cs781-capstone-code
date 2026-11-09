"""Layer-wise partial B1 adaptation: only encoder layers 5–6 and head train."""
from phase5_b1_train import B1Classifier


class PartialB1Classifier(B1Classifier):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        for block in self.backbone.encoder:
            block.attn.wqkv = block.attn.wqkv.base
            block.attn.wo = block.attn.wo.base
            for name in ("w1", "w2", "w3"):
                setattr(block.ffn, name, getattr(block.ffn, name).base)
        for param in self.parameters():
            param.requires_grad_(False)
        for block in self.backbone.encoder[-2:]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in self.head.parameters():
            param.requires_grad_(True)
