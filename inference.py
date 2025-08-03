"""Model Inference."""

# ignore UserWarning: Overwriting tiny_vit_5m_224 in registry with models.tiny_vit.tiny_vit_5m_224
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="models.tiny_vit")

import torch
import numpy as np
from PIL import Image

from models.tiny_vit import TinyViT  # tiny_vit_21m_224, tiny_vit_11m_224 , tiny_vit_5m_224
from data import build_transform, imagenet22k_classnames  # imagenet_classnames
from config import get_config

config = get_config()
executor = 'cuda'  # 'cpu'


# model = tiny_vit_21m_224(pretrained=False) # don't download weights
# model_path = "checkpoints/tiny_vit_21m_22kto1k_distill.pth"

# TODO: load from "checkpoints/22k/tiny_vit_21m_22k_distill.yaml"
model = TinyViT(
    img_size=224,
    in_chans=3,
    num_classes=21841,
    embed_dims=[96, 192, 384, 576],
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 18],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.1,
    use_checkpoint=False
)
model_path = "checkpoints/22k/tiny_vit_21m_22k_distill.pth"


checkpoint = torch.load(model_path, map_location=executor)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Load Image
fname = 'images/bee.jpg'
image = Image.open(fname)
transform = build_transform(is_train=False, config=config)

# (1, 3, img_size, img_size)
batch = transform(image)[None]

# Inference
with torch.no_grad():
    logits = model(batch)

# Top-5 predictions
probs = torch.softmax(logits, -1)
scores, inds = probs.topk(5, largest=True, sorted=True)

print('=' * 30)
print(fname)
for score, ind in zip(scores[0].numpy(), inds[0].numpy()):
    print(f'{imagenet22k_classnames[ind]}: {score:.2f}')
