'''
TinyViT 
with 22k classes and highres support
variants: {5,11,21}m_{1,22}k_{224,384,512}

checkpoints:
https://github.com/wkcn/tinyvit?tab=readme-ov-file#model-zoo
'''

# ignore UserWarning: Overwriting tiny_vit_5m_224 in registry with models.tiny_vit.tiny_vit_5m_224
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="models.tiny_vit")

import yaml
import json
import torch
import time
from os import path
from PIL import Image
from data import build_transform
from config import get_config
from models.tiny_vit import TinyViT


class TinyViTInference:
    def __init__(self, variant="21m_22k_224", device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = get_config()

        # get variation-specific paths
        self.variant = variant
        cfg_path, weights_path = self.__get_variant_paths__()

        # load model config
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # get classes
        self.num_classes, self.classnames = self.__get_classes__()

        # update the image size config
        self.config.DATA.IMG_SIZE = self.cfg.get("DATA", {}).get("IMG_SIZE", 224)

        
        self.model = self.__get_model__().to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.model.eval()

        # build transform
        self.transform = build_transform(is_train=False, config=self.config)
        
    def __get_variant_paths__(self):
        variants = {
            "5m_1k_224": {
                "cfg": "configs/22kto1k/tiny_vit_5m_22kto1k.yaml",
                "weights": "checkpoints/tiny_vit_5m_22kto1k_distill.pth"
            },
            "11m_1k_224": {
                "cfg": "configs/22kto1k/tiny_vit_11m_22kto1k.yaml",
                "weights": "checkpoints/tiny_vit_11m_22kto1k_distill.pth"
            },
            "21m_1k_224": {
                "cfg": "configs/22kto1k/tiny_vit_21m_22kto1k.yaml",
                "weights": "checkpoints/tiny_vit_21m_22kto1k_distill.pth"
            },
            "21m_22k_224": {
                "cfg": "configs/22k_distill/tiny_vit_21m_22k_distill.yaml",
                "weights": "checkpoints/tiny_vit_21m_22k_distill.pth"
            },
            "21m_22k_384": {
                "cfg": "configs/higher_resolution/tiny_vit_21m_224to384.yaml",
                "weights": "checkpoints/tiny_vit_21m_22kto1k_384_distill.pth"
            },
            "21m_22k_512": {
                "cfg": "configs/higher_resolution/tiny_vit_21m_384to512.yaml",
                "weights": "checkpoints/tiny_vit_21m_22kto1k_512_distill.pth"
            }
        }

        paths = variants.get(self.variant)
        if paths is None:
            raise ValueError(f"Variant '{self.variant}' not found in available options.")
        
        return paths["cfg"], paths["weights"]

    def __get_classes__(self):
        # get classes and classnames
        dataset = self.cfg.get("DATA", {}).get("DATASET", None)
        if dataset == "imagenet22k":
            num_classes = 21841
            classnames_path = "configs/imagenet22k_classes.json"
        else:
            num_classes = 1000
            classnames_path = "configs/imagenet1k_classes.json"

        with open(classnames_path, "r") as f:
            classnames = json.load(f)
        return num_classes, classnames

    def __get_model__(self):
        model_cfg = self.cfg["MODEL"]
        tiny_cfg = model_cfg["TINY_VIT"]

        model = TinyViT(
            img_size=self.config.DATA.IMG_SIZE,
            in_chans=3,
            num_classes=self.num_classes,
            embed_dims=tiny_cfg["EMBED_DIMS"],
            depths=tiny_cfg["DEPTHS"],
            num_heads=tiny_cfg["NUM_HEADS"],
            window_sizes=tiny_cfg["WINDOW_SIZES"],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=model_cfg["DROP_PATH_RATE"],
            use_checkpoint=False
        )
        return model

    def predict(self, image, topk=5, print_results=True):
        start_time = time.time()
        # add batch dimension: (1, 3, img_size, img_size)
        batch = self.transform(image)[None].to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(batch)
        self.probs = torch.softmax(logits, -1)

        self.prediction_time = time.time() - start_time

        scores_np, inds_np = self.topk(topk, print_results=print_results)
        return scores_np, inds_np

    def topk(self, topk, print_results=True):
        # Top predictions
        scores, inds = self.probs.topk(topk, largest=True, sorted=True)

        # Convert to NumPy arrays
        scores_np = scores[0].cpu().numpy()
        inds_np = inds[0].cpu().numpy()

        if print_results:
            print(f'image: {path.basename(image_path)} ({variant}) elapsed: {self.prediction_time:.3f} s')
            for score, ind in zip(scores_np, inds_np):
                print(f'{classifier.classnames[str(ind)][0]}: {score:.2f}')

        return scores_np, inds_np


if __name__ == "__main__":
    
    image_path = 'images/dog.jpg'
    variant = "21m_22k_224"
    device = "cpu"  # 'cuda'

    classifier = TinyViTInference(variant=variant, device=device)

    image = Image.open(image_path)
    scores_np, inds_np = classifier.predict(image, topk=5, print_results=False)

    # top result, [0] sometimes the classnames contain more than one string)
    print(f"{classifier.classnames[str(inds_np[0])][0]}: {scores_np[0]:.2f}")
