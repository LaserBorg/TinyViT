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
    def set_variant(self, variant):
        """Switch to a different model variant and reload model, config, and class labels."""
        self.variant = variant
        self.variant_info = TinyViTInference.variants.get(self.variant)
        if self.variant_info is None:
            raise ValueError(f"Variant '{self.variant}' not found in available options.")

        # load model config
        with open(self.variant_info["cfg"], "r") as f:
            self.cfg = yaml.safe_load(f)

        # get classes
        self.num_classes, self.classnames = self.__get_classes__()

        # update the image size config
        if self.variant_info["is_finetuned"]:
            checkpoint = torch.load(self.variant_info["weights"], map_location=self.device, weights_only=True)
            self.config.DATA.IMG_SIZE = checkpoint.get('img_size', 224)
        else:
            self.config.DATA.IMG_SIZE = self.cfg.get("DATA", {}).get("IMG_SIZE", 224)

        self.model = self.__get_model__().to(self.device)
        self.__load_model_weights__()
        self.model.eval()
        self.transform = build_transform(is_train=False, config=self.config)
    # Shared variants dictionary for all instances
    variants = {
        "5m_1k_224": {
            "cfg": "configs/22kto1k/tiny_vit_5m_22kto1k.yaml",
            "weights": "checkpoints/tiny_vit_5m_22kto1k_distill.pth",
            "num_classes": 1000,
            "classlabels": "configs/imagenet1k_classes.json",
            "is_finetuned": False
        },
        "11m_1k_224": {
            "cfg": "configs/22kto1k/tiny_vit_11m_22kto1k.yaml",
            "weights": "checkpoints/tiny_vit_11m_22kto1k_distill.pth",
            "num_classes": 1000,
            "classlabels": "configs/imagenet1k_classes.json",
            "is_finetuned": False
        },
        "21m_1k_224": {
            "cfg": "configs/22kto1k/tiny_vit_21m_22kto1k.yaml",
            "weights": "checkpoints/tiny_vit_21m_22kto1k_distill.pth",
            "num_classes": 1000,
            "classlabels": "configs/imagenet1k_classes.json",
            "is_finetuned": False
        },
        "21m_22k_224": {
            "cfg": "configs/22k_distill/tiny_vit_21m_22k_distill.yaml",
            "weights": "checkpoints/tiny_vit_21m_22k_distill.pth",
            "num_classes": 21841,
            "classlabels": "configs/imagenet22k_classes.json",
            "is_finetuned": False
        },
        "21m_22k_384": {
            "cfg": "configs/higher_resolution/tiny_vit_21m_224to384.yaml",
            "weights": "checkpoints/tiny_vit_21m_22kto1k_384_distill.pth",
            "num_classes": 1000,
            "classlabels": "configs/imagenet1k_classes.json",
            "is_finetuned": False
        },
        "21m_22k_512": {
            "cfg": "configs/higher_resolution/tiny_vit_21m_384to512.yaml",
            "weights": "checkpoints/tiny_vit_21m_22kto1k_512_distill.pth",
            "num_classes": 1000,
            "classlabels": "configs/imagenet1k_classes.json",
            "is_finetuned": False
        }
    }

    def __init__(self, variant="21m_22k_224", device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = get_config()

        # get variation-specific paths and info
        self.variant = variant
        self.variant_info = TinyViTInference.variants.get(self.variant)
        if self.variant_info is None:
            raise ValueError(f"Variant '{self.variant}' not found in available options.")

        # load model config
        with open(self.variant_info["cfg"], "r") as f:
            self.cfg = yaml.safe_load(f)

        # get classes
        self.num_classes, self.classnames = self.__get_classes__()

        # update the image size config
        if self.variant_info["is_finetuned"]:
            # For finetuned models, get img_size from checkpoint
            checkpoint = torch.load(self.variant_info["weights"], map_location=self.device, weights_only=True)
            self.config.DATA.IMG_SIZE = checkpoint.get('img_size', 224)
        else:
            # For official models, get from config
            self.config.DATA.IMG_SIZE = self.cfg.get("DATA", {}).get("IMG_SIZE", 224)

        self.model = self.__get_model__().to(self.device)

        # Load model weights
        self.__load_model_weights__()

        self.model.eval()

        # build transform
        self.transform = build_transform(is_train=False, config=self.config)
    
    def __get_classes__(self):
        # get classes and classnames from variant info
        if self.variant_info["is_finetuned"]:
            # For finetuned models, load from checkpoint and classlabels file
            checkpoint = torch.load(self.variant_info["weights"], map_location=self.device, weights_only=True)
            num_classes = checkpoint.get('num_classes', self.variant_info["num_classes"])
        else:
            # For official models, use predefined num_classes
            num_classes = self.variant_info["num_classes"]

        # Load class names from classlabels file
        with open(self.variant_info["classlabels"], "r") as f:
            classnames = json.load(f)
        
        return num_classes, classnames

    def __load_model_weights__(self):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(self.variant_info["weights"], map_location=self.device, weights_only=True)
        
        if self.variant_info["is_finetuned"]:
            # For finetuned models, use 'model_state_dict' key
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # For official models, use 'model' key
            self.model.load_state_dict(checkpoint['model'], strict=False)

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
            print(f'model: {self.variant} elapsed: {self.prediction_time:.3f} s')
            for score, ind in zip(scores_np, inds_np):
                print(f'{self.classnames[str(ind)][0]}: {score:.2f}')

        return scores_np, inds_np


if __name__ == "__main__":
    
    image_path = 'images/dog.jpg'
    

    # PRETRAINED
    print("Testing with pretrained checkpoint:")
    variant = "21m_22k_384"
    device = "cuda"  # 'cpu'

    classifier = TinyViTInference(variant=variant, device=device)

    image = Image.open(image_path)
    scores_np, inds_np = classifier.predict(image, topk=5, print_results=False)

    # top result, [0] sometimes the classnames contain more than one string)
    print(f"pretrained model - {classifier.classnames[str(inds_np[0])][0]}: {scores_np[0]:.2f}")


    # -------------------------------------------------------------------------------------------------
    # FINETUNED
    print("Testing with finetuned checkpoint:")

    classifier = TinyViTInference(device=device)

    classifier.variants["21m_384_finetuned"] = {
        "cfg": "configs/higher_resolution/tiny_vit_21m_224to384.yaml",
        "weights": "checkpoints/finetune/tiny_vit_21m_384_finetuned.pth",
        "num_classes": 5,  # or be loaded from checkpoint
        "classlabels": "checkpoints/finetune/finetuned_classes.json",
        "is_finetuned": True
    }
    classifier.set_variant("21m_384_finetuned")

    scores_np_ft, inds_np_ft = classifier.predict(image, topk=3, print_results=False)
    print(f"Finetuned model - {classifier.classnames[str(inds_np_ft[0])][0]}: {scores_np_ft[0]:.2f}")
    print(f"Available classes in finetuned model: {[labels[0] for labels in classifier.classnames.values()]}")
