import json
import numpy as np
from PIL import Image
import torch

import xpatch_helpers

def open_image(image_path, image_size):
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    new_width = image_size
    new_height = int((image_size / width) * height)
    image = image.resize((new_width, new_height))
    return image

def format_prompt(samples, modality, remove_last=True, special_token=":", image_root="", image_size=224):
    prompt, images = "", []
    query_fn, answer_fn = lambda x: f"Q{special_token}{x}\nA{special_token}", lambda x: f"{x}\n\n"
    for i, sample in enumerate(samples):
        if modality == "image":
            user = f"<image>"
            image_path = f"{image_root}/{sample['image']}"
            image = open_image(image_path, image_size)
            images.append(image)
        elif modality == "text":
            user = sample["input"]
        else:
            raise NotImplementedError
        prompt += query_fn(user)
        asst = sample.get("output", "")
        if not remove_last or (i != len(samples) - 1):
            prompt += answer_fn(asst)
    return prompt, images

class xPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_file,
        processor,
        num_examples=5,
        src_modality="text",
        tgt_modality="image",
        special_token=":",
        image_root="data",
        image_size=224,
        seed=1,
        include_regular=True,
        src_match=None,
        tgt_match=None,
        image_offset=0
    ):
        super().__init__()

        np.random.seed(seed)
        self.data = json.load(open(annotation_file))
        self.processor = processor
        self.num_examples = num_examples
        self.src_modality = src_modality
        self.tgt_modality = tgt_modality
        self.image_root = image_root
        self.image_size = image_size
        self.include_regular = include_regular
        self.image_offset = image_offset
        self.random_state = {}

        if type(self.data) is dict:
            self.data = self.data["data"]
            
        # processor fails when first image doesn't have image
        if src_modality == "text" and tgt_modality == "image":
            self.src_idx, self.tgt_idx = 1, 0
        else:
            self.src_idx, self.tgt_idx = 0, 1
        
        if src_match is None:
            self.src_match = -3 if (self.include_regular and num_examples > 0) else -1
        else:
            self.src_match = src_match
        if tgt_match is None:
            self.tgt_match = -1
        else:
            self.tgt_match = tgt_match
        
        self.special_token = special_token
        self.find_token_kwargs = {
            "token": self.special_token,
            "token_ids": [processor.tokenizer(f"A{self.special_token}")["input_ids"][-1]]
        }

    def get_text_images(self, idx, train_idxs):
        test_sample = self.data[idx]
        test_text, test_images = format_prompt(
            [test_sample], 
            self.tgt_modality, 
            remove_last=True,
            special_token=self.special_token,
            image_root=self.image_root,
            image_size=self.image_size
        )
        
        # Select random ICL examples
        train_samples = [self.data[i] for i in train_idxs]
        train_text, train_images = format_prompt(
            train_samples, 
            self.src_modality, 
            remove_last=False,
            special_token=self.special_token,
            image_root=self.image_root,
            image_size=self.image_size
        )
        
        # We copy the task vector only 
        # from an unrelated ICL example (match = -3)
        # Run 1 = ICL + test sample
        # Run 2 = ZS test sample
        # text = [train_text, test_text]
        # images = [train_images, test_images]

        # Set src and tgt
        if self.include_regular:
            train_text += test_text
            train_images += test_images
        text = [None, None]
        images = [None, None]
        text[self.src_idx] = train_text
        text[self.tgt_idx] = test_text
        images[self.src_idx] = train_images
        images[self.tgt_idx] = test_images

        if len(sum(images, [])) == 0:
          images = None
        
        return text, images
    
    def get_meta(self, meta, batch, flag, find_idx=None):
        idx = getattr(self, f"{flag}_idx")
        match = getattr(self, f"{flag}_match")
        if find_idx is None:
            find_idx = idx
        token = xpatch_helpers.find_token(
            self.processor, 
            batch["input_ids"][find_idx],
            match=match,
            **self.find_token_kwargs
        )
        if getattr(self, f"{flag}_modality") == "image":
            token += self.image_offset
        meta[f"{flag}_token"] = token
        meta[f"{flag}_idx"] = idx
        return meta

    def get_image_or_none(self, images, idx):
        if images is None:
            return None
        images = images[idx]
        if len(images) == 0:
            return None
        else:
            return images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, flag=None):
        # since __getitem__ may be called twice
        # ensure that train_idxs are the same
        if idx in self.random_state:
            np.random.set_state(self.random_state[idx])
        else:
            self.random_state[idx] = np.random.get_state()
        all_idxs = [i for i in range(len(self.data)) if i != idx]
        train_idxs = np.random.permutation(all_idxs)[:self.num_examples].tolist()

        meta = {}
        text, images = self.get_text_images(idx, train_idxs)
        if flag == "src":
            batch = self.processor(text=text[self.src_idx], images=self.get_image_or_none(images, self.src_idx), return_tensors="pt")
            meta = self.get_meta(meta, batch, "src", 0)
        elif flag == "tgt":
            batch = self.processor(text=text[self.tgt_idx], images=self.get_image_or_none(images, self.tgt_idx), return_tensors="pt")
            meta = self.get_meta(meta, batch, "tgt", 0)
        else:
            batch = self.processor(text=text, images=images, padding=True, return_tensors="pt")
            meta = self.get_meta(meta, batch, "src")
            meta = self.get_meta(meta, batch, "tgt")
        meta["train_idxs"] = train_idxs
        meta["labels"] = self.processor(text=self.data[idx]["output"], return_tensors="pt")["input_ids"]
        return batch, meta