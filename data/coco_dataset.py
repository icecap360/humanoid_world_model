from torchvision.datasets import CocoCaptions
from transformers import T5Tokenizer, T5EncoderModel
import random

class CustomCoco(CocoCaptions):
    def __init__(self, root, annFile, text_tokenizer, transform = None, target_transform = None, transforms = None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.text_tokenizer = text_tokenizer

    def __getitem__(self, index: int):

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # captions = []
        # for i in range(b):
        #     i_captions = []
        #     for j in range(len(metadata)):
        #         i_captions.append(metadata[j][i])
        #     # captions.append(i_captions)
        #     captions.append(random.choice(i_captions))
        captions = random.choice(target)
        target = self.text_tokenizer.tokenize(captions)

        return image, target
    
