import os
import sys

from sentence_transformers import SentenceTransformer


class LUAR_PAUSIT:
    def __init__(self, device) -> None:
        current_dir = os.path.dirname(__file__)

        # Add the directory to sys.path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        self.model = SentenceTransformer(
            os.path.join(current_dir, "aa_model-luar"), device=device
        )

    def encode(self, paragraphs, batch_size=16):
        """ """
        return self.model.encode(
            paragraphs,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
