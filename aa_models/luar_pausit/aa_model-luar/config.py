from transformers import PretrainedConfig


class LUARConfig(PretrainedConfig):
    model_type = "LUAR"

    def __init__(
        self,
        embedding_size: int = 512,
        use_memory_efficient_attention=False,
        q_bucket_size=512,
        k_bucket_size=1024,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size
        super().__init__(**kwargs)
