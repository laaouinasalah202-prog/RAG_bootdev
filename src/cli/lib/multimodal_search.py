from sentence_transformers import SentenceTransformer


class MultimodelSearch:
    def __init__(self, model_name="jinaai/jina-clip-v2"):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )

    def embed_image(self, img_path):
        self.image_embeddings = self.model.encode(
            [img_path],
            normalize_embeddings=True,
        )
        return self.image_embeddings


def verify_image_embedding(img_path):
    multimodel = MultimodelSearch()
    embeddings = multimodel.embed_image(img_path)

    print(f"Embedding shape: {embeddings.shape}")
