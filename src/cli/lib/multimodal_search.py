from PIL import Image
from sentence_transformers import SentenceTransformer, util
from outiles import load_movie 


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [f'{doc["title"]}: {doc["description"]}' for doc in self.documents]
        self.text_embeddings = (
            self.model.encode(self.texts, show_progress_bar=True) if self.texts else None
        )

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]

        results = []
        for i, (doc, sim) in enumerate(zip(self.documents, similarities)):
            results.append(
                {
                    "doc_id": doc.get("id", i),
                    "title": doc["title"],
                    "description": doc["description"],
                    "cosine_sim": float(sim),
                }
            )

        results.sort(key=lambda item: item["cosine_sim"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    movies = load_movie()["movies"]
    search = MultimodalSearch(documents=movies)
    return search.search_with_image(image_path)