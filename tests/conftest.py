class FakeEmbeddingProvider:
    provider_id = "test-provider"

    def embed(self, text):
        return (1.0, 0.0)
