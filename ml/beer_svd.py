from typing import List

from sklearn.decomposition import PCA, SparsePCA
import numpy as np

from repository.mongo_beer_extractor import Checkin
from data_handling import create_vocabulary

class BeerToTastePCA:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def __build_cooccurrence_matrix(self, examples: List[Checkin]):
        self.beer2idx, self.idx2beer = create_vocabulary({e.beer for e in examples})
        self.tag2idx, self.idx2tag = create_vocabulary({tag for e in examples for tag in e.tags})
        self.cooccurrence_matrix = np.zeros((len(self.beer2idx), len(self.tag2idx)))
        for example in examples:
            for tag in example.tags:
                self.cooccurrence_matrix[self.beer2idx[example.beer]][self.tag2idx[tag]] += 1

    def fit(self, examples: List[Checkin]):
        self.__build_cooccurrence_matrix(examples)
        self.pca.fit(self.cooccurrence_matrix)
        self.embeddings = self.pca.transform(self.cooccurrence_matrix)

    def get_n_similar_beers(self, beer, n=10):
        beer_embedding = self.embeddings[self.beer2idx[beer]]
        others = self.embeddings
        scores = np.dot(beer_embedding.reshape(1, -1), others.reshape(self.n_components, -1))[0]
        # scores = np.dot(self.cooccurrence_matrix[self.beer2idx[beer]].reshape(1, -1), self.cooccurrence_matrix.transpose())[0]

        indices = scores.argsort()
        scores.sort()

        topk_indices = indices[-(n+1):].tolist()  # Add one more indice so we can skip self token
        topk_indices.reverse()

        topk_values = scores[-(n+1):].tolist()  # Add one more indice so we can skip self token
        topk_values.reverse()
        import pdb; pdb.set_trace()

        return [(self.idx2beer[i], v) for i, v in zip(topk_indices[1:], topk_values[1:])]  # Here we skip the self token


    def get_n_not_similar_beers(self, beer, n=10):
        # TODO
        pass
    

