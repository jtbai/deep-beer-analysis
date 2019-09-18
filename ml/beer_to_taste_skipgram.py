from typing import List
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from repository.mongo_beer_extractor import Checkin
import numpy as np
import json

SkipgramExample = namedtuple('SkipgramExample', ['beer', 'tag'])

class BeerToTasteSkipgram(nn.Module):
    def __init__(self, beer_to_idx, idx_to_beer, tag_to_idx, embedding_dimension):
        super().__init__()
        self.beer_to_idx = beer_to_idx
        self.idx_to_beer = idx_to_beer
        self.tag_to_idx = tag_to_idx
        self.idx_to_tag = {idx:tag for tag, idx  in tag_to_idx.items()}
        self.embedding_dimension = embedding_dimension
        self.beer_embeddings = nn.Embedding(len(beer_to_idx), embedding_dimension)
        self.tag_embeddings = nn.Embedding(len(tag_to_idx), embedding_dimension)
        self.loss_function = nn.CrossEntropyLoss()

    def get_n_similar_beers(self, beer, n=10):
        beer_embedding = self.beer_embeddings(beer)
        others = self.beer_embeddings.weight
        similarities = F.cosine_similarity(beer_embedding, others)
        values, indices = similarities.sort()

        topk_indices = indices[-(n+1):].tolist()  # Add one more indice so we can skip self token
        topk_indices.reverse()

        topk_values = values[-(n+1):].tolist()  # Add one more indice so we can skip self token
        topk_values.reverse()

        return [(self.idx_to_beer[i], v) for i, v in zip(topk_indices[1:], topk_values[1:])]  # Here we skip the self token


    def get_n_not_similar_beers(self, beer, n=10):
        beer_embedding = self.beer_embeddings(beer)
        others = self.beer_embeddings.weight
        similarities = F.cosine_similarity(beer_embedding, others)
        values, indices = similarities.sort()

        topk_indices = indices[:n].tolist()

        topk_values = values[:n].tolist()

        return [(self.idx_to_beer[i], v) for i, v in zip(topk_indices[1:], topk_values[1:])]  # Here we skip the self token

    def get_beer_similarities(self, beer):
        beer_embedding = self.beer_embeddings(beer)
        others = self.beer_embeddings.weight
        similarities = F.cosine_similarity(beer_embedding, others)
        values, indices = similarities.sort()

        return [(self.idx_to_beer[i], v) for i, v in zip(indices.tolist()[1:], values.tolist()[1:])]  # Here we skip the self token

    def create_beer_vector(self, tags_with_proportion):
        total_proprotion = sum([proportion for _, proportion in tags_with_proportion])
        beer_vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        for idx, proportion in tags_with_proportion:
            beer_vector += np.array(self.tag_embeddings(idx).squeeze(0).detach().numpy()*proportion/total_proprotion)

        beer_tensor = torch.from_numpy(beer_vector)
        beer_similarity = torch.matmul(beer_tensor, self.beer_embeddings.weight.transpose(0, 1)).squeeze().detach().numpy()

        return [(self.idx_to_beer[idx], similarity) for idx, similarity in enumerate(beer_similarity)]


    def get_beer_flavors(self, beer, topn):
        beer_embedding = self.beer_embeddings(beer)
        flavor_similarity = torch.matmul(beer_embedding, self.tag_embeddings.weight.transpose(0, 1)).squeeze(0).detach().numpy()
        top_fice_flavor_index = np.argsort(flavor_similarity)[-topn:]
        flavors = [(self.idx_to_tag[idx], flavor_similarity[idx]) for idx in top_fice_flavor_index]
        top_flavors = list(reversed(flavors))
        return top_flavors

    def forward(self, beers):
        return torch.matmul(self.beer_embeddings(beers), self.tag_embeddings.weight.transpose(0, 1))


def generate_skip_gram_examples(checkins: List[Checkin]):
    for checkin in checkins:
        for tag in checkin.tags:
            yield SkipgramExample(
                beer=checkin.beer,
                tag=tag
            )


def format_examples_for_neural_net(examples, beer_to_idx, tag_to_idx):
    for example in examples:
        yield (beer_to_idx[example.beer], tag_to_idx[example.tag])
