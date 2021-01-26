import logging

import torch
from poutyne.framework.callbacks import Callback


class SimilaritiesCallback(Callback):
    def __init__(self, beer_to_idx, beers_to_check):
        super(SimilaritiesCallback, self).__init__()
        self.beer_to_idx = beer_to_idx
        self.beers_to_check = beers_to_check

    def __compute_stats(self):
        for word in self.beers_to_check:
            logging.info(word)
            word_idx = self.beer_to_idx[word]
            word_tensor = torch.LongTensor([word_idx])
            if torch.cuda.is_available():
                word_tensor = word_tensor.cuda(0)

            similar_words = self.model.model.get_n_similar_beers(word_tensor, n=5)
            logging.info("Most similar beers:")
            logging.info(similar_words)

            logging.info("Most dissimilar beers:")
            not_similar_words = self.model.model.get_n_not_similar_beers(word_tensor, n=5)
            logging.info(not_similar_words)

    def on_epoch_end(self, epoch, logs):
        self.__compute_stats()

    def on_train_end(self, logs):
        self.__compute_stats()
