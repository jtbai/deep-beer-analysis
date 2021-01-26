import json
import logging
import random

import numpy as np

from repository.mongo_beer_extractor import MongoBeerExtractor, format_checkins
from ml.beer_svd import BeerToTastePCA


MONGO_CONGIF_FILE_PATH = "config/mongo_connection_details.json"


def main():
    logging.info("Getting MongoDB connection")
    mongo_extractor = MongoBeerExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))

    logging.info("Getting checkins with tags")
    checkins = mongo_extractor.get_all_checking_with_tags_and_score()

    logging.info("Formatting checkins")
    checkins = list(format_checkins(checkins))

    logging.info("Training PCA")
    model = BeerToTastePCA(n_components=260)
    model.fit(checkins)

    similar_words = model.get_n_similar_beers('Catnip', n=5)
    logging.info("Most similar beers:")
    logging.info(similar_words)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(42)
    random.seed(42)
    main()
