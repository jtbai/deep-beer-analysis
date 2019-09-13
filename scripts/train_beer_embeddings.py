import pickle
import logging
import os
import json
import operator

import random
import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader

from pytoune.framework import Experiment as PytouneExperiment

from repository.mongo_extractor import MongoExtractor
from data_handling import create_vocabulary
from ml.beer_to_taste_skipgram import BeerToTasteSkipgram, format_examples_for_neural_net, generate_skip_gram_examples
from repository.mongo_extractor import format_checkins
from ml.similarities_callback import SimilaritiesCallback


MONGO_CONGIF_FILE_PATH = "config/mongo_connection_details.json"


def get_source_directory(directory_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)


def get_experiment_directory(directory_name):
    default_dir = get_source_directory('./results')
    dest_directory = os.environ.get('RESULTS_DIR', default_dir)
    return os.path.join(dest_directory, directory_name)


def main():
    logging.info("Getting MongoDB connection")
    mongo_extractor = MongoExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))

    logging.info("Getting checkins with tags")
    checkins = mongo_extractor.get_all_checking_with_tags_and_score()

    logging.info("Formatting checkins")
    checkins = list(format_checkins(checkins))

    logging.info("From beer to beer type")
    beer_to_beertype = {c.beer: c.beer_type for c in checkins}

    logging.info("Generating skipgram examples")
    skipgram_examples = list(generate_skip_gram_examples(checkins))

    logging.info("Generating vocabularies")
    beer_to_idx, idx_to_beer = create_vocabulary({e.beer for e in skipgram_examples})
    tag_to_idx, idx_to_tag = create_vocabulary({e.tag for e in skipgram_examples})

    logging.info("Generating examples for neural net")
    nn_examples = list(format_examples_for_neural_net(skipgram_examples, beer_to_idx, tag_to_idx))

    train_loader = DataLoader(
        nn_examples,
        batch_size=128,
        shuffle=True,
    )

    valid_loader = DataLoader(
        nn_examples,
        batch_size=128,
    )

    expt_dir = get_experiment_directory("LOL2")

    device_id = 1
    device = None
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id) # Fix bug where memory is allocated on GPU0 when ask to take GPU1.
        torch.cuda.manual_seed(42)
        device = torch.device('cuda:%d' % device_id)
        logging.info("Training on GPU %d" % device_id)
    else:
        logging.info("Training on CPU")


    model = BeerToTasteSkipgram(beer_to_idx, idx_to_beer, tag_to_idx, 5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    expt = PytouneExperiment(
        expt_dir,
        model,
        device=device,
        optimizer=optimizer,
        monitor_metric='val_loss',
        monitor_mode='min'
    )

    beers_to_check = {
        'Catnip',
        'Limoiloise',
        'Nordet IPA',
        'Moralité',
        'IPA du Nord-Est',
        'Herbe à Détourne'
    }

    callbacks = [
        SimilaritiesCallback(beer_to_idx, beers_to_check)
    ]


    try:
        expt.train(train_loader, valid_loader, callbacks=callbacks, seed=42, epochs=200)
        pass
    except KeyboardInterrupt:
        print('-' * 89)

    sorted_beer_names = [v[0] for v in sorted(beer_to_idx.items(), key=operator.itemgetter(1))]
    sorted_tag_names = [v[0] for v in sorted(tag_to_idx.items(), key=operator.itemgetter(1))]

    with open("vector_model/beers.tsv", 'w') as beer_file:
        for name in sorted_beer_names:
            beer_file.write("{}\t{}\n".format(name, beer_to_beertype[name]))

    with open("vector_model/beers_vectors.tsv", "w") as vector_file:
        for v in model.beer_embeddings.weight.data.tolist():
            vector_file.write("{}\n".format("\t".join([str(i) for i in v])))

    pickle.dump(beer_to_idx, open('./data/beer_vocab.pkl', 'wb'))
    pickle.dump(tag_to_idx, open('./data/tag_vocab.pkl', 'wb'))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()
