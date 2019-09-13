import json
import pickle
import torch
from flask import Flask, escape, request
from ml.beer_to_taste_skipgram import BeerToTasteSkipgram
from repository.mongo_extractor import MongoExtractor

MONGO_CONGIF_FILE_PATH = "config/mongo_connection_details.json"
mongo_extractor = MongoExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))

app = Flask(__name__)

beer_to_idx = pickle.load(open('./data/beer_vocab.pkl', 'rb'))
idx_to_beer = {i: b for b, i in beer_to_idx.items()}
tag_to_idx = pickle.load(open('./data/tag_vocab.pkl', 'rb'))

model = BeerToTasteSkipgram(beer_to_idx, idx_to_beer, tag_to_idx, 5)
model.load_state_dict(torch.load('./scripts/results/LOL2/checkpoint.ckpt'))

def format_beer_list_with_brewery(beers):
    str = "<ul>{}</ul>"
    beer_strs = ["<li>{}: {}</li>".format(mongo_extractor.get_beer_brewery(beer), beer) for beer in beers]
    return str.format(" ".join(beer_strs))


def format_beer_list(beers):
    str = "<ul>{}</ul>"
    beer_strs = ["<li><a href='/beer?name={0}'>{0}</a></li>".format(beer) for beer in beers]
    return str.format(" ".join(beer_strs))

@app.route('/')
def home():
    return format_beer_list([b for b, _ in beer_to_idx.items()])

@app.route('/beer')
def beer():
    name = request.args.get("name", "Catnip")
    beer_idx = beer_to_idx[name]
    beer_tensor = torch.LongTensor([beer_idx])
    similar_beers = model.get_n_similar_beers(beer_tensor)
    dissimilar_beers = model.get_n_not_similar_beers(beer_tensor)
    similar_beers = [b for b, _ in similar_beers]
    dissimilar_beers = [b for b, _ in dissimilar_beers]
    return '<h1>Similar:</h1> {} <br/><br/> <h1>Dissimilar:</h1>: {}'.format(format_beer_list_with_brewery(similar_beers), format_beer_list_with_brewery(dissimilar_beers))
