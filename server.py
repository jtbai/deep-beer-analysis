import json
import pickle
import torch
from flask import Flask, escape, request
from ml.beer_to_taste_skipgram import BeerToTasteSkipgram
from repository.mongo_extractor import MongoExtractor
import re

MONGO_CONGIF_FILE_PATH = "config/cache_mongo_connection_details.json"
mongo_extractor = MongoExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))

app = Flask(__name__)

beer_to_idx = pickle.load(open('./data/beer_vocab.pkl', 'rb'))
idx_to_beer = {i: b for b, i in beer_to_idx.items()}
tag_to_idx = pickle.load(open('./data/tag_vocab.pkl', 'rb'))

model = BeerToTasteSkipgram(beer_to_idx, idx_to_beer, tag_to_idx, 5)
model.load_state_dict(torch.load('./scripts/results/LOL2/checkpoint.ckpt'))
model = model.eval()


def is_regional_beer(beer, region):
    location = mongo_extractor.get_beer_location(beer)
    print(beer, location)
    return re.match(location, region)

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
    # similar_beers = model.get_n_similar_beers(beer_tensor, 100)
    # dissimilar_beers = model.get_n_not_similar_beers(beer_tensor, 100)

    beer_similarities = model.get_beer_similarities(beer_tensor)
    local_beers = mongo_extractor.get_local_beers("QC")
    print(local_beers)
    similar_local_beer = [(name, score) for name, score in beer_similarities if name in local_beers]

    similar_beers = [b for b, _ in similar_local_beer][-10:]
    dissimilar_beers = [b for b, _ in similar_local_beer][:10]

    return '<h1>{} - {}</h1>' \
           '<h1>Similar:</h1> {} <br/><br/> ' \
           '<h1>Dissimilar:</h1> {}'.format(mongo_extractor.get_beer_brewery(name), name,
        format_beer_list_with_brewery(similar_beers),
        format_beer_list_with_brewery(dissimilar_beers))

if __name__== "__main__":
    app.run(debug=True)