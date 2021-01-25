import json
import pickle
import torch
from flask import Flask, escape, request
from ml.beer_to_taste_skipgram import BeerToTasteSkipgram
from repository.mongo_beer_extractor import MongoBeerExtractor
from repository.mongo_checkin_extractor import MongoCheckinExtractor
import re
import html


BEER_MONGO_CONGIF_FILE_PATH = "config/beer_mongo_connection_details.json"
CHECKIN_MONGO_CONGIF_FILE_PATH = "config/checkin_mongo_connection_details.json"
beer_extractor = MongoBeerExtractor.get_connection(json.load(open(BEER_MONGO_CONGIF_FILE_PATH)))
checking_extractor = MongoCheckinExtractor.get_connection(json.load(open(CHECKIN_MONGO_CONGIF_FILE_PATH)))

app = Flask(__name__)

beer_to_idx = pickle.load(open('./data/beer_vocab.pkl', 'rb'))
idx_to_beer = {i: b for b, i in beer_to_idx.items()}
tag_to_idx = pickle.load(open('./data/tag_vocab.pkl', 'rb'))

model = BeerToTasteSkipgram(beer_to_idx, idx_to_beer, tag_to_idx, 20)
model.load_state_dict(torch.load('./vector_model/embedding_model'))
model = model.eval()

def format_top_taste(tastes):
    total_proximity = sum([x[1] for x in tastes])
    max_proximity = max([x[1] for x in tastes])
    output = "<ul>"
    for taste, proximity in tastes:
       output += "<li>{} : {:2.0f}</li>".format(taste, proximity/total_proximity*100)
    output += "</ul>"

    return output

def generate_html_header(title):
    html_header = "<html><head><title>beer2vec: {}</title></head>".format(title)
    html_header +='<link href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">'
    html_header += "<body>"
    return html_header


def format_dictionary_for_select_dropdown(idx_to_value):
    options = "<option value=-1></option>"
    options += " ".join(["<option value={}>{}</option>".format(idx, value) for idx, value in idx_to_value.items()])
    return options

def is_regional_beer(beer, region):
    location = beer_extractor.get_beer_location(beer)
    print(beer, location)
    return re.match(location, region)

def prepare_request_data_for_requested_profile(request_data):
    print(request_data)
    requested_profile = []
    for taste_number in range(5):
        taste_field = "profile_{}".format(taste_number)
        proportion_field = "proportion_{}".format(taste_number)
        if taste_field in request_data:
            if request_data[taste_field] != '-1':
                requested_profile.append((int(request_data[taste_field]), float(request_data[proportion_field])))

    return requested_profile

def format_beer_with_base_info(beers):
    str = "<ul>{}</ul>"
    beer_strs = ["<li><b>{0}</b>: <a href='/beer/{1}'>{1}</a> ({2}) | {3}</li>".format(beer_extractor.get_beer_brewery(beer), html.escape(beer), beer_extractor.get_beer_type(beer), beer_extractor.get_beer_score(beer)) for beer in beers]
    beer_strs.sort()
    return str.format(" ".join(beer_strs))

def format_beer_list(beers):
    str = "<ul>{}</ul>"
    beer_strs = ["<li><a href='/beer/{0}'>{0}</a></li>".format(html.escape(beer)) for beer in beers]
    return str.format(" ".join(beer_strs))

def format_checkin(checkin_list):
    str = "<ul>{}</ul>"
    review_str = ["<li><b>{}</b> ({}): {} </li>".format(checkin['user_name'][:2]+"*****"+checkin['user_name'][-1:], checkin['score'], checkin['review']) for checkin in checkin_list]
    return str.format(" ".join(review_str))

@app.route('/')
def home():
    local = request.args.get("local", "")
    local_beers = beer_extractor.get_local_beers(local)
    html_output = generate_html_header("Accueil")
    html_output += "<h1><a href=/similar_beer>Build a beer</a></h1>" \
                  "<h2><a href=/?local=QC>Québec</a> | <a href=/?local=Canada>Canada</a> | <a href=/?local=Belgium>Belgique</a> | <a href=/?local=VT>Vermont</a> | <a href=/?local=>All</a></h2>"
    html_output += format_beer_with_base_info([b for b, _ in beer_to_idx.items() if b in local_beers])
    html_output += "</body></html>"
    return html_output

@app.route('/similar_beer', methods=["GET", "POST"])
def similar_beer():
    if request.method == "POST":
        requested_profile = prepare_request_data_for_requested_profile(request.form)
        # test_values = [(113, 50), (88, 20), (52, 20), (13, 10)]  # (IPA, fruité, avec un peu de bois)
        # test_values =  [(250,20),(237,30),(58,50)] # (Stout vanille en barique)
        print(requested_profile)
        proportion_tensors = [(torch.LongTensor([idx]), proportion) for idx, proportion in requested_profile]

        local_beers = beer_extractor.get_local_beers("QC")
        most_similar_beer = model.create_beer_vector(proportion_tensors)
        local_similar_beer = [(name, score) for name, score in most_similar_beer if name in local_beers]
        local_similar_beer.sort(key=lambda x: x[1], reverse=True)

        output = generate_html_header("Build a beer")
        output += "<a href=/>Home</a>" \
                 "<h1><a href=/similar_beer>Build a beer</a></h1>".format()
        output += "<h2>Requested Profile</h2>"
        output += format_top_taste([(model.idx_to_tag[idx], proportion) for idx, proportion in requested_profile])
        output += "<h2>Suggested drink</h2>"
        output += format_beer_with_base_info([b for b, _ in local_similar_beer[0:10]])

    elif request.method == "GET":
        output = generate_html_header("Build a beer")
        output += "<a href=/>Home</a>" \
                 "<h1><a href=/similar_beer>Build a beer</a></h1>".format()
        output += "<h2>Requested Profile</h2>"
        output += "<form name=request_profile method=POST action=/similar_beer>"
        for taste_number in range(5):
            output += "<label for=proportion_{0}>Flavor {0}</label> " \
                      "<select name=profile_{0}>{1}</select> " \
                      "<input text name=proportion_{0} size=4></br>".format(taste_number, format_dictionary_for_select_dropdown(model.idx_to_tag))
        output += "</br><input type=submit value=Chercher></form>"

    return output


@app.route('/beer/<name>')
def beer(name):
    beer_idx = beer_to_idx[name]
    beer_tensor = torch.LongTensor([beer_idx])

    review = checking_extractor.get_all_checking_for_beer(name)

    beer_similarities = model.get_beer_similarities(beer_tensor)
    flavors = model.get_beer_flavors(beer_tensor, topn=5)

    local_beers = beer_extractor.get_local_beers("QC")
    similar_local_beer = [(name, score) for name, score in beer_similarities if name in local_beers]

    similar_beers = [b for b, _ in similar_local_beer][-10:]
    dissimilar_beers = [b for b, _ in similar_local_beer][:10]
    output = generate_html_header(name)
    output += '<a href=/>Home</a>  ' \
    '<h1>{} - {} ({})</h1>' \
    '<h1>Taste profile</h1>' \
    '{}' \
    '<h1>User Reviews</h1>' \
    '<h2>Happy people</h2>{}' \
    '<h2>Sad people</h2>{}' \
    '<h1>Beer Comparison</h1>' \
    '<h2>Similar:</h2> {}  ' \
    '<h2>Dissimilar:</h2> {}'.format(beer_extractor.get_beer_brewery(name), name, beer_extractor.get_beer_type(name),
                                     format_top_taste(flavors),
                                     format_checkin(review[-10:]), format_checkin(review[:10]),
                                     format_beer_with_base_info(similar_beers),
                                     format_beer_with_base_info(dissimilar_beers))

    return output
if __name__== "__main__":
    app.run(debug=False, host="0.0.0.0", port=5100)
