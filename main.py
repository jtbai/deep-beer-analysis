import json
from repository.mongo_beer_extractor import MongoBeerExtractor

MONGO_CONGIF_FILE_PATH = "config/mongo_connection_details.json"
mongo_extractor = MongoBeerExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))


for checking in mongo_extractor.get_all_checking_with_tags():
    pass