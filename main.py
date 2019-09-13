import json
from repository.mongo_extractor import MongoExtractor

MONGO_CONGIF_FILE_PATH = "config/mongo_connection_details.json"
mongo_extractor = MongoExtractor.get_connection(json.load(open(MONGO_CONGIF_FILE_PATH)))


for checking in mongo_extractor.get_all_checking_with_tags():
    pass