from repository.mongo_extractor import MongoExtractor
from unittest import TestCase
import json

MONGO_CONGIF_FILE_PATH = "./config/mongo_connection_details.json"

class TestMongoExtractor(TestCase):

    def setUp(self):
        connection_details = json.load(open(MONGO_CONGIF_FILE_PATH))
        self.mongo_extractor = MongoExtractor.get_connection(connection_details)

    def test_request_for_cheking_works(self):
        checkin_with_tags = self.mongo_extractor.get_all_checking_with_tags()
        for checkin in checkin_with_tags:
            self.assertGreaterEqual(len(checkin['tags']), 1)

    def test_request_for_user_min_max_words(self):
        user_with_min_max = self.mongo_extractor.get_min_max_score_for_all_users()
        distinct_users = []
        for user in user_with_min_max :
            self.assertNotIn(user['_id'], distinct_users)
            distinct_users.append(user['_id'])
            self.assertGreaterEqual(user['maximum_score'], user['minimum_score'])
