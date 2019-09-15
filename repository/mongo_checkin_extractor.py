from collections import namedtuple
import re
import tqdm
from pymongo import MongoClient

Checkin = namedtuple('Checkin', ['user', 'beer', 'score', 'tags', 'beer_type'])

class MongoCheckinExtractor:

    def __init__(self, collection):
        self.collection = collection

    def get_all_checking_with_tags(self):
        return self.collection.find({
            "tags.0": {"$exists": True}
        })

    def get_all_checking_with_tags_and_score(self):
        return self.collection.find({
            "tags.0": {"$exists": True},
            "score": {"$exists": True},
        })

    def get_min_max_score_for_all_users(self):
        return self.collection.aggregate([
                {"$match": {"score": {"$gt": 0}}},
                {"$group": {"_id": "$user_name", "nb_review": {"$sum": 1}, "maximum_score": {"$max": "$score"},
                            "minimum_score": {"$min": "$score"}}}
            ])

    @classmethod
    def get_connection(cls, connection_configuration_details):

        host = connection_configuration_details['host']
        port = connection_configuration_details['port']
        db_name = connection_configuration_details['db_name']
        data_collection_name = connection_configuration_details["data_collection_name"]
        username = connection_configuration_details["username"]
        password = connection_configuration_details["password"]

        client = MongoClient(host=host, port=port)
        client[db_name].authenticate(username, password)
        data_collection = client[db_name][data_collection_name]
        return cls(data_collection)


def format_checkins(checkins):
    """
    Format all the checkins to obtain the following examples;
    (user_name, beer_name, score, tags)
    """
    for checkin in checkins:
        yield Checkin(
            user=checkin['user_name'],
            beer=checkin['beer_name'],
            score=checkin['score'],
            tags=checkin['tags'],
            beer_type=checkin['beer_type']
        )
