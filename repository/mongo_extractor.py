from collections import namedtuple
import re
import tqdm
from pymongo import MongoClient

Checkin = namedtuple('Checkin', ['user', 'beer', 'score', 'tags', 'beer_type'])


class MongoExtractor:

    def __init__(self, collection):
        self.collection = collection
        self.beer_locations = None
        self.beer_breweries = None
        self.beer_types = None

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

    def _get_beers_brewery(self):
        if self.beer_breweries:
            return self.beer_breweries
        else:
            dataset = self.collection.find({}, {"name":True ,"brewery_name":True,"_id":False})
            self.beer_breweries = {document['name']: document['brewery_name'] for document in dataset}

        return self._get_beers_brewery()

    def _get_beers_locations(self):
        if self.beer_locations:
            return self.beer_locations
        else:
            dataset = self.collection.find({}, {"name": True, "brewery_location": True, "_id": False})
            self.beer_locations = {document['name']: document['brewery_location'] for document in dataset}

        return self._get_beers_locations()

    def _get_beers_types(self):
        if self.beer_types:
            return self.beer_types
        else:
            dataset = self.collection.find({}, {"name": True, "type": True, "_id": False})
            self.beer_types = {document['name']: document['type'] for document in dataset}

        return self._get_beers_types()


    def get_beer_brewery(self, beer_name):
        beer_brewery = self._get_beers_brewery()
        return beer_brewery.get(beer_name, "unknown")

    def get_beer_location(self, beer_name):
        beer_location = self._get_beers_locations()
        return beer_location.get(beer_name, "unknown")

    def get_beer_type(self, beer_name):
        beer_types = self._get_beers_types()
        return beer_types.get(beer_name, "unknown")


    def get_local_beers(self, local):
        beer_location = self._get_beers_locations()
        return {name for name, location  in beer_location.items() if len(re.findall(local,location))>0}

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
            beer_type=checkin['beer'][0]['type'] if len(checkin['beer']) > 0 else "None"
        )
