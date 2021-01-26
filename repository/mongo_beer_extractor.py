from collections import namedtuple
import re
import tqdm
from pymongo import MongoClient

Checkin = namedtuple('Checkin', ['user', 'beer', 'score', 'tags', 'beer_type'])


class MongoBeerExtractor:

    def __init__(self, collection):
        self.collection = collection
        self.beer_locations = None
        self.beer_scores = None
        self.beer_breweries = None
        self.beer_types = None

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

    def _get_beers_scores(self):
        if self.beer_scores:
            return self.beer_scores
        else:
            dataset = self.collection.find({}, {"name": True, "score": True, "_id": False})
            self.beer_scores = {document['name']: document['score'] for document in dataset}

        return self._get_beers_scores()


    def get_beer_score(self, beer_name):
        beer_scores = self._get_beers_scores()
        return beer_scores.get(beer_name, "unknown")

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
        # client[db_name].authenticate(username, password)
        data_collection = client[db_name][data_collection_name]
        return cls(data_collection)

