from pymongo import MongoClient


class MongoExtractor:

    def __init__(self, collection):
        self.collection = collection

    def get_all_checking_with_tags(self):
        return self.collection.find({
            "data_type": "checkin",
            "tags.0": {"$exists": True}
        })

    def get_min_max_score_for_all_users(self):
        return self.collection.aggregate([
                {"$match": {"data_type": "checkin", "score": {"$gt": 0}}},
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