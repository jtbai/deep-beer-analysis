# Database setup

* Use Mongo DB 4.2 to import beer, checkin and brewery
* Make sure you download the 3 data files (path given in the manuscript)
* In the same folder of the downloaded data file type the following command

```
mongoimport --db beer2vec --collection deepbeer-beer deepbeer-beer
mongoimport --db beer2vec --collection deepbeer-brewery deepbeer-brewery
mongoimport --db beer2vec --collection deepbeer-checkin deepbeer-checkin
```

# Running the web applicatoin
* Install python 3.7 / 3.8 depedencies
`pip install -r requirement.txt`

* Run the flask application
`python -m server`

# Generating Beer Embeddings
* Install python 3.7 / 3.8 depedencies
* Install Cuda (and have a GPU ready)

`python -m scripts.scripts.train_beer_embeddings`
