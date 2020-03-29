# Youtube music recommender

This is a simple youtube music/video recommender. You can checkout the deployed solution on Heroku:

https://thawing-shore-99052.herokuapp.com/

The problem is solved by:
1. Scrapping video data from Youtube pages;
2. Extracting the video information from each page;
3. Preprocess data from each video into a single dataset;
4. Manually label some of samples, active learning the rest;
5. Extract features from the dataset;
6. Train a Random Forest and a LightGBM model and ensamble them;
7. Build a simple app to serve the model through Heroku.

## Running:

Clone the repository and then go to _/src/data_science_ folder.

- Scrap data from Youtube pages using:

```
python search_data_collection.py
```

- Extract information from the pages by:

```
python search_data_parsing.py
```

- Process the video data using

```
python video_data_processing.py
```

- *Manually* label the data creating a new column named "y".
- Fit the final model using:

```
python final_model.py
```

Now that the model is trained, we can deploy it. Go to _/src/deploy_ folder.

#### Run Locally
Start the database with:

```
python db_starter.py
```

Then run the app with:

```
python app.py
```

#### Make a local Docker image to run

Build the docker image:

```
docker build . -t deploy_ytr
```

Run the Docker image:

```
docker run -e PORT=8000 -p 80:80 deploy_ytr
```
