# docker build . -t deploy_ytr
# docker run -e PORT=8000 deploy_ytr
FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ 
RUN pip install -r requirements.txt
RUN python db_starter.py

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku			
CMD gunicorn --bind 0.0.0.0:$PORT -t 300 wsgi 

#https://devcenter.heroku.com/articles/container-registry-and-runtime

# Creating app... done, ⬢ thawing-shore-99052
# https://thawing-shore-99052.herokuapp.com/ | https://git.heroku.com/thawing-shore-99052.git

