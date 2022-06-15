FROM gcr.io/gnomondigital-sandbox/tvmbase
ENV APP_HOME /app
WORKDIR ${APP_HOME}
COPY requirements.txt /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /app
CMD python app/main.py 