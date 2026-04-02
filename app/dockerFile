FROM python:3.10-slim

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 80

RUN mkdir ~/.streamlit

RUN cp config.toml ~/.streamlit/config.toml

RUN cp credentials.toml ~/.streamlit/credentials.toml

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]
