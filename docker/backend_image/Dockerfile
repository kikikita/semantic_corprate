FROM python:3.9

COPY ./requirements.txt requirements.txt

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /src

COPY . /src

ENTRYPOINT [ "python3", "src/bot/bot_telegram.py"]
