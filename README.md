# Semantic Bot
Telegram bot based on GPT-4 and Bert.

The bot can communicate by text, as well as video circles!
To start communicating, just say hello to the bot.
To change the type of responses - go to "Settings".

The bot can also calculate the company's rating based on a text review.

## RUN:

Clone repository
- git clone https://github.com/kikikita/clever_bot.git

Build docker-image
- docker build -t semantic_bot .

Change docker-compose.yml vars:
- TOKEN: ... [enter your token]
- GPT_TOKEN: ... [enter your GTP token]

Run docker-compose
- docker-compose up
