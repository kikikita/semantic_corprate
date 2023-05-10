# Semantic Corporation Rate
![2023-05-10 14-29-33](https://github.com/kikikita/semantic_corprate/assets/110126453/57ce329f-54aa-47ec-a9e2-3f0dd6ee71d9)
Telegram bot based on GPT-4 and distilbert.

- The bot can communicate by text, as well as video circles! To start communicating, just say hello to the bot. To change the type of responses - go to "/settings".

- The bot can also calculate the company's rating based on a text reviews. To start calculating, go to '/settings' and choose variant "Сделать что то полезное" -> "Рейтинг по отзыву".

## RUN:

Clone repository 
- git clone https://gitlab.com/kikikita/semantic-corprate

Build docker-image
- docker build -t semantic_bot .

Change docker-compose.yml vars:
- TOKEN: ... [enter your token]
- GPT_TOKEN: ... [enter your GPT token]

Run docker-compose
- docker-compose up
