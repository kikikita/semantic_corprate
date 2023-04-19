from sqlalchemy import Column, Integer, VARCHAR

from .base import BaseModel


class User(BaseModel):
    __tablename__ = 'users'

    # Telegram user id
    user_id = Column(Integer, unique=True, nullable=False, primary_key=True)

    # Telegram user name
    username = Column(VARCHAR(32), unique=False, nullable=True)

    # Target of talking with bot
    target = Column(VARCHAR(32), unique=False, nullable=True)

    # Method of talking with bot
    method = Column(VARCHAR(32), unique=False, nullable=True)

    # Photo of user
    photo = Column(VARCHAR(100), unique=False, nullable=True)

    def __str__(self) -> str:
        return f'<User:{self.user_id}>'
