import os
from aiogram import types
from sqlalchemy.orm import selectinload
from sqlalchemy import URL
from db import BaseModel, User, create_async_engine,\
    get_session_maker, proceed_chemas
from sqlalchemy import select, update


async def sql_start():
    global async_engine, session_maker
    postgres_url = URL.create(
            'postgresql+asyncpg',
            username=os.getenv('PG_USER'),
            password=os.getenv('PG_PASSWORD'),
            database=os.getenv('PG_NAME'),
            host=os.getenv('PG_HOST'),
            port=os.getenv('PG_PORT')
        )

    async_engine = create_async_engine(postgres_url)
    session_maker = get_session_maker(async_engine)
    await proceed_chemas(async_engine, BaseModel.metadata)


async def checking_user(message: types.Message):
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(User)
                .filter(User.user_id == message.from_user.id)
                )
            user: User = result.one_or_none()

            if user is not None:
                pass
            else:
                user = User(
                    user_id=message.from_user.id,
                    username=message.from_user.username
                )
                await session.merge(user)


async def get_user(message: types.Message) -> User:
    """
    Получить пользователя по его id
    :param user_id:
    :param session_maker:
    :return:
    """
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(User)
                .options(selectinload(User.photo))
                .filter(User.user_id == message.from_user.id)
                )
            return result.scalars().one()


async def upload_settings(message: types.Message, state):
    async with session_maker() as session:
        async with session.begin():
            async with state.proxy() as data:
                await session.execute(
                    update(User)
                    .where(User.user_id == message.from_user.id)
                    .values(data)
                    )
            await session.commit()


async def get_photo(message: types.Message):
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(User.photo)
                .where(User.user_id == message.from_user.id)
                )
            photo = result.scalars().one()
            return photo


async def get_target(message: types.Message):
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(User.target)
                .where(User.user_id == message.from_user.id)
            )
            target = result.scalars().one()
            return target


async def get_method(message: types.Message):
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(User.method)
                .where(User.user_id == message.from_user.id)
            )
            method = result.scalars().one()
            return method
