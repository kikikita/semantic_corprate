from aiogram.utils import executor
from create_bot import dp
from db.queries import sql_start
from handlers import client, talk


async def on_startup(_):
    print('Бот вышел в онлайн')
    await sql_start()


client.register_handlers_client(dp)
talk.register_handlers_settings(dp)


executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
