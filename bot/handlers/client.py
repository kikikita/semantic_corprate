from aiogram import types, Dispatcher
from create_bot import bot
from keyboards import kb_client
from db.queries import checking_user


async def command_start(message: types.Message):
    try:
        await bot.send_message(
            message.from_id,
            'Привет! Я умный бот Semantic CorpRate!' +
            '\n\nЯ умею общаться текстом, а также видео-кружочками!' +
            '\nЧтобы начать общение - просто поздоровайся со мной.' +
            '\nЧтобы изменить вид моих ответов - зайди в "Настройки".',
            reply_markup=kb_client
            )
        await checking_user(message)
        await message.delete()
    except Exception:
        await message.reply('Упс, произошли технические шоколадки(')


def register_handlers_client(dp: Dispatcher):
    dp.register_message_handler(command_start, commands=['start', 'help'])
