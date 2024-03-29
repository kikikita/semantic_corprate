"""Talk module"""
import io
import os
import requests
import openai
from aiogram import types
from aiogram.dispatcher import FSMContext, Dispatcher
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from create_bot import bot
from keyboards import kb_target_settings, kb_client, kb_method_settings, \
    kb_remove, kb_cancel, kb_useful_settings, kb_photo_load
from db.queries import upload_settings, get_method, get_photo
from PIL import Image
from bot_models import classify, text_to_video, translate

URI_INFO = f"https://api.telegram.org/bot{os.getenv('TOKEN')}/getFile?file_id="
URI = f"https://api.telegram.org/file/bot{os.getenv('TOKEN')}/"

openai.api_key = os.getenv('GPT_TOKEN')


class FSMAdmin(StatesGroup):
    """Group of states FSMAdmin"""
    target = State()
    method = State()
    choose_photo = State()
    photo = State()

    def __eq__(self, other):
        return self.target == other.target and self.method == other.method \
               and self.choose_photo == other.choose_photo \
               and self.photo == other.photo

    def __str__(self):
        pass


class FSMRate(StatesGroup):
    """Group of states FSMRate"""
    again = State()
    rating = State()

    def __eq__(self, other):
        return self.again == other.again and self.rating == other.rating

    def __str__(self):
        pass


available_target_names = ['Сделать что-то полезное', 'Поболтать']
available_method_names = ['Текст', 'Видео-кружочки']
available_choose_photo_names = ['Загрузить новое', 'Оставить текущее']

MAX_DIMENSION = 1000


async def check_dimensions(message):
    """Checking dimensions method"""
    file_id = message.photo[1].file_id
    resp = requests.get(URI_INFO + file_id, timeout=30)
    img_path = resp.json()['result']['file_path']
    img = requests.get(URI + img_path, timeout=30)
    with Image.open(io.BytesIO(img.content)) as img:
        width, height = img.size
        if abs(height - width) not in {0, 1}:
            await message.reply('Фото должно быть строго квадратным!')
            file_id = 0
        if height > MAX_DIMENSION or width > MAX_DIMENSION:
            await message.reply('Фото слишком большое')
            file_id = 0
        return file_id


async def ai_settings(message: types.Message):
    """Method with AI settings"""
    await FSMAdmin.target.set()
    await message.reply('Выбери цель общения', reply_markup=kb_target_settings)


async def choose_target(message: types.Message, state: FSMContext):
    """Method for choosing target"""
    if message.text not in available_target_names:
        await message.reply(
            'Пожалуйста, выбери вариант из предложенных',
            reply_markup=kb_target_settings
        )
    elif message.text == 'Сделать что-то полезное':
        await message.reply(
            'Вот список того, что ты можешь сделать при помощи бота:',
            reply_markup=kb_useful_settings
        )
        await state.finish()
    else:
        async with state.proxy() as data:
            data['target'] = message.text
        await FSMAdmin.next()
        await message.reply(
            'Выбери в каком виде хочешь получать ответы от бота',
            reply_markup=kb_method_settings
        )


async def choose_method(message: types.Message, state: FSMContext):
    """Method for method choosing"""
    if message.text not in available_method_names:
        await message.reply(
            'Пожалуйста, выбери вариант из предложенных',
            reply_markup=kb_method_settings
        )
    elif message.text == 'Видео-кружочки':
        async with state.proxy() as data:
            data['method'] = message.text
        photo = await get_photo(message)
        if photo is not None:
            await bot.send_photo(
                message.from_user.id, photo=photo,
                caption='У тебя есть загруженное фото.\nЖелаешь его изменить?',
                reply_markup=kb_photo_load
            )
            await FSMAdmin.next()
        else:
            await message.reply(
                'Загрузи фото для твоего бота.\n\nОбрати внимание:' +
                '\n - фото должно быть строго квадратным',
                reply_markup=kb_cancel)
            await FSMAdmin.photo.set()
    else:
        async with state.proxy() as data:
            data['method'] = message.text
        await upload_settings(message, state)
        await message.reply('Поздоровайся с ботом!', reply_markup=kb_remove)
        await state.finish()


async def choose_photo(message: types.Message, state: FSMContext):
    """Method for photo choosing"""
    if message.text not in available_choose_photo_names:
        await message.reply(
            'Пожалуйста, выбери вариант из предложенных',
            reply_markup=kb_photo_load
        )
    else:
        if message.text == 'Оставить текущее':
            await message.reply(
                'Поздоровайся с ботом!', reply_markup=kb_remove
            )
            await upload_settings(message, state)
            await state.finish()
        else:
            await message.reply(
                'Загрузи фото для твоего бота.\n\nОбрати внимание:' +
                '\n - фото должно быть строго квадратным',
                reply_markup=kb_cancel
            )
            await FSMAdmin.next()


async def load_photo(message: types.Message, state: FSMContext):
    """Method for photo loading"""
    if message.content_type == 'photo':
        file_id = await check_dimensions(message)
        if file_id != 0:
            async with state.proxy() as data:
                data['photo'] = file_id
            await upload_settings(message, state)
            await state.finish()
            await message.reply('Фото загружено.\nПоздоровайся с ботом!',
                                reply_markup=kb_remove)

    else:
        await message.reply(
            'Это не фото!\nПожалуйста, загрузи фото, ' +
            'либо нажми кнопку "Отмена"',
            reply_markup=kb_cancel)


async def cancel_handler(message: types.Message, state: FSMContext):
    """Cancelling handler method"""
    current_state = await state.get_state()
    if current_state is not None:
        await state.finish()
    await message.reply('ОК', reply_markup=kb_client)


async def get_text_rate(message: types.Message):
    """Get text rating method"""
    await message.reply('Введите ваш отзыв',
                        reply_markup=kb_cancel)
    await FSMRate.rating.set()


async def return_rate_answer(message: types.Message):
    """Method for rating returns"""
    answer = await classify(await translate(message.text))
    await message.reply(answer, reply_markup=kb_cancel)
    await message.answer('Введите новый отзыв или нажмите кнопку "Отмена"')
    await FSMRate.again.set()


async def friend_talk(message: types.Message):
    """Method for just talking"""
    if await get_method(message) == 'Видео-кружочки':
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=message.text,
            temperature=0.5,
            max_tokens=400,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"]
        )
        file_id = await get_photo(message)
        resp = requests.get(URI_INFO + file_id, timeout=30)
        img_path = resp.json()['result']['file_path']
        img = URI + img_path
        await text_to_video(response['choices'][0]['text'], img)
        with open('src/bot/temp/output.mp4', 'rb') as video:
            await message.answer_video_note(video)
            os.remove("src/bot/temp/output.mp4")
    else:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=message.text,
            temperature=0.5,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"]
        )
        await message.answer(response['choices'][0]['text'])


def register_handlers_settings(dispatcher: Dispatcher):
    """Register settings method"""
    dispatcher.register_message_handler(
        ai_settings, commands='settings', state=None
    )
    dispatcher.register_message_handler(
        ai_settings, Text(equals='Настройки', ignore_case=True), state=None
    )
    dispatcher.register_message_handler(
        cancel_handler, state="*", commands='отмена'
    )
    dispatcher.register_message_handler(
        cancel_handler, Text(equals='отмена', ignore_case=True), state="*"
    )
    dispatcher.register_message_handler(choose_target, state=FSMAdmin.target)
    dispatcher.register_message_handler(choose_method, state=FSMAdmin.method)
    dispatcher.register_message_handler(
        choose_photo, state=FSMAdmin.choose_photo
    )
    dispatcher.register_message_handler(
        load_photo, content_types=['any'], state=FSMAdmin.photo
    )
    dispatcher.register_message_handler(
        get_text_rate, Text(equals='Рейтинг по отзыву', ignore_case=True),
        state=None
    )
    dispatcher.register_message_handler(
        return_rate_answer, state=FSMRate.rating
    )
    dispatcher.register_message_handler(
        return_rate_answer, state=FSMRate.again
    )
    dispatcher.register_message_handler(friend_talk)
