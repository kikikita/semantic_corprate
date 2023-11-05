"""Keyboard settings module"""
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, \
    ReplyKeyboardRemove

# Убрать клавиатуру
kb_remove = ReplyKeyboardRemove()

# Кнопка отмены
cancel_button = KeyboardButton('Отмена')
kb_cancel = ReplyKeyboardMarkup(resize_keyboard=True)
kb_cancel.add(cancel_button)

# Основное меню настроек
bt1 = KeyboardButton('Сделать что-то полезное')
bt2 = KeyboardButton('Поболтать')
kb_target_settings = ReplyKeyboardMarkup(resize_keyboard=True)
kb_target_settings.add(bt1, bt2).row(cancel_button)

# Меню настроек общения
bm1 = KeyboardButton('Текст')
bm2 = KeyboardButton('Видео-кружочки')
kb_method_settings = ReplyKeyboardMarkup(resize_keyboard=True)
kb_method_settings.add(bm1, bm2).row(cancel_button)

# Загрузка фото
bp1 = KeyboardButton('Загрузить новое')
bp2 = KeyboardButton('Оставить текущее')
kb_photo_load = ReplyKeyboardMarkup(resize_keyboard=True)
kb_photo_load.add(bp1, bp2).row(cancel_button)
