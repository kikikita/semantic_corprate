"""Keyboard rating module"""
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton


cancel_button = KeyboardButton('Отмена')

bu1 = KeyboardButton('Рейтинг по отзыву')
kb_useful_settings = ReplyKeyboardMarkup(resize_keyboard=True)
kb_useful_settings.add(bu1).row(cancel_button)
