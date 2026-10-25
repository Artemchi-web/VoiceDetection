import asyncio

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
import config
import librosa as lb
import numpy as np
import joblib
import os

# оптимальные границы классификации
GREEN_ZONE = 0.3
YELLOY_ZONE = 0.4

# функции анализа звука
def mfcc_fixed(y: int):
    mfcc = lb.feature.mfcc(
        y = y,
        sr = config.sr,
        n_mfcc = config.n_mfcc,
        n_fft = config.n_fft,
        hop_length = config.hop_length,
        n_mels= config.n_mels
    )
    return mfcc

def centr(y: int):
    centroid = lb.feature.spectral_centroid(
        y=y,
        sr=config.sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length
    )
    return centroid[0]

def zcrs(y: int):
    zcr = lb.feature.zero_crossing_rate(
    y=y,
    frame_length=config.n_fft,
    hop_length=config.hop_length
    )
    return zcr[0]

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
bot = Bot(token=config.TOKEN)
dp = Dispatcher()

@dp.message(Command('start'))
async def hello(mes:Message):
    await mes.answer("Пришлите голосовое сообщение или аудио *дольше 3 секунд*.\nПомните: *чем длительнее аудио, тем больше точность.*", parse_mode="Markdown")


@dp.message(Command('help'))
async def help(mes:Message):
    await mes.answer("Данный бот специализируется на анализе речи в голосовых сообщениях с целью выявления признаков синтеза. Бот предоставляет информацию о прогнозе по поводу запроса, а также высчитывает 'индекс подозрительности' при помощи специализированного алгоритма. *Точность анализа около 80%.*", parse_mode="Markdown")

def voice_analyzer(file_path : str):
    y, _ = lb.load(file_path, sr=config.sr)
    os.remove(file_path)
    K = config.sr * 3
    
    count_of_iterations = len(y) // K
    if count_of_iterations == 0: return [0,0]
    S = 0
    for i in range(count_of_iterations):
        y_part = y[K * i : K * (i + 1)]
        mfcc = mfcc_fixed(y_part)
        delta = lb.feature.delta(mfcc)
        centroid = centr(y_part).flatten()
        zcr = zcrs(y_part).flatten()
        feat = np.concatenate([
            mfcc.flatten(),
            delta.flatten(),
            centroid,
            zcr
        ])

        features = scaler.transform(feat.reshape(1, -1))
        proba = model.predict_proba(features)[0, 1]
        S += proba
    status = ""
    res_proba = S / count_of_iterations
    if res_proba <= GREEN_ZONE:
        status = "НАСТОЯЩАЯ РЕЧЬ🟢" 
    elif res_proba <= YELLOY_ZONE:
        status = "C ПРИЗНАКАМИ СИНТЕЗА🟡"
    else: status = "СИНТЕЗ🔴⚠️"
    return [res_proba, status]

@dp.message(F.voice)
async def audio(mes:Message, bot:Bot):
    await mes.answer("Скачивание и анализ голосового сообщения...")
    voice = mes.voice.file_id
    file_info = await bot.get_file(voice)
    file_path = f"temp/voice_{voice}.ogg"
    await bot.download_file(file_info.file_path, file_path)
    params = voice_analyzer(file_path)
    if params[0] == 0 and params[1] == 0: await mes.answer("Аудио слишком короткое!")
    else:
        await mes.answer(f"""*Результаты анализа:*\nВердикт: *{params[1]}*\nВероятность синтеза: *{round(params[0] * 100)}%*\n\n\n_Точность результата составляет ~80%. Чем дольше запись, тем точнее результат._""", parse_mode="Markdown")
    
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())