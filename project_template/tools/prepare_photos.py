"""Готовит фото и видео с телефона для проекта.

Как пользоваться (дома, после каждой съёмки):
1. Скопируйте файлы с телефона в папку data/phone/ — она не попадает в git.
2. В папке проекта выполните: python tools/prepare_photos.py
3. В data/raw/ появятся копии. Просмотрите их: лица и номера скрипт не скрывает.
   Только разрешённые и проверенные файлы коммитьте:
       git add data/raw
       git commit -m "Новые кадры"
       git push

Что делает скрипт:
- фото поворачивает правильно (по пометке в EXIF), уменьшает до 1280 пикселей
  по длинной стороне и сохраняет в JPEG без метаданных — в том числе без GPS;
- видео уменьшает до 480 пикселей по короткой стороне, обрезает до минуты
  и сохраняет без звука;
- уже подготовленные файлы пропускает, так что запускать можно сколько угодно раз.

Фото с iPhone в формате HEIC скрипт прочитает, если установить пакет pillow-heif:
    pip install pillow-heif
"""

import math
import platform
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / 'data' / 'phone'
TARGET_DIR = ROOT / 'data' / 'raw'

PHOTO_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.3gp'}
MAX_PHOTO_SIDE = 1280
VIDEO_SHORT_SIDE = 480
MAX_VIDEO_SECONDS = 60


def prepare_photo(source, target):
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE))
        # Новое изображение из одних пикселей: метаданные с собой не переедут
        clean = Image.new('RGB', image.size)
        clean.paste(image)
        clean.save(target, 'JPEG', quality=90)


def _even(value):
    return max(2, int(round(value / 2)) * 2)


def prepare_video(source, target):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError('OpenCV не смог открыть видео')
    writer = None
    frames = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        max_frames = int(fps * MAX_VIDEO_SECONDS)
        while frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale = min(1.0, VIDEO_SHORT_SIDE / min(height, width))
            size = (_even(width * scale), _even(height * scale))
            if size != (width, height):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                if not writer.isOpened():
                    raise ValueError('не удалось создать видео: проверьте кодек и путь')
            writer.write(frame)
            frames += 1
        if frames == 0:
            raise ValueError('в видео не нашлось ни одного кадра')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError('выходное видео не записано')
    return frames / fps


def main():
    if not SOURCE_DIR.is_dir():
        print(f'Нет папки {SOURCE_DIR}. Создайте её и положите туда файлы с телефона.')
        return 1
    if platform.system() == 'Windows' and not str(ROOT).isascii():
        print('Внимание: в пути к проекту есть не латинские буквы, OpenCV может не открыть видео.')
        print('Надёжнее перенести проект в папку вроде C:\\CV\\cv_project.')
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    photos, videos, skipped, problems = 0, 0, 0, []
    for source in sorted(SOURCE_DIR.iterdir()):
        suffix = source.suffix.lower()
        if source.name.startswith('.') or not source.is_file():
            continue
        if suffix not in PHOTO_SUFFIXES | VIDEO_SUFFIXES:
            problems.append(f'{source.name}: не фото и не видео, пропускаю')
            continue
        # Сохраняем исходное расширение в имени: IMG.jpg и IMG.png не затрут друг друга.
        stem = source.name
        target = TARGET_DIR / (stem + ('.jpg' if suffix in PHOTO_SUFFIXES else '.mp4'))
        if target.exists():
            skipped += 1
            continue
        if suffix in {'.heic', '.heif'} and not HEIC_SUPPORTED:
            problems.append(f'{source.name}: формат HEIC, выполните pip install pillow-heif и запустите снова')
            continue
        try:
            if suffix in PHOTO_SUFFIXES:
                prepare_photo(source, target)
                photos += 1
            else:
                seconds = prepare_video(source, target)
                videos += 1
                print(f'{source.name}: видео {seconds:.0f} с')
        except Exception as error:
            target.unlink(missing_ok=True)
            problems.append(f'{source.name}: {error}')

    total_mb = sum(path.stat().st_size for path in TARGET_DIR.iterdir() if path.is_file()) / 1024 ** 2
    print(f'Готово: фото {photos}, видео {videos}, уже были готовы {skipped}.')
    print(f'В data/raw сейчас {total_mb:.1f} МБ.')
    for problem in problems:
        print('⚠️', problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
