"""Вспомогательные функции курса «Компьютерное зрение».

В тетради:
    from course_support.cvtools import imread, show

Здесь нет никакой магии: каждую функцию можно (и полезно) прочитать.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """Читает изображение, как cv2.imread, но с двумя отличиями.

    1. Понимает русские буквы в пути: на Windows cv2.imread с ними не справляется.
    2. Если файла нет, сразу об этом говорит, а не возвращает молча None.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Файл не найден: {path}')
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Файл есть, но прочитать его как изображение не получилось: {path}')
    return image


def imwrite(path, image, params=()):
    """Сохраняет изображение, как cv2.imwrite, но тоже понимает русские буквы в пути."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image, list(params))
    if not ok:
        raise ValueError(f'Не получилось сохранить изображение в формате {path.suffix}')
    encoded.tofile(path)
    return path


def _to_displayable(image):
    """Готовит массив для plt.imshow: переставляет каналы BGR → RGB и выбирает шкалу."""
    image = np.asarray(image)
    if image.dtype == bool:
        return image.astype(np.uint8) * 255, dict(cmap='gray', vmin=0, vmax=255)
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image, dict(cmap='gray', vmin=0, vmax=255)
        return image, dict(cmap='gray')
    if image.ndim == 3 and image.shape[2] == 1:
        return _to_displayable(image[:, :, 0])
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1], {}
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]], {}
    raise ValueError(f'Не понимаю, как показать массив формы {image.shape}')


def show(*images, titles=None, axis=False, size=4, cols=4):
    """Показывает одно или несколько изображений в ряд.

    Цветные изображения ожидаются в порядке BGR, как их читает OpenCV: функция
    сама переставит каналы для Matplotlib. Одноканальные изображения типа uint8
    показываются без растягивания яркости: 0 — чёрный, 255 — белый.

    axis=True включает оси с координатами пикселей.
    """
    if not images:
        raise ValueError('Передайте хотя бы одно изображение')
    if titles is not None and len(titles) != len(images):
        raise ValueError('Число подписей должно совпадать с числом изображений')
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    heights = [np.asarray(image).shape[0] / np.asarray(image).shape[1] for image in images]
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * max(heights)), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(images):
            ax.axis('off')
            continue
        data, style = _to_displayable(images[index])
        ax.imshow(data, **style)
        if titles is not None:
            ax.set_title(titles[index])
        if not axis:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def _gpu_description():
    """Ищет видеокарту: сначала NVIDIA через nvidia-smi, потом PyTorch, если он установлен."""
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                return 'NVIDIA: ' + '; '.join(result.stdout.strip().splitlines())
        except (OSError, subprocess.TimeoutExpired):
            return 'диагностика NVIDIA недоступна — первый блок работает на CPU'
    # PyTorch не импортируем ради первого занятия: он пока не нужен.
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        return 'Apple Silicon: встроенная графика, PyTorch сможет её использовать'
    return 'не найдена — будем считать на процессоре'


def hardware_report():
    """Печатает, на чём мы работаем: версии библиотек, процессор, видеокарта."""
    print(f'Python      {platform.python_version()}')
    print(f'OpenCV      {cv2.__version__}')
    print(f'NumPy       {np.__version__}')
    print(f'Система     {platform.system()} {platform.release()}')
    print(f'Процессор   {platform.processor() or platform.machine()}, ядер: {os.cpu_count()}')
    print(f'Видеокарта  {_gpu_description()}')
