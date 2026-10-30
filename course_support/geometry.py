"""Учебное преобразование готового серого изображения. Только NumPy.

H переводит координаты исходника в результат; shape=(высота, ширина).
Для каждого выходного пикселя берём ближайший исходный через обратную H.
Возвращаем (картинка, маска покрытия). Нет поиска точек, оценивания H или сглаживания.
"""
import numpy as np


def warp_nearest(image, H, shape, fill=0.0):
    image = np.asarray(image, dtype=float)
    H = np.asarray(H, dtype=float)
    if image.ndim != 2 or H.shape != (3, 3):
        raise ValueError('Нужны серое изображение и матрица3×3')
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError('Размер результата должен быть положительным')
    yy, xx = np.indices((height, width))
    output_points = np.vstack([xx.ravel(), yy.ravel(), np.ones(height * width)])
    source_points = np.linalg.solve(H, output_points)
    denominator = source_points[2]
    finite = np.isfinite(source_points).all(axis=0) & (np.abs(denominator) > 1e-12)
    source_x = np.zeros(height * width)
    source_y = np.zeros(height * width)
    source_x[finite] = source_points[0, finite] / denominator[finite]
    source_y[finite] = source_points[1, finite] / denominator[finite]
    # Check floating bounds before converting to integers (no integer overflow).
    valid = (finite & (source_x >= 0) & (source_x <= image.shape[1]-1)
             & (source_y >= 0) & (source_y <= image.shape[0]-1))
    out = np.full(height * width, fill, dtype=float)
    x_index = np.rint(source_x[valid]).astype(int)
    y_index = np.rint(source_y[valid]).astype(int)
    out[valid] = image[y_index, x_index]
    return out.reshape(height, width), valid.reshape(height, width)
