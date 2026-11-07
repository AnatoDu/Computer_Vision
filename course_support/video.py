"""Небольшая служебная функция блока 07, только NumPy.

Вход: двумерная bool-маска. Выход: список (x0,y0,x1,y1,area) для
4-связных компонент не меньше min_area пикселей. Границы полуоткрытые.
Это обход маски, а не детектор посылок или трекер. Для CPU передавайте
уменьшенную маску; пересчёт координат показан в notebook.
"""
import numpy as np


def component_boxes(mask, min_area=1):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min_area < 1:
        raise ValueError('Нужна двумерная маска и min_area >= 1')
    height, width = mask.shape
    visited = np.zeros_like(mask)
    boxes = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        xmin = xmax = int(x)
        ymin = ymax = int(y)
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            xmin, xmax = min(xmin, cx), max(xmax, cx)
            ymin, ymax = min(ymin, cy), max(ymax, cy)
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if area >= min_area:
            boxes.append((xmin, ymin, xmax+1, ymax+1, area))
    return boxes
