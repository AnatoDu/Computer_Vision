"""Автопроверки для рабочих тетрадей курса «Компьютерное зрение».

В тетради:
    from course_support.checks import check, summary
    check('01.1.2', phone_frame, phone_width, phone_height)

Номер задания — NN.M.K: занятие, пара, номер задания в паре.
Проверка печатает ✅ или ❌ с подсказкой, а результат запоминает в словаре RESULTS
(ничего не возвращает, чтобы Jupyter не печатал лишнего). Необязательные задания
могут ответить ℹ️ — это не ошибка. summary('01.1') показывает итог по тетради.
Проверки не требуют интернета. Модуль использует только стандартную библиотеку,
NumPy и OpenCV.
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
DATA_DIR = COURSE_DIR / 'data'

_TASKS = {}
RESULTS = {}


def _task(task_id):
    def register(func):
        _TASKS[task_id] = func
        return func
    return register


def check(task_id, *args, **kwargs):
    """Запускает проверку задания task_id и печатает результат."""
    func = _TASKS.get(task_id)
    if func is None:
        print(f'❓ Проверка для задания {task_id} не найдена. Получите актуальную папку course_support у преподавателя')
        return
    try:
        ok, message = func(*args, **kwargs)
    except Exception as error:  # студенту важнее понятное сообщение, чем трассировка
        ok, message = False, f'проверка не смогла выполниться: {error!r}'
    RESULTS[task_id] = ok
    mark = 'ℹ️ ' if ok is None else ('✅ ' if ok else '❌ ')
    print(mark + f'Задание {task_id}: {message}')


def summary(notebook):
    """Показывает, какие задания тетради notebook ('01.1') выполнены в этом запуске.

    Необязательные задания (ответ ℹ️) засчитываются как выполненные.
    """
    prefix = f'{notebook}.'
    task_ids = sorted((t for t in _TASKS if t.startswith(prefix)), key=lambda t: int(t.split('.')[-1]))

    def mark(task_id):
        if task_id not in RESULTS:
            return '⬜'
        return {True: '✅', False: '❌', None: 'ℹ️'}[RESULTS[task_id]]

    done = sum(1 for t in task_ids if t in RESULTS and RESULTS[t] is not False)
    print(f'Тетрадь {notebook}: выполнено {done} из {len(task_ids)}')
    print(' '.join(mark(t) + t for t in task_ids))
    if done == len(task_ids):
        print('Всё готово — сохраните тетрадь и проверьте копию; объясните один результат.')


# ---------- вспомогательные функции ----------

def _is_filled(value):
    return isinstance(value, str) and value.strip() != ''


def _git_env():
    """Запрещаем git спрашивать логин и пароль, иначе ячейка зависнет."""
    env = dict(os.environ)
    env['GIT_TERMINAL_PROMPT'] = '0'
    env['GIT_ASKPASS'] = ''
    return env


def _git(project_dir, *args):
    result = subprocess.run(
        ['git', '-C', str(project_dir), *args],
        capture_output=True, text=True, encoding='utf-8', errors='replace', env=_git_env(), timeout=15,
    )
    return result.returncode, result.stdout.strip()


def _git_all_output(project_dir, *args):
    """То же, но вместе с stderr: git часть сообщений печатает туда."""
    result = subprocess.run(
        ['git', '-C', str(project_dir), *args],
        capture_output=True, text=True, encoding='utf-8', errors='replace', env=_git_env(), timeout=15,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _check_committed(project_dir):
    """Проверяет файлы на диске; несохранённые изменения в интерфейсе Jupyter не видны."""
    code, status = _git(project_dir, 'status', '--porcelain', '--untracked-files=all')
    if code != 0:
        return False, 'не удалось проверить git status; проверьте Git и путь проекта'
    if status:
        return False, ('есть изменённые или новые файлы вне коммита. Сначала Ctrl+S, '
                       'затем git status и осознанное добавление нужных файлов в коммит')
    return True, ''


def _commit_count(project_dir):
    code, out = _git(project_dir, 'rev-list', '--count', 'HEAD')
    return int(out) if code == 0 and out.isdigit() else 0


def _describe_shape(shape):
    if len(shape) == 2:
        return f'{shape[1]}×{shape[0]} (ширина × высота), один канал'
    return f'{shape[1]}×{shape[0]} (ширина × высота), каналов: {shape[2]}'


# ---------- Занятие 1. Как компьютер видит мир ----------

TRACKS = [
    'Счёт на видео',
    'Полка магазина',
    'Документ в кадре',
    'Дефекты',
    'Городская среда',
    'Свой трек',
]
README_PLACEHOLDER = '<впишите трек>'

# Иллюзия одновременного контраста: data/illusion.png, см. tools/make_data.py
ILLUSION_SIZE = (240, 600)       # высота, ширина
ILLUSION_BAR_ROWS = (105, 135)   # строки полоски: от и до (не включая)
ILLUSION_BAR_COLS = (40, 560)    # столбцы полоски
ILLUSION_BAR_VALUE = 128

# Светофоры: цвет горящей и погасшей лампы в порядке BGR
LIGHT_NAMES = ['красный', 'жёлтый', 'зелёный']
_LAMP_COLORS = {
    'красный': ((40, 40, 255), (25, 25, 75)),
    'жёлтый': ((40, 210, 255), (20, 65, 80)),
    'зелёный': ((120, 255, 60), (40, 80, 25)),
}
_LIGHT_CASES = [
    dict(width=120, height=320, background=(170, 140, 110)),  # днём на фоне неба
    dict(width=90, height=240, background=(35, 30, 25)),       # ночью
    dict(width=150, height=400, background=(130, 130, 130)),   # в пасмурный день
    dict(width=70, height=200, background=(90, 110, 140)),     # на фоне стены
]
_COLOR_ALIASES = {'red': 'красный', 'yellow': 'жёлтый', 'amber': 'жёлтый', 'green': 'зелёный'}


def illusion_image():
    """Полоска одного цвета на фоне градиента от чёрного к белому."""
    height, width = ILLUSION_SIZE
    gradient = np.linspace(0, 255, width).round().astype(np.uint8)
    image = np.tile(gradient, (height, 1))
    (top, bottom), (left, right) = ILLUSION_BAR_ROWS, ILLUSION_BAR_COLS
    image[top:bottom, left:right] = ILLUSION_BAR_VALUE
    return image


def traffic_light_image(lit, width=120, height=320, background=(170, 140, 110), noise=8, seed=0):
    """Рисует светофор, в котором горит лампа lit: 'красный', 'жёлтый' или 'зелёный'."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = background
    margin = max(3, int(width * 0.08))
    cv2.rectangle(image, (margin, margin), (width - 1 - margin, height - 1 - margin), (45, 45, 45), -1)
    inner_height = height - 2 * margin
    radius = int(min(width - 2 * margin, inner_height / 3) * 0.36)
    center_x = width // 2
    centers = {name: (center_x, int(margin + inner_height * (i + 0.5) / 3)) for i, name in enumerate(LIGHT_NAMES)}
    for name, center in centers.items():
        cv2.circle(image, center, radius, _LAMP_COLORS[name][1], -1, cv2.LINE_AA)
    lit_color = _LAMP_COLORS[lit][0]
    glow = np.zeros_like(image)
    cv2.circle(glow, centers[lit], int(radius * 1.5), lit_color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), radius * 0.5)
    image = np.maximum(image, (glow * 0.6).astype(np.uint8))
    cv2.circle(image, centers[lit], radius, lit_color, -1, cv2.LINE_AA)
    image = cv2.GaussianBlur(image, (0, 0), 0.8)
    noisy = image.astype(np.int16) + rng.normal(0, noise, image.shape).round().astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _normalize_color(answer):
    if not isinstance(answer, str):
        return None
    word = answer.strip().lower()
    word = _COLOR_ALIASES.get(word, word)
    return word.replace('ё', 'е')


@_task('01.1.1')
def _check_project_created(project_dir):
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        return False, f'папка {project_dir} не найдена. Скопируйте project_template и проверьте путь'
    if not (project_dir / 'README.md').exists():
        return False, 'в папке проекта нет README.md — скопируйте заготовку project_template целиком'
    required = ['course_support/checks.py', 'course_support/cvtools.py', 'workbooks']
    missing = [name for name in required if not (project_dir / name).exists()]
    if missing:
        return False, 'скопируйте полный project_template; не хватает: ' + ', '.join(missing)
    return True, 'собственная папка с тетрадями и поддержкой готова; Git необязателен весь семестр'



@_task('01.1.2')
def _check_phone_frame(frame, width, height):
    if not isinstance(frame, np.ndarray):
        return False, f'нужен массив NumPy, а получен {type(frame).__name__}'
    if not all(isinstance(value, (int, np.integer)) and value > 0 for value in (width, height)):
        return False, 'ширина и высота должны быть положительными целыми числами'
    if frame.dtype != np.uint8:
        return False, f'тип элементов должен быть uint8 (целые от 0 до 255), а у вас {frame.dtype}. Укажите dtype=np.uint8'
    if width != height and frame.shape == (width, height, 3):
        return False, 'ширина и высота перепутаны: в NumPy сначала строки (высота), потом столбцы (ширина), потом каналы'
    if frame.shape != (height, width, 3):
        return False, f'ожидалась форма {(height, width, 3)}: высота, ширина, три канала. А у вас {frame.shape}'
    megabytes = frame.nbytes / 1024 ** 2
    return True, f'кадр {width}×{height} занимает в памяти {frame.nbytes:,} байт — около {megabytes:.0f} МБ'.replace(',', ' ')


@_task('01.1.3')
def _check_kirsch(image):
    if not isinstance(image, np.ndarray):
        return False, f'нужен массив NumPy, а получен {type(image).__name__}'
    if image.ndim == 3:
        return False, 'изображение цветное (три канала), а в 1957 году снимок был чёрно-белым: переведите его в оттенки серого'
    if image.shape != (176, 176):
        return False, f'нужен размер 176×176, а у вас {_describe_shape(image.shape)}'
    if image.dtype != np.uint8:
        return False, f'тип элементов должен быть uint8, а у вас {image.dtype}'
    if image.std() < 5:
        return False, 'картинка почти однотонная — похоже, вместо портрета получился пустой массив'
    return True, '176 × 176 = 30 976 пикселей, как у первой цифровой фотографии 1957 года'


@_task('01.2.1')
def _check_illusion_bar(bar):
    if not isinstance(bar, np.ndarray):
        return False, f'нужен срез массива, а получен {type(bar).__name__}'
    if bar.size == 0:
        return False, 'срез пустой: проверьте, что начало диапазона меньше конца'
    if bar.ndim not in (2, 3):
        return False, 'нужен срез вида illusion[строки, столбцы] — двумерный кусок картинки'
    low, high = int(bar.min()), int(bar.max())
    if low != high:
        return False, f'в срез попал фон: значения от {low} до {high}. Сузьте границы строк или столбцов'
    if low != ILLUSION_BAR_VALUE:
        return False, f'все пиксели равны {low}, но это фон, а не полоска. Найдите полоску по осям графика'
    bar_width = ILLUSION_BAR_COLS[1] - ILLUSION_BAR_COLS[0]
    if bar.shape[1] < 0.7 * bar_width:
        return False, (f'столбцов в срезе: {bar.shape[1]}, а в полоске — около {bar_width}. '
                       'Возьмите полоску почти целиком — иначе не проверить, что она везде одинаковая')
    pixels = f'{bar.size:,}'.replace(',', ' ')
    return True, f'все {pixels} пикселей полоски равны {low}: полоска одного цвета, а обманул нас мозг'


@_task('01.2.2')
def _check_traffic_light(func):
    if not callable(func):
        return False, 'передайте в проверку саму функцию, без скобок: check("01.2.2", traffic_light)'
    failures = []
    total = 0
    for case_index, case in enumerate(_LIGHT_CASES):
        for color_index, color in enumerate(LIGHT_NAMES):
            total += 1
            image = traffic_light_image(color, seed=100 + 10 * case_index + color_index, **case)
            answer = func(image.copy())
            if answer is None:
                return False, 'функция вернула None — не забудьте return'
            if _normalize_color(answer) != _normalize_color(color):
                failures.append(f'на светофоре {case["width"]}×{case["height"]} горит {color}, а ваш ответ — {answer!r}')
    if failures:
        return False, f'верно {total - len(failures)} из {total}. Например, {failures[0]}'
    return True, f'все {total} светофоров распознаны верно'


@_task('01.2.3')
def _check_project_choice(project):
    if not isinstance(project, dict):
        return False, 'нужен словарь'
    for key in ['трек', 'вариант', 'классы']:
        if key not in project:
            return False, f'не хватает ключа "{key}"'
    if not _is_filled(project['трек']):
        return False, 'укажите направление: можно своё, тема пока предварительная'
    if not _is_filled(project['вариант']) or len(project['вариант'].strip()) < 10:
        return False, 'опишите вариант конкретнее: что именно и где вы будете снимать'
    classes = project['классы']
    if not isinstance(classes, list) or not 2 <= len(classes) <= 6:
        return False, 'классы — это список из 2–6 строк'
    if not all(_is_filled(name) for name in classes):
        return False, 'каждый класс — непустая строка'
    if len({name.strip().lower() for name in classes}) < len(classes):
        return False, 'классы повторяются'
    return True, f'трек «{project["трек"]}», классы: {", ".join(classes)}'


@_task('01.2.4')
def _check_project_readme(project_dir, project):
    project_dir = Path(project_dir)
    readme = project_dir / 'README.md'
    if not readme.exists():
        return False, 'не найден README.md в папке проекта'
    text = readme.read_text(encoding='utf-8')
    if README_PLACEHOLDER in text:
        return False, 'в README.md осталась заготовка "<впишите трек>" — впишите свой трек'
    if isinstance(project, dict) and project.get('трек') and project['трек'] not in text:
        return False, f'в README.md не найден ваш трек «{project["трек"]}»'
    return True, 'предварительная тема записана в сохранённом README; обсудите её посильность с преподавателем'



def _check_bundle(project_dir, bundle_path):
    clean, message = _check_committed(project_dir)
    if not clean:
        return False, message
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        return False, f'файл {bundle_path.name} не создан — запустите ячейку с git bundle create'
    code, out = _git_all_output(project_dir, 'bundle', 'verify', str(bundle_path))
    if code != 0:
        first_line = out.splitlines()[0] if out else 'без сообщения'
        return False, f'файл не проходит проверку git: {first_line}'
    code, head = _git(project_dir, 'rev-parse', 'HEAD')
    code_heads, heads = _git_all_output(project_dir, 'bundle', 'list-heads', str(bundle_path))
    if code != 0 or code_heads != 0 or not head:
        return False, 'не удалось сравнить последний коммит и содержимое бандла'
    if head not in {line.split()[0] for line in heads.splitlines() if line.split()}:
        return False, 'в файле нет последнего коммита — сделайте бандл заново после коммита'
    size = bundle_path.stat().st_size / 1024
    size_text = f'{size / 1024:.1f} МБ' if size > 1024 else f'{size:.0f} КБ'
    return True, f'файл {bundle_path.name} готов, {size_text}. Заберите его с собой'


def _check_remote(project_dir):
    """Необязательное задание: проект связан с GitHub или GitVerse и всё отправлено.

    Проверка работает без интернета: смотрим только то, что git знает локально.
    """
    project_dir = Path(project_dir)
    code, url = _git(project_dir, 'remote', 'get-url', 'origin')
    if code != 0 or not url:
        return None, ('сервер не подключён — это нормально: сначала сохраняем файлы, '
                      'а на GitHub отправляем дома или со своего ноутбука (docs/СДАЧА.md)')
    if not url.startswith(('https://', 'http://', 'git@', 'ssh://')):
        return None, ('проект восстановлен из файла, сервер не подключён. '
                      'На GitHub отправляем дома или со своего ноутбука (docs/СДАЧА.md)')
    code, upstream = _git(project_dir, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}')
    if code != 0 or not upstream:
        return False, f'origin есть ({url}), но ветка не связана с сервером: git push -u origin main'
    code, unpushed = _git(project_dir, 'log', '--oneline', '@{u}..HEAD')
    if code == 0 and unpushed:
        count = len(unpushed.splitlines())
        return False, (f'не отправлено коммитов: {count}. Без доступа к сети это нормально: '
                       'отправьте их дома или со своего ноутбука (git push)')
    return True, f'всё отправлено на {url}. Эту ссылку и кладите в резюме'


# Занятие 2. CPU, предоставленные кадры, без Git.
@_task('02.1.1')
def _distance_sizes(near, far):
    ok = np.isclose(near,250) and np.isclose(far,125)
    return bool(ok), 'Расстояние вдвое — высота вдвое меньше' if ok else 'Проверьте h=f*H/Z, f=500, H=1'

@_task('02.1.2')
def _clean_exif(count, size):
    ok = count==0 and tuple(size)==(640,480)
    return ok, 'EXIF отсутствует, размер сохранён; пиксели проверьте сами' if ok else 'Откройте именно сохранённую копию без EXIF'

@_task('02.2.1')
def _corners_other(found,corners):
    ok = bool(found) and corners is not None and np.asarray(corners).shape in ((54,1,2),(54,2))
    return ok, 'Найдены 54 внутренних угла' if ok else 'Шаблон (9,6), исходный учебный кадр'

@_task('02.2.2')
def _corrected_other(original,corrected,K,dist):
    expected=cv2.undistort(original,K,dist)
    ok=corrected.shape==original.shape and np.allclose(corrected,expected,atol=1) and not np.array_equal(original,corrected)
    return bool(ok), 'Исправление применено; объясните картинку' if ok else 'Примените undistort к новому кадру с найденными K/dist'

@_task('02.2.3')
def _bad_measurement(base,bad):
    ok=np.isfinite(base) and np.isfinite(bad) and 0<=base<bad and 2<bad<4
    return bool(ok), 'Одно неверное измерение увеличило RMS' if ok else 'Сдвиньте x только первого угла на 20 px; модель оставьте прежней'
