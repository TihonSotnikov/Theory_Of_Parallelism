import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt

BUILD_DIR = Path("build")
THREADS = [1, 2, 4, 7, 8, 16, 20, 40]
SCHEDULE_MAP = {
    "static": BUILD_DIR / "benchmark_static",
    "dynamic": BUILD_DIR / "benchmark_dynamic",
    "guided": BUILD_DIR / "benchmark_guided",
}

def build_project() -> None:
    """
    Собирает C++ проект с помощью CMake.
    Создает директорию 'build', конфигурирует и компилирует проект.
    При ошибке сборки завершает выполнение скрипта.
    """

    print("--- Сборка C++ проекта ---")
    try:
        BUILD_DIR.mkdir(exist_ok=True)
        subprocess.run(["cmake", "-S", ".", "-B", str(BUILD_DIR)], check=True)
        subprocess.run(["cmake", "--build", str(BUILD_DIR)], check=True)
        print("Сборка успешно завершена.\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Ошибка при сборке проекта: {e}", file=sys.stderr)
        sys.exit(1)


def run_benchmark(
    task: int,
    n_val: int,
    threads: int,
    runs: int,
    schedule: str = "static",
) -> Tuple[float, ...]:
    """
    Запускает скомпилированный C++ бинарник и возвращает время выполнения.

    Parameters
    ----------
    task : int
        Номер задачи (1, 2 или 3).
    n_val : int
        Размер входных данных N.
    threads : int
        Количество потоков.
    runs : int
        Количество прогонов для усреднения.
    schedule : str, optional
        Тип schedule для задачи 3 ('static', 'dynamic', 'guided'), по умолчанию 'static'.

    Returns
    -------
    Tuple[float, ...]
        Кортеж со средним временем выполнения.
        Для задач 1 и 2 возвращает одно значение.
        Для задачи 3 возвращает два значения (время v1, время v2).
    """

    exec_path = SCHEDULE_MAP[schedule]
    cmd = [str(exec_path), str(task), str(n_val), str(threads), str(runs)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    times = tuple(float(t) for t in result.stdout.strip().split())
    return times


def plot_results(
    threads: List[int],
    speedups: List[List[float]],
    labels: List[str],
    title: str,
    filename: str,
) -> None:
    """
    Строит и сохраняет график ускорения.

    Parameters
    ----------
    threads : List[int]
        Список значений количества потоков (ось X).
    speedups : List[List[float]]
        Список списков со значениями ускорения (ось Y).
    labels : List[str]
        Список меток для каждой линии на графике.
    title : str
        Заголовок графика.
    filename : str
        Имя файла для сохранения изображения.
    """

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    plt.plot(threads, threads, 'k--', label='Идеальное ускорение')
    for s_vals, label in zip(speedups, labels):
        plt.plot(threads, s_vals, 'o-', label=label)

    plt.title(title, fontsize=16)
    plt.xlabel("Количество потоков (p)", fontsize=12)
    plt.ylabel("Ускорение (Sp)", fontsize=12)
    plt.xticks(threads)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"График сохранен в файл: {filename}")
    plt.show()


def get_user_input() -> Tuple[int, int]:
    """
    Получает от пользователя номер задачи и количество прогонов.
    """

    while True:
        try:
            task = int(input("Введите номер задачи (1, 2, 3): "))
            if task not in [1, 2, 3]:
                raise ValueError
            runs = int(input("Введите количество прогонов для усреднения: "))
            if runs <= 0:
                raise ValueError
            return task, runs
        except ValueError:
            print("Некорректный ввод. Пожалуйста, попробуйте снова.", file=sys.stderr)


def orchestrate_task_1(runs: int) -> None:
    """
    Управляет выполнением и визуализацией для Задачи 1.
    """

    print("\n--- Запуск Задачи 1: Умножение матрицы на вектор ---")
    n_values = [20000, 40000]
    all_speedups = []
    labels = []

    for n in n_values:
        print(f"\nРасчет для N = {n}...")
        t1 = run_benchmark(task=1, n_val=n, threads=1, runs=runs)[0]
        speedups = [1.0]
        for p in THREADS[1:]:
            print(f"  Потоков: {p}...")
            tp = run_benchmark(task=1, n_val=n, threads=p, runs=runs)[0]
            speedups.append(t1 / tp if tp > 0 else 0)
        all_speedups.append(speedups)
        labels.append(f"N = {n}")

    plot_results(THREADS, all_speedups, labels, "Задача 1: Матрица-вектор", "task1_speedup.png")


def orchestrate_task_2(runs: int) -> None:
    """
    Управляет выполнением и визуализацией для Задачи 2.
    """

    print("\n--- Запуск Задачи 2: Интегрирование ---")
    n = 40_000_000
    t1 = run_benchmark(task=2, n_val=n, threads=1, runs=runs)[0]
    speedups = [1.0]
    for p in THREADS[1:]:
        print(f"  Потоков: {p}...")
        tp = run_benchmark(task=2, n_val=n, threads=p, runs=runs)[0]
        speedups.append(t1 / tp if tp > 0 else 0)

    plot_results(THREADS, [speedups], [f"N = {n}"], "Задача 2: Интегрирование", "task2_speedup.png")


def orchestrate_task_3(runs: int) -> None:
    """
    Управляет выполнением и визуализацией для Задачи 3.
    """

    print("\n--- Запуск Задачи 3: Итерационный метод ---")
    n = 8000
    all_speedups: List[List[float]] = []
    labels: List[str] = []

    for schedule in SCHEDULE_MAP.keys():
        print(f"\nРасчет для schedule = '{schedule}'...")
        t1_v1, t1_v2 = run_benchmark(task=3, n_val=n, threads=1, runs=runs, schedule=schedule)
        speedups_v1, speedups_v2 = [1.0], [1.0]

        for p in THREADS[1:]:
            print(f"  Потоков: {p}...")
            tp_v1, tp_v2 = run_benchmark(task=3, n_val=n, threads=p, runs=runs, schedule=schedule)
            speedups_v1.append(t1_v1 / tp_v1 if tp_v1 > 0 else 0)
            speedups_v2.append(t1_v2 / tp_v2 if tp_v2 > 0 else 0)

        all_speedups.extend([speedups_v1, speedups_v2])
        labels.extend([f"v1 {schedule}", f"v2 {schedule}"])

    plot_results(THREADS, all_speedups, labels, "Задача 3: Итерационный метод", "task3_speedup.png")


def main() -> None:
    """
    Главная функция: собирает проект и запускает выбранную задачу.
    """
    build_project()
    task, runs = get_user_input()

    if task == 1:
        orchestrate_task_1(runs)
    elif task == 2:
        orchestrate_task_2(runs)
    elif task == 3:
        orchestrate_task_3(runs)


if __name__ == "__main__":
    main()
