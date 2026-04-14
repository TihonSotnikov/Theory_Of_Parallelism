import sys
from typing import List

from make_lab import (
    build_project,
    run_benchmark,
    plot_results,
    get_user_input,
    THREADS
)

DEMO_THREADS: List[int] = [t for t in THREADS if t <= 8]
DEMO_RUNS: int = 1

DEMO_N_TASK1: List[int] = [1000, 2000]
DEMO_N_TASK2: int = 1_000_000
DEMO_N_TASK3: int = 500


def demo_task_1() -> None:
    """
    Запускает быстрый демо-тест для Задачи 1.
    """

    print("\n--- Запуск ДЕМО-ТЕСТА Задачи 1 ---")
    all_speedups = []
    labels = []

    for n in DEMO_N_TASK1:
        print(f"Используются уменьшенные параметры: N={n}, Runs={DEMO_RUNS}")
        t1 = run_benchmark(task=1, n_val=n, threads=1, runs=DEMO_RUNS)[0]
        speedups = [1.0]
        for p in DEMO_THREADS[1:]:
            tp = run_benchmark(task=1, n_val=n, threads=p, runs=DEMO_RUNS)[0]
            speedups.append(t1 / tp if tp > 0 else 0)
        all_speedups.append(speedups)
        labels.append(f"N = {n} (демо)")

    plot_results(
        DEMO_THREADS, all_speedups, labels,
        "Задача 1: Матрица-вектор (Демо)", "task1_speedup_demo.png"
    )


def demo_task_2() -> None:
    """
    Запускает быстрый демо-тест для Задачи 2.
    """

    print("\n--- Запуск ДЕМО-ТЕСТА Задачи 2 ---")
    n = DEMO_N_TASK2
    print(f"Используются уменьшенные параметры: N={n}, Runs={DEMO_RUNS}")

    t1 = run_benchmark(task=2, n_val=n, threads=1, runs=DEMO_RUNS)[0]
    speedups = [1.0]
    for p in DEMO_THREADS[1:]:
        tp = run_benchmark(task=2, n_val=n, threads=p, runs=DEMO_RUNS)[0]
        speedups.append(t1 / tp if tp > 0 else 0)

    plot_results(
        DEMO_THREADS, [speedups], [f"N = {n} (демо)"],
        "Задача 2: Интегрирование (Демо)", "task2_speedup_demo.png"
    )


def demo_task_3() -> None:
    """
    Запускает быстрый демо-тест для Задачи 3.
    """

    print("\n--- Запуск ДЕМО-ТЕСТА Задачи 3 ---")
    n = DEMO_N_TASK3
    print(f"Используются уменьшенные параметры: N={n}, Runs={DEMO_RUNS}, Schedule=static")

    t1_v1, t1_v2 = run_benchmark(task=3, n_val=n, threads=1, runs=DEMO_RUNS, schedule="static")
    speedups_v1, speedups_v2 = [1.0], [1.0]

    for p in DEMO_THREADS[1:]:
        tp_v1, tp_v2 = run_benchmark(task=3, n_val=n, threads=p, runs=DEMO_RUNS, schedule="static")
        speedups_v1.append(t1_v1 / tp_v1 if tp_v1 > 0 else 0)
        speedups_v2.append(t1_v2 / tp_v2 if tp_v2 > 0 else 0)

    plot_results(
        DEMO_THREADS, [speedups_v1, speedups_v2], ["v1 static (демо)", "v2 static (демо)"],
        "Задача 3: Итерационный метод (Демо)", "task3_speedup_demo.png"
    )


def main() -> None:
    """
    Главная функция для демо-теста.
    """

    build_project()
    print("\n--- РЕЖИМ ДЕМО-ТЕСТА ---")
    print("Используются уменьшенные параметры для быстрой проверки.")

    try:
        task = int(input("Введите номер задачи для демо-теста (1, 2, 3): "))
        if task not in [1, 2, 3]:
            raise ValueError
    except ValueError:
        print("Некорректный ввод. Выход.", file=sys.stderr)
        sys.exit(1)

    if task == 1:
        demo_task_1()
    elif task == 2:
        demo_task_2()
    elif task == 3:
        demo_task_3()


if __name__ == "__main__":
    main()
