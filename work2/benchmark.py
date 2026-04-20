import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

BUILD_DIR = "./build"
REPORT_DIR = "./report"
R_PARAM = 1
THREADS = [1, 2, 4, 6, 8, 16, 20, 40]

REGEX_SINGLE_FLOAT = r"^[\d.e+-]+$"
REGEX_TWO_FLOATS = r"^([\d.e+-]+)\s+([\d.e+-]+)$"

TASK_CONFIGS = {
    1: {
        "task_id": 1,
        "N": [20000, 40000],
        "iterations": 15,
        "binaries": ["benchmark_static"],
        "schedules": ["static"],
    },
    2: {
        "task_id": 2,
        "N": [40000000],
        "iterations": 100,
        "binaries": ["benchmark_static"],
        "schedules": ["static"],
    },
    3: {
        "task_id": 3,
        "N": [20000],
        "iterations": 5,
        "binaries": ["benchmark_static", "benchmark_dynamic", "benchmark_guided"],
        "schedules": ["static", "dynamic", "guided"],
    },
}


def wait_until(target_hour: int, target_minute: int) -> None:
    """
    Блокирует выполнение потока до наступления указанного времени.
    """
    now = datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    if target <= now:
        target += timedelta(days=1)
        
    wait_seconds = (target - now).total_seconds()
    print(f"[Timer] Ожидание старта до {target.strftime('%Y-%m-%d %H:%M:%S')} (Осталось: {wait_seconds:.0f} сек.)")
    time.sleep(wait_seconds)


def parse_output(stdout: str, task_id: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Разбирает вывод консоли в зависимости от типа задачи.

    Parameters
    ----------
    stdout : str
        Текст стандартного вывода программы.
    task_id : int
        Идентификатор задачи для выбора логики парсинга.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        Время выполнения (time_1, time_2). Значения None, если разбор не удался.
    """

    stdout = stdout.strip()

    if task_id in (1, 2):
        match = re.match(REGEX_SINGLE_FLOAT, stdout)
        if match:
            return float(stdout), None
    elif task_id == 3:
        match = re.match(REGEX_TWO_FLOATS, stdout)
        if match:
            return float(match.group(1)), float(match.group(2))

    print(
        f"\n[Error] Invalid output format for Task {task_id}: '{stdout}'",
        file=sys.stderr,
    )
    return None, None


def run_benchmark(
    binary: str,
    task_id: int,
    n_val: int,
    threads: int,
    build_dir: str = BUILD_DIR,
    r_param: int = R_PARAM,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Запускает бинарный файл с нужными параметрами и возвращает результаты.

    Parameters
    ----------
    binary : str
        Имя исполняемого файла.
    task_id : int
        Идентификатор задачи.
    n_val : int
        Параметр N для бенчмарка.
    threads : int
        Количество потоков.
    build_dir : str, optional
        Путь к директории сборки (по умолчанию BUILD_DIR).
    r_param : int, optional
        Параметр R для бенчмарка (по умолчанию R_PARAM).

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        Распарсенное время выполнения или (None, None) в случае ошибки.
    """

    binary_path = os.path.join(build_dir, binary)

    if not os.path.isfile(binary_path):
        print(f"\n[Error] Binary not found: {binary_path}", file=sys.stderr)
        return None, None

    # Привязка потоков к ядрам для снижения дисперсии измерений
    omp_env = os.environ.copy()
    omp_env["OMP_PROC_BIND"] = "true"
    omp_env["OMP_PLACES"] = "cores"

    try:
        cmd = [binary_path, str(task_id), str(n_val), str(threads), str(r_param)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=omp_env)
        return parse_output(result.stdout, task_id)

    except subprocess.CalledProcessError as err:
        print(
            f"\n[Error] Subprocess failed for {binary} (N={n_val}, threads={threads}): {err}",
            file=sys.stderr,
        )
        return None, None
    except Exception as err:
        print(f"\n[Error] Unexpected failure: {err}", file=sys.stderr)
        return None, None


def collect_task_data(
    task_num: int,
    config: Dict[str, Any],
    threads_list: List[int] = THREADS,
) -> List[Dict[str, Any]]:
    """
    Выполняет все итерации бенчмарка для задачи и возвращает списком результаты.

    Parameters
    ----------
    task_num : int
        Номер задачи из конфигурации.
    config : Dict[str, Any]
        Словарь с настройками задачи.
    threads_list : List[int], optional
        Список потоков для тестирования (по умолчанию THREADS).

    Returns
    -------
    List[Dict[str, Any]]
        Список словарей с результатами каждого запуска.
    """

    task_id = config["task_id"]
    n_values = config["N"]
    iterations = config["iterations"]
    binaries = config["binaries"]
    schedules = config["schedules"]

    results = []
    print(f"\n[Task {task_num}] Iterations: {iterations}")

    for n_val in n_values:
        for binary, schedule in zip(binaries, schedules):
            # Аккуратный вывод в одну строку с выравниванием
            print(f"  N={n_val:<10} | {schedule:<10} | Threads: ", end="", flush=True)

            for threads in threads_list:
                print(
                    f"{threads} ", end="", flush=True
                )  # Показываем текущий шаг в реальном времени

                for iter_num in range(1, iterations + 1):
                    time_1, time_2 = run_benchmark(binary, task_id, n_val, threads)

                    if task_id == 3 and time_1 is not None and time_2 is not None:
                        results.append(
                            {
                                "Task": task_num,
                                "N": n_val,
                                "Threads": threads,
                                "Schedule": f"{schedule}_v1",
                                "Iteration": iter_num,
                                "Time_1": time_1,
                                "Time_2": None,
                            }
                        )
                        results.append(
                            {
                                "Task": task_num,
                                "N": n_val,
                                "Threads": threads,
                                "Schedule": f"{schedule}_v2",
                                "Iteration": iter_num,
                                "Time_1": time_2,
                                "Time_2": None,
                            }
                        )
                    else:
                        results.append(
                            {
                                "Task": task_num,
                                "N": n_val,
                                "Threads": threads,
                                "Schedule": schedule,
                                "Iteration": iter_num,
                                "Time_1": time_1,
                                "Time_2": time_2,
                            }
                        )

            print("OK")  # Маркер успешного окончания строки конфигурации

    return results


def plot_comparison_metrics(
    df: pd.DataFrame, task_id: int, n_val: int, report_dir: str = REPORT_DIR
) -> None:
    """
    Строит и сохраняет графики сравнения двух версий алгоритма (для Task 3).

    Parameters
    ----------
    df : pd.DataFrame
        Отфильтрованный датафрейм с результатами для построения.
    task_id : int
        Идентификатор задачи.
    n_val : int
        Значение параметра N.
    report_dir : str, optional
        Директория для сохранения графиков (по умолчанию REPORT_DIR).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Benchmark Task {task_id} (N={n_val}) [Version Comparison]", fontsize=16
    )

    threads_list = sorted(df["Threads"].unique())
    v1_times, v2_times = [], []

    for threads in threads_list:
        thread_data = df[df["Threads"] == threads]
        v1_data = thread_data[thread_data["Schedule"].str.contains("_v1")]
        v2_data = thread_data[thread_data["Schedule"].str.contains("_v2")]

        v1_times.append(v1_data["Time"].mean() if not v1_data.empty else None)
        v2_times.append(v2_data["Time"].mean() if not v2_data.empty else None)

    v1_valid = [t for t in v1_times if t is not None]
    v2_valid = [t for t in v2_times if t is not None]

    if v1_valid and v2_valid:
        t1_v1, t1_v2 = v1_valid[0], v2_valid[0]

        speedup_v1 = [t1_v1 / t if t else None for t in v1_times]
        speedup_v2 = [t1_v2 / t if t else None for t in v2_times]

        s2_p_v1 = [
            (s**2 / p) if s and p else None for s, p in zip(speedup_v1, threads_list)
        ]
        s2_p_v2 = [
            (s**2 / p) if s and p else None for s, p in zip(speedup_v2, threads_list)
        ]

        ax1.plot(threads_list, speedup_v1, marker="o", label="Version 1 (avg)")
        ax1.plot(threads_list, speedup_v2, marker="s", label="Version 2 (avg)")
        ax1.plot(
            threads_list, threads_list, linestyle="--", color="gray", label="Ideal"
        )

        ax2.plot(threads_list, s2_p_v1, marker="o", label="Version 1 (avg)")
        ax2.plot(threads_list, s2_p_v2, marker="s", label="Version 2 (avg)")

    ax1.set(
        title="Ускорение (Speedup S = T1/Tp)",
        xlabel="Количество потоков (p)",
        ylabel="S",
    )
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.set(title="Метрика S^2 / p", xlabel="Количество потоков (p)", ylabel="S^2 / p")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    filepath = os.path.join(report_dir, f"plot_task{task_id}_N{n_val}_all.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath)
    print(f"  Plot saved: {filepath}")
    plt.close()


def plot_standard_metrics(
    df: pd.DataFrame,
    task_id: int,
    n_val: int,
    suffix: str = "",
    report_dir: str = REPORT_DIR,
) -> None:
    """
    Строит и сохраняет стандартные графики ускорения и эффективности.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с данными для построения графика.
    task_id : int
        Идентификатор задачи.
    n_val : int
        Значение параметра N.
    suffix : str, optional
        Суффикс для имени файла и заголовка графика (по умолчанию пустая строка).
    report_dir : str, optional
        Директория для сохранения графиков (по умолчанию REPORT_DIR).
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    version_label = "v1"
    if suffix == "_v1":
        version_label = "Version 1"
    elif suffix == "_v2":
        version_label = "Version 2"
    elif suffix == "_all":
        version_label = "All Versions"

    fig.suptitle(f"Benchmark Task {task_id} (N={n_val}) [{version_label}]", fontsize=16)
    schedules = df["Schedule"].unique()

    for sch in schedules:
        data = df[df["Schedule"] == sch].sort_values("Threads")
        t1_values = data[data["Threads"] == 1]["Time"].values

        if len(t1_values) == 0:
            continue

        t1 = t1_values[0]
        threads = data["Threads"].values
        times = data["Time"].values

        speedup = t1 / times
        s2_p = (speedup**2) / threads

        ax1.plot(threads, speedup, marker="o", label=f"{sch}")
        if sch == schedules[0]:
            ax1.plot(threads, threads, linestyle="--", color="gray", label="Ideal")

        ax2.plot(threads, s2_p, marker="s", label=f"{sch}")

    ax1.set(
        title="Ускорение (Speedup S = T1/Tp)",
        xlabel="Количество потоков (p)",
        ylabel="S",
    )
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.set(title="Метрика S^2 / p", xlabel="Количество потоков (p)", ylabel="S^2 / p")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    filepath = os.path.join(report_dir, f"plot_task{task_id}_N{n_val}{suffix}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath)
    print(f"  Plot saved: {filepath}")
    plt.close()


def process_and_plot_results(df: pd.DataFrame) -> None:
    """
    Подготавливает данные и распределяет их по функциям отрисовки графиков.

    Parameters
    ----------
    df : pd.DataFrame
        Общий датафрейм со всеми собранными результатами бенчмарка.
    """

    if df.empty or "Time_1" not in df.columns:
        print(
            "\n[Error] Invalid or empty data available for plotting.", file=sys.stderr
        )
        return

    df["Time"] = df["Time_1"]
    grouped = (
        df.groupby(["Task", "N", "Schedule", "Threads"])["Time"].mean().reset_index()
    )

    for task_id in grouped["Task"].unique():
        task_df = grouped[grouped["Task"] == task_id]

        for n_val in task_df["N"].unique():
            subset = task_df[task_df["N"] == n_val]

            if task_id == 3:
                schedules = subset["Schedule"].unique()
                v1_schedules = [s for s in schedules if "_v1" in s]
                v2_schedules = [s for s in schedules if "_v2" in s]

                if v1_schedules:
                    v1_subset = subset[subset["Schedule"].isin(v1_schedules)].copy()
                    v1_subset["Schedule"] = v1_subset["Schedule"].str.replace("_v1", "")
                    plot_standard_metrics(v1_subset, task_id, n_val, suffix="_v1")

                if v2_schedules:
                    v2_subset = subset[subset["Schedule"].isin(v2_schedules)].copy()
                    v2_subset["Schedule"] = v2_subset["Schedule"].str.replace("_v2", "")
                    plot_standard_metrics(v2_subset, task_id, n_val, suffix="_v2")

                plot_comparison_metrics(subset, task_id, n_val)
            else:
                plot_standard_metrics(subset, task_id, n_val)


def main() -> None:
    """
    Главная функция, управляющая логикой проверки, запуска и отрисовки результатов.
    """

    print("=" * 60)
    print("Multithreaded C++ Benchmarking")
    print("=" * 60)

    # Запуск таймера ожидания перед выполнением тестов
    wait_until(3, 0)

    if not os.path.isdir(BUILD_DIR):
        print(
            f"\n[Error] Build directory '{BUILD_DIR}' not found. Please build first.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    try:
        for task_num, config in TASK_CONFIGS.items():
            task_data = collect_task_data(task_num, config)
            all_results.extend(task_data)

    except KeyboardInterrupt:
        print("\n\n[Warning] Benchmark interrupted by user.")

    successful = sum(1 for r in all_results if r.get("Time_1") is not None)

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Successful runs : {successful}")
    print(f"Failed runs     : {len(all_results) - successful}")

    if successful > 0:
        print("\nGenerating Plots...")
        df = pd.DataFrame(all_results)
        process_and_plot_results(df)
        print("Done.")


if __name__ == "__main__":
    main()
