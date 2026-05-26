import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

HOUR = 0
MINUTE =0

BUILD_DIR = "."
REPORT_DIR = "./report"
THREADS = [1, 2, 4, 6, 8, 16, 20, 40]

CONFIG = {
    "N": [20000, 40000, 45000],  # Настраиваемые размеры вектора/матрицы
    "runs_num": 1000,       # Количество итераций (передается в C++)
    "binary": "benchmark",
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


def run_benchmark(
    binary: str,
    n_val: int,
    threads: int,
    runs_num: int,
    build_dir: str = BUILD_DIR,
) -> Optional[float]:
    """
    Запускает бинарный файл с нужными параметрами и возвращает среднее время.
    Ожидаемый CLI: <binary> <N> <threads_num> <runs_num>
    """
    binary_path = os.path.join(build_dir, binary)

    if not os.path.isfile(binary_path):
        print(f"\n[Error] Binary not found: {binary_path}", file=sys.stderr)
        return None

    try:
        cmd = [binary_path, str(n_val), str(threads), str(runs_num)]
        # Переменные OMP удалены, так как бинарник содержит pin_thread_to_cpu
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Бинарник возвращает суммарное время всех прогонов
        total_time = float(result.stdout.strip())
        return total_time / runs_num

    except subprocess.CalledProcessError as err:
        print(f"\n[Error] Subprocess failed (N={n_val}, threads={threads}): {err}", file=sys.stderr)
        return None
    except ValueError:
        print(f"\n[Error] Invalid output format: '{result.stdout}'", file=sys.stderr)
        return None
    except Exception as err:
        print(f"\n[Error] Unexpected failure: {err}", file=sys.stderr)
        return None


def collect_data(
    config: Dict[str, Any],
    threads_list: List[int] = THREADS,
) -> List[Dict[str, Any]]:
    """
    Выполняет бенчмарк для заданных параметров и возвращает результаты.
    Итерации обрабатываются внутри C++, Python делает один запуск на конфигурацию.
    """
    n_values = config["N"]
    runs_num = config["runs_num"]
    binary = config["binary"]

    results = []
    print(f"\n[Benchmark] Runs per config: {runs_num}")

    for n_val in n_values:
        print(f"  N={n_val:<10} | Threads: ", end="", flush=True)

        for threads in threads_list:
            print(f"{threads} ", end="", flush=True)
            
            avg_time = run_benchmark(binary, n_val, threads, runs_num)

            if avg_time is not None:
                results.append(
                    {
                        "N": n_val,
                        "Threads": threads,
                        "Time": avg_time,
                    }
                )

        print("OK")

    return results


def plot_metrics(
    df: pd.DataFrame,
    n_val: int,
    report_dir: str = REPORT_DIR,
) -> None:
    """
    Строит и сохраняет графики ускорения и метрики S^2/p.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Benchmark (N={n_val})", fontsize=16)

    data = df[df["N"] == n_val].sort_values("Threads")
    t1_values = data[data["Threads"] == 1]["Time"].values

    if len(t1_values) == 0:
        return

    t1 = t1_values[0]
    threads = data["Threads"].values
    times = data["Time"].values

    speedup = t1 / times
    s2_p = (speedup**2) / threads

    ax1.plot(threads, speedup, marker="o", label="Speedup")
    ax1.plot(threads, threads, linestyle="--", color="gray", label="Ideal")

    ax2.plot(threads, s2_p, marker="s", label="S^2 / p")

    ax1.set(title="Ускорение (Speedup S = T1/Tp)", xlabel="Количество потоков (p)", ylabel="S")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.set(title="Метрика S^2 / p", xlabel="Количество потоков (p)", ylabel="S^2 / p")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    filepath = os.path.join(report_dir, f"plot_N{n_val}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath)
    print(f"  Plot saved: {filepath}")
    plt.close()


def process_and_plot_results(df: pd.DataFrame) -> None:
    """
    Подготавливает данные и вызывает отрисовку для каждого N.
    """
    if df.empty or "Time" not in df.columns:
        print("\n[Error] Invalid or empty data available for plotting.", file=sys.stderr)
        return

    for n_val in df["N"].unique():
        plot_metrics(df, n_val)


def main() -> None:
    print("=" * 60)
    print("Multithreaded C++ Benchmarking")
    print("=" * 60)

    wait_until(HOUR, MINUTE)

    if not os.path.isdir(BUILD_DIR):
        print(f"\n[Error] Build directory '{BUILD_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    
    try:
        all_results = collect_data(CONFIG)
    except KeyboardInterrupt:
        print("\n\n[Warning] Benchmark interrupted by user.")
        all_results = []

    successful = len(all_results)

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Successful runs : {successful}")

    if successful > 0:
        print("\nGenerating Plots...")
        df = pd.DataFrame(all_results)
        process_and_plot_results(df)
        print("Done.")


if __name__ == "__main__":
    main()
