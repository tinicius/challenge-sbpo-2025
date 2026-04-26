import multiprocessing
from algorithms.ga.ga_heuristic import GeneticAlgorithm
from algorithms.ga.ga_heuristic_full import GeneticAlgorithmFull
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic

from problems.base import ProblemInput, load_instance

import pandas as pd


# Função worker que roda no processo separado
def _solve_worker(solver, instance, queue):
    try:
        result = solver.solve(instance)
        queue.put(result)
    except Exception as e:
        queue.put({"error": str(e)})


# Função gerenciadora do timeout
def run_with_timeout(solver, instance, timeout=90):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_solve_worker, args=(solver, instance, queue)
    )

    process.start()
    process.join(timeout)  # Aguarda no máximo 'timeout' segundos

    if process.is_alive():
        print(
            f"[!] Timeout: O algoritmo {solver.__class__.__name__} excedeu {timeout} segundos e foi abortado."
        )
        process.terminate()  # Mata o processo à força
        process.join()  # Limpa os recursos do processo
        return None  # Retorna None (ou você pode retornar uma string/dicionário avisando do timeout)

    if not queue.empty():
        return queue.get()

    return None


if __name__ == "__main__":
    instance_path = "datasets/a/instance_0001.txt"
    instance = load_instance(instance_path)

    # ga_full = GeneticAlgorithmFull(
    #     {
    #         "variant": "BaseGA",
    #         "pop_size": 50,
    #         "epoch": 300,
    #         "pc": 0.9,
    #         "pm": 0.05,
    #         "selection": "tournament",
    #         "crossover": "uniform",
    #         "mutation": "flip",
    #         "k_way": 0.2,
    #         "aisle_selector": "multi",
    #         "seed_with_heuristics": True,
    #         "start": "useful_seed_aisle",
    #     }
    # )

    ga_aisle = GeneticAlgorithm(
        {
            "variant": "BaseGA",
            "pop_size": 50,
            "epoch": 300,
            "pc": 0.9,
            "pm": 0.05,
            "selection": "tournament",
            "crossover": "uniform",
            "mutation": "flip",
            "k_way": 0.2,
            "aisle_selector": "multi",
            "seed_with_heuristics": True,
            "start": "random",
        }
    )

    aisle_first = AisleFirstHeuristic(
        {"score": "useful", "order": "desc", "prune": "multi"}
    )

    TIMEOUT_SECONDS = 90

    # print("Executando 1...")
    # result = run_with_timeout(ga_full, instance, TIMEOUT_SECONDS)
    # print("Result GA Full:", result["objective"] if result else "Timeout!")

    print("Executando 2...")
    result2 = run_with_timeout(ga_aisle, instance, TIMEOUT_SECONDS)
    print("Result GA Aisle:", result2["objective"] if result2 else "Timeout!")

    print("Executando 3...")
    result3 = run_with_timeout(aisle_first, instance, TIMEOUT_SECONDS)
    print("Result AisleFirst:", result3["objective"] if result3 else "Timeout!")

    df = pd.read_csv("best_solutions/best_objectives.csv")

    df = df[df["instance"] == instance_path.split("/")[-1]]

    print(df)
