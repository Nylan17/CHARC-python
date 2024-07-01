import numpy as np
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import random
from scipy.spatial import KDTree
import pickle

from Support_files.reservoir_type.Multi_reservoir.RoR_architecture.createRoR import create_ror
from Support_files.reservoir_type.Multi_reservoir.RoR_architecture.mutateRoR import mutate_ror
from Support_files.reservoir_type.Multi_reservoir.RoR_architecture.recombRoR import recomb_ror
from Support_files.other.measureSearchSpace import measure_search_space
from Support_files.other.Metrics.getMetrics import evaluate_individual


# Set random seed for experiments
np.random.seed(1)

# Setup
config = {
    "parallel": True,
    "res_type": "RoR",
    "num_nodes": [100],
    "metrics": ['KR', 'GR', 'linearMC'],
    "voxel_size": 10,
    "train_input_sequence": [],
    "train_output_sequence": [],
    "dataset": "blank",
    "num_tests": 1,
    "pop_size": 100,
    "total_gens": 2000,
    "mut_rate": 0.02,
    "deme_percent": 0.1,
    "deme": round(100 * 0.1),  # pre-calculated since the formula is constant
    "rec_rate": 0.5,
    "k_neighbours": 10,
    "p_min_start": np.sqrt(sum([100])),
    "p_min_check": 100,
    "gen_print": 10,
    "start_time": datetime.datetime.now().strftime("%H:%M:%S"),
    "save_gen": float('inf'),
    "param_indx": 1,
    "get_prediction_data": False,
    "task_list": ['laser', 'narma_10', 'signal_classification', 'non_chan_eq_rodan', 'narma_30', 'henon_map', 'secondorder_task'],
    "figure_array": [plt.figure(), plt.figure()]
}

def find_knn(behaviours, y, k_neighbours):
    tree = KDTree(behaviours)
    dist, _ = tree.query(y, k=k_neighbours)
    return np.mean(dist)

def plot_search(database, gen, config):
    all_behaviours = np.array([d['behaviours'] for d in database])
    param = range(len(all_behaviours))

    plt.figure(config['figure_array'][0].number)
    plt.title(f'Gen: {gen}')
    plt.scatter(all_behaviours[:, 0], all_behaviours[:, 1], c=param, cmap='viridis')
    plt.xlabel(config['metrics'][0])
    plt.ylabel(config['metrics'][1])
    plt.colorbar()
    plt.show()

def plot_quality(quality, config):
    plt.figure(config['figure_array'][1].number)
    plt.plot(range(len(quality)), quality)
    plt.xlabel('Generation')
    plt.ylabel('Quality')
    plt.show()

def save_data(database_history, database, quality, tests, config):
    with open(f'Framework_substrate_{config["res_type"]}_run{tests}_gens{config["total_gens"]}.pickle', 'wb') as f:
        pickle.dump({'database_history': database_history, 'database': database, 'quality': quality, 'config': config}, f)

def create_population(config):
    if config['res_type'] == 'RoR':
        return create_ror(config)

def evaluate_quality(database, config):
    # Assume database is a collection of behaviors from the current population
    behaviors = np.array([ind['behaviors'] for ind in database])
    # Call measure_search_space to calculate the diversity or quality of these behaviors
    quality, stats = measure_search_space(behaviors, resolution=config.get('voxel_size', 10))
    
    # Optionally print or log the quality and stats
    print(f"Quality at generation {config['current_gen']}: {quality}")
    print(f"Stats: {stats}")

    # Store or return the quality and stats for further analysis or decision making
    return quality, stats

def evolve_generation(population, archive, config):
    new_population = []

    for _ in range(len(population)):
        # Tournament selection
        parent1, parent2 = np.random.choice(population, 2, replace=False)
        if np.random.rand() < config['rec_rate']:
            # Recombination
            offspring = recomb_ror(parent1, parent2, config)
        else:
            # If no recombination, randomly choose one of the parents to clone
            offspring = parent1.copy() if np.random.rand() > 0.5 else parent2.copy()

        # Mutation
        mutate_ror(offspring, config)

        # Evaluate the new offspring
        offspring['behaviors'] = config['assessFcn'](offspring, config)
        new_population.append(offspring)

    # Update the population with the new generation
    population[:] = new_population

    # Optional: Update the archive based on novelty or fitness criteria
    # update_archive(archive, population, config)
    
# Additional main experiment setup and processing would be required here
def main(config):
    np.random.seed(1)  # Set the random seed for reproducibility

    # Parallel processing setup
    if config["parallel"]:
        pool = Pool(processes=4)

    for tests in range(config["num_tests"]):
        np.random.seed(tests)  # Update random seed for each test
        print(f"\nTest: {tests + 1}")
        print(f"Processing genotype......... {datetime.datetime.now().strftime('%H:%M:%S')}")
        start_time = datetime.datetime.now()

        # Create and evaluate initial population
        population = create_population(config)
        if config["parallel"]:
            results = pool.map(lambda indv: evaluate_individual(indv, config), population)
            for i, res in enumerate(results):
                population[i]["behaviours"] = res
        else:
            for indv in population:
                indv["behaviours"] = evaluate_individual(indv, config)

        # Initialize the archive with initial population behaviours
        archive = np.array([indv["behaviours"] for indv in population])
        database = population.copy()

        # Display initial state
        plot_search(database, 1, config)
        print(f"Processing took: {(datetime.datetime.now() - start_time).total_seconds()} sec, Starting GA")

        # Evolutionary process
        for gen in range(1, config["total_gens"]):
            np.random.seed(gen)
            evolve_generation(population, archive, config, gen)
            
            # Assuming database needs to be updated here with the latest behaviors:
            database = [config['assessFcn'](ind, config) for ind in population]

            if gen % config["gen_print"] == 0:
                plot_search(database, gen, config)
                evaluate_quality(database, config)

    # Clean up parallel pool
    if config["parallel"]:
        pool.close()
        pool.join()

    print(f"Experiment completed at {datetime.datetime.now().strftime('%H:%M:%S')}")