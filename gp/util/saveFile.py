import inspect
import json
from string import Template
import functools
from pathlib import Path


def ensure_directory_exists(file_path_arg_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            config = bound_args.arguments["config"]
            config["seed"] = config["--seed"]
            file_path = file_path_arg_name.substitute(**config)

            file_path = Path(file_path)

            file_path.parent.mkdir(parents=True, exist_ok=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# root_dir = r"./data/${algo}_${path_surfix}/scenario_${scenarios}"
root_dir = r"./data/${algo}"

formula_base_dir = Template(f"{root_dir}/${{lookahead}}/${{seed}}_formula_format.json")
base_dir = Template(f"{root_dir}/${{lookahead}}/${{seed}}_individual.json")
txt_base_dir = Template(f"{root_dir}/${{lookahead}}/${{seed}}_each_gen.txt")

surrogate_accuracy_dir = Template(f"{root_dir}/${{seed}}_accuracy_trend.json")
save_rl_utils_dir = Template(f"./data/rl/rl_${{seed}}_utils.json")
save_rl_rewards_dir = Template(f"./data/rl/rl_${{seed}}_rewards.json")
surrogate_proportion_index_dir = Template(f"{root_dir}/${{seed}}_proportion_index.txt")

save_heuristic_utils_dir = Template(
    f"./data/${{method}}/${{lookahead}}/${{seed}}_formula_format.json"
)


@ensure_directory_exists(save_heuristic_utils_dir)
def save_heuristic_utils(config, utils):
    path = save_heuristic_utils_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(utils, fileName_individual)
    return


@ensure_directory_exists(save_rl_utils_dir)
def save_rl_utils(config, utils):
    path = save_rl_utils_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(utils, fileName_individual)
    return


@ensure_directory_exists(save_rl_rewards_dir)
def save_rl_rewards(config, rewards):
    path = save_rl_rewards_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(rewards, fileName_individual)
    return


@ensure_directory_exists(surrogate_accuracy_dir)
def save_surrogate_accuracy_trend(config, accuracy_trend):
    path = surrogate_accuracy_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(accuracy_trend, fileName_individual)
    return


@ensure_directory_exists(formula_base_dir)
def save_each_gen_best_individual_json_format(config, best_ind_all_gen):
    individual_dict = []

    for key, ind in enumerate(best_ind_all_gen):
        individual_dict.append(
            {
                "T0": str(ind[0]),
                # "T1": str(ind[1]),
                "fitness": ind.fitness.values[0] if hasattr(ind, "fitness") else 0,
            }
        )

    path = formula_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)

    return


@ensure_directory_exists(formula_base_dir)
def save_each_gen_best_individual_on_test_dataset(config, best_ind_all_gen_dict):
    path = formula_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(best_ind_all_gen_dict, fileName_individual)


@ensure_directory_exists(base_dir)
def save_each_gen_best_individual_meng(config, best_ind_all_gen):
    individual_dict = []

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.append(individual)

    path = base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)


@ensure_directory_exists(txt_base_dir)
def clear_individual_each_gen_to_txt(config):
    path = txt_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as file:
        file.write("Best individuals from each gen:\n")
    return


@ensure_directory_exists(txt_base_dir)
def save_individual_each_gen_to_txt(config, individuals, gen):
    path = txt_base_dir.substitute(**config)
    with open(
        path,
        "a",
    ) as file:
        file.write("\nGen: " + str(gen) + "\n")
        file.write("Individual:\n")
        file.write("Tree 0:\n")  # routing rule
        file.write(str(individuals[0]) + "\n")
        # file.write("Tree 1:\n")  # sequencing rule
        # file.write(str(individuals[1]) + "\n")
    return


@ensure_directory_exists(surrogate_proportion_index_dir)
def clear_index_of_selected_inds_in_intermediate(config):
    path = surrogate_proportion_index_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as file:
        file.write("")
    return


@ensure_directory_exists(surrogate_proportion_index_dir)
def save_index_of_selected_inds_in_intermediate(config, index):
    path = surrogate_proportion_index_dir.substitute(**config)
    with open(
        path,
        "a",
    ) as file:
        file.write(str(index) + "\n")
    return
