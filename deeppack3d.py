import numpy as np
import os
import shutil
import time

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from deap import gp as deap_gp

from rl.env import MultiBinPackerEnv
from rl.conveyor import FileConveyor, InputConveyor
from rl.agent import (
    bottom_left,
    best_area_fit,
    best_short_side_fit,
    best_long_side_fit,
    Agent,
    HeuristicAgent,
)
from rl.split_gen import reset_rng

from gp import gp_main as gp
from gp.util import saveFile, load_individual_from_gen_json_format


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        metavar="method",
        type=str,
        choices=["rl", "bl", "baf", "bssf", "blsf", "gp"],
        help='choose the method from {"rl", "bl", "baf", "bssf", "blsf", "gp"}.',
    )

    parser.add_argument(
        "--lookahead", metavar="lookahead", type=int, help="choose the lookahead value."
    )

    parser.add_argument(
        "--data",
        metavar="",
        type=str,
        default="generated",
        choices=["generated", "input", "file"],
        help='choose the input source from {"generated", "input", "file"} (default: generated).',
    )

    parser.add_argument(
        "--path",
        metavar="",
        type=str,
        default=None,
        help='set the file path, only used if --data is "file" (default: None).',
    )

    parser.add_argument(
        "--n_iterations",
        metavar="",
        type=int,
        default=100,
        help='set the number of iterations, only used if --data is "generated" (default: 100).',
    )

    parser.add_argument(
        "--seed",
        metavar="",
        type=int,
        default=None,
        help='set the random seed for reproducibility, only used if --data is "generated" (default: None).',
    )

    parser.add_argument(
        "--verbose",
        metavar="",
        type=int,
        default=1,
        help="set verbose level (default: 1).",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help='enable training mode, only used if method is "rl" (default: False).',
    )

    parser.add_argument(
        "--batch_size",
        metavar="",
        type=int,
        default=32,
        help="set batch_size, only used if train is True (default: 32).",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="enable visualization mode (default: False).",
    )

    return parser.parse_args()


heuristics = {
    "bl": bottom_left,
    "baf": best_area_fit,
    "bssf": best_short_side_fit,
    "blsf": best_long_side_fit,
}


def deeppack3d(
    cfg,
    method,
    lookahead,
    *,
    n_iterations=100,
    seed=None,
    verbose=1,
    data="generated",
    path="",
    train=False,
    visualize=False,
    batch_size=32,
):
    reset_rng(seed)

    env = MultiBinPackerEnv(
        n_bins=1,
        max_bins=1,
        size=(32, 32, 32),
        k=lookahead,
        prealloc_items=100,
        verbose=verbose,
    )

    if data == "file":
        env.conveyor = FileConveyor(k=env.k, path=path).reset()
    elif data == "input":
        env.conveyor = InputConveyor(k=env.k).reset()

    if visualize:
        if os.path.exists("./outputs"):
            shutil.rmtree("./outputs")
        os.makedirs("./outputs")

    if train:
        print(f'Training with method "{method}" and lookahead {lookahead}...')

        if method != "rl" and method != "gp":
            raise Exception('training mode can only be used if method is "rl" or "gp".')

        agent = Agent(
            env,
            train=True,
            verbose=verbose > 0,
            visualize=visualize,
            batch_size=batch_size,
        )

        agent.eps = 1.0
        if method == "rl":
            for i in range(n_iterations):
                print(f"Iteration {i}")
                start_time = time.time()
                yield from agent.run(100, verbose=verbose > 1)
                agent.eps = max(agent.eps * 0.95, 0.025)
                data = (
                    np.asarray([utils for utils, n_bins, ep_reward in agent.ep_history])
                    .flatten()
                    .astype(float)
                )
                saveFile.save_rl_utils(cfg, data.tolist())
                y = np.ones(100)
                data = np.convolve(data, y, "valid") / len(y)
                sns.lineplot(data=data)
                plt.savefig("./util.jpg")
                plt.show()

                data = np.asarray(
                    [ep_reward for utils, n_bins, ep_reward in agent.ep_history]
                )
                saveFile.save_rl_rewards(cfg, data.tolist())
                y = np.ones(100)
                data = np.convolve(data, y, "valid") / len(y)
                sns.lineplot(data=data)
                plt.savefig("./ep_reward.jpg")
                plt.show()
        elif method == "gp":
            gp.main(cfg, agent)

        if method == "rl":
            import uuid

            uid = uuid.uuid4()
            print(f"saved model at ./{uid}.keras")
            agent.q_net.save(f"{uid}.keras")
    else:
        if verbose > 0:
            print(f'Testing with method "{method}" and lookahead {lookahead}...')

        if method == "rl":
            model_path = f"./models/k={lookahead}.h5"
            agent = Agent(
                env,
                train=False,
                verbose=verbose > 0,
                visualize=visualize,
                batch_size=batch_size,
            )
            agent.q_net = tf.keras.models.load_model(model_path, compile=False)
            agent.eps = 0.0
        elif method == "gp":
            # dict_best_MTGP_individuals = load_individual_from_gen(cfg)
            dict_best_MTGP_individuals_dict = load_individual_from_gen_json_format(cfg)

            from gp.util import init_primitives
            from deap import base

            pset = deap_gp.PrimitiveSetTyped(
                "MAIN", [np.ndarray, np.ndarray, np.ndarray, float], float, prefix="f"
            )
            pset.renameArguments(
                f0="Constant", f1="Height_Map", f2="Action_Map", f3="Item_Map"
            )
            pset.context["array"] = np.array
            init_primitives(pset)
            toolbox = base.Toolbox()
            toolbox.register("compile", deap_gp.compile, pset=pset)

            agent = Agent(
                env,
                train=False,
                verbose=verbose > 0,
                visualize=visualize,
                batch_size=batch_size,
            )
            for run in range(cfg.evaluation_iterations):
                seed = np.random.randint(2000000)
                print(
                    "******************* ITERATION-{} on SEED-{} *******************".format(
                        run, seed
                    )
                )

                for idx, individual in enumerate(dict_best_MTGP_individuals_dict):
                    rule_string = individual.get("T0")
                    rule = deap_gp.PrimitiveTree.from_string(rule_string, pset=pset)
                    from gp.evaluate import _space_utilization_evaluate_once

                    fitness_value = _space_utilization_evaluate_once(
                        agent,
                        [
                            rule,
                        ],
                        gen=seed,
                        config=cfg,
                        toolbox=toolbox,
                    )
                    individual["fitness"] += fitness_value

            for ind in dict_best_MTGP_individuals_dict:
                ind["fitness"] = ind["fitness"] / cfg.evaluation_iterations
            saveFile.save_each_gen_best_individual_on_test_dataset(
                cfg, dict_best_MTGP_individuals_dict
            )

        else:
            agent = HeuristicAgent(
                heuristics[method], env, verbose=verbose > 0, visualize=visualize
            )

        start_time = time.time()

        try:
            yield from agent.run(n_iterations, verbose=verbose > 1)
        except Exception as e:
            if np.all(np.array(env.conveyor.reset().peek()) == None):
                if verbose > 0:
                    print("\n=====the end of conveyor line=====")
            else:
                print(e)

        if verbose > 0:
            print()
            next_items = np.array(env.conveyor.reset().peek()).tolist()
            avg_util = np.mean(
                [
                    util
                    for utils, n_bins, ep_reward in agent.ep_history[:]
                    for util in utils[:]
                ]
            )
            used_items = np.sum(
                [
                    n_bins
                    for utils, n_bins, ep_reward in agent.ep_history[:]
                    for util in utils[:]
                ]
            )

            print(f"Used time: {int(time.time() - start_time)} seconds")
            print(f"Next items: {next_items}")
            print(f"Average space util: {avg_util}")
            print(f"Used bins: {used_items}")


def merge_conf(cli_conf: DictConfig) -> DictConfig:
    base_conf = OmegaConf.load("conf/defaults/base.yaml")
    algo_conf = OmegaConf.load(f"conf/exp/{cli_conf['--method']}.yaml")
    conf = OmegaConf.merge(base_conf, algo_conf, cli_conf)
    conf.seed = cli_conf.get("--seed", None)
    conf.lookahead = cli_conf.get("--lookahead", None)
    conf.method = cli_conf.get("--method", None)
    conf.train = cli_conf.get("--train", None)
    conf.verbose = cli_conf.get("--verbose", None)
    return conf


def main():
    args = parse_args()
    reset_rng(args.seed)
    cli_conf = OmegaConf.from_cli()  # Default to KNN if not specified
    if args.method == "gp":
        cfg = merge_conf(cli_conf)
    else:
        cfg = cli_conf
    for _ in deeppack3d(
        cfg,
        args.method,
        args.lookahead,
        n_iterations=args.n_iterations,
        seed=args.seed,
        train=args.train,
        verbose=args.verbose,
        data=args.data,
        path=args.path,
        visualize=args.visualize,
        batch_size=args.batch_size,
    ):
        pass


if __name__ == "__main__":
    main()
