import os
from multiprocessing.pool import ThreadPool
from typing import List

from sebs.faas.system import System as FaaSSystem
from sebs.faas.function import Trigger
from sebs.experiments.experiment import Experiment
from sebs.experiments.result import Result as ExperimentResult
from sebs.experiments.config import Config as ExperimentConfig
from sebs.utils import serialize


class PerfCost(Experiment):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    @staticmethod
    def name() -> str:
        return "perf-cost"

    @staticmethod
    def typename() -> str:
        return "Experiment.PerfCost"

    def prepare(self, sebs_client: "SeBS", deployment_client: FaaSSystem):

        from sebs import SeBS

        # create benchmark instance
        settings = self.config.experiment_settings(self.name())
        self._benchmark = sebs_client.get_benchmark(
            settings["benchmark"], deployment_client, self.config
        )
        self._function = deployment_client.get_function(self._benchmark)
        # prepare benchmark input
        self._storage = deployment_client.get_storage(replace_existing=True)
        self._benchmark_input = self._benchmark.prepare_input(
            storage=self._storage, size=settings["input-size"]
        )

        # add HTTP trigger
        triggers = self._function.triggers(Trigger.TriggerType.HTTP)
        if len(triggers) == 0:
            self._trigger = deployment_client.create_trigger(
                self._function, Trigger.TriggerType.HTTP
            )
        else:
            self._trigger = triggers[0]

        self._out_dir = os.path.join(sebs_client.output_dir, "perf-cost")
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        self._deployment_client = deployment_client

    def run(self):
       
        settings = self.config.experiment_settings(self.name())

        # Execution on systems where memory configuration is not provided
        memory_sizes = settings["memory-sizes"]
        if len(memory_sizes) == 0:
            self.logging.info("Begin experiment")
            self.run_configuration(settings["invocations"], settings["repetitions"])
        for memory in memory_sizes:
            self.logging.info(f"Begin experiment on memory size {memory}")
            self.run_configuration(settings["invocations"], settings["repetitions"], suffix=str(memory))

    def compute_statistics(self, times: List[float]):

        import numpy as np
        import scipy.stats as st

        mean = np.mean(times)
        std = np.std(times)
        cv = std/mean*100
        self.logging.info(f"Mean {mean}, std {std}, CV {cv}")
        for alpha in [0.9, 0.95, 0.99]:
            ci_interval = st.t.interval(alpha, len(times)-1, loc=mean, scale=st.sem(times))
            interval_width = ci_interval[1] - ci_interval[0]
            ratio = 100*interval_width / mean / 2.0
            self.logging.info(f"CI {alpha} from {ci_interval[0]} to {ci_interval[1]}, within {ratio}% of mean")

    def run_configuration(self, invocations: int, repetitions: int, suffix: str =""):

        # Randomize starting value to ensure that it's not the same as in the previous run.
        # Otherwise we could not change anything and containers won't be killed.
        from random import randrange
        self._deployment_client.cold_start_counter = randrange(100)
        self._deployment_client.enforce_cold_start(self._function, self._benchmark)
        # FIXME: update memory configuration
        # FIXME: repetitions

        with ThreadPool(invocations) as pool:
            """
                Cold experiment: schedule all invocations in parallel.
            """
            file_name = f"cold_results_{suffix}.json" if suffix else "cold_results.json"
            with open(os.path.join(self._out_dir, file_name), "w") as out_f:

                result = ExperimentResult(
                    self.config, self._deployment_client.config
                )
                client_times = []

                result.begin()
                results = []
                for i in range(0, invocations):
                    results.append(
                        pool.apply_async(self._trigger.sync_invoke, args=(self._benchmark_input,))
                    )
                result.end()

                for res in results:
                    ret = res.get()
                    result.add_invocation(self._function, ret)
                    client_times.append(ret.times.client / 1000.0)
                    if not ret.stats.cold_start:
                        self.logging.info(f"Invocation {ret.request_id} not cold!")
                self.compute_statistics(client_times)
                out_f.write(serialize(result))

            """
                Warm experiment: schedule many invocations in parallel.
                Here however it doesn't matter if they're perfectly aligned.
            """
            file_name = f"warm_results_{suffix}.json" if suffix else "warm_results.json"
            with open(os.path.join(self._out_dir, file_name), "w") as out_f:

                result = ExperimentResult(
                    self.config, self._deployment_client.config
                )
                client_times = []

                result.begin()
                results = []
                for i in range(0, invocations):
                    results.append(
                        pool.apply_async(self._trigger.sync_invoke, args=(self._benchmark_input,))
                    )
                result.end()

                for res in results:
                    ret = res.get()
                    result.add_invocation(self._function, ret)
                    client_times.append(ret.times.client / 1000.0)
                self.compute_statistics(client_times)
                out_f.write(serialize(result))

    def process(self, directory: str):
        pass
