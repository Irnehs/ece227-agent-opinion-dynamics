import pathlib

import yaml
from pydantic import BaseModel, Field


class Personality(BaseModel):
    personality_type: str = "1"


class Agent(BaseModel):
    port: int
    id: int
    personality: Personality
    requests: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)

    @property
    def hostname(self):
        return f"http://localhost:{self.port}/"


class AgentGroup(BaseModel):
    agents: list[Agent]
    startup_instructions: str


class GraphModel(BaseModel):
    nodes: dict[int, list[int]] = Field(default_factory=dict)


class Experiment(BaseModel):
    agent_group: AgentGroup
    graph: GraphModel
    time_steps: int

    @staticmethod
    def from_yaml(p: pathlib.Path) -> "Experiment":
        with open(p) as f:
            config = yaml.safe_load(f)
        return Experiment.model_validate(config)


class ExperimentRunner:
    def __init__(self) -> None:
        self._active_experiment: Experiment | None = None
        self._running = False

    def run(self):
        """Runs the experiment defined by the current config file"""
        # Run the experiment, then dump the outputs to a file
        pass

    @property
    def running(self):
        return self._running

    def load_experiment(self, experiment: pathlib.Path) -> bool:
        """
        Loads an experiment from a given filepath.

        Returns True if the experiment is successfully loaded, False otherwise.
        """
        if self.running:
            print("Cannot run an experiment when an experiment is already active")
            return False
        self._experiment = Experiment.from_yaml(experiment)
        print("Experiment loaded succesfully")
        return True


def main():
    dummy_experiment = pathlib.Path("../experiments/dummy_experiment.yaml")
    runner = ExperimentRunner()
    runner.load_experiment(dummy_experiment)


if __name__ == "__main__":
    main()
