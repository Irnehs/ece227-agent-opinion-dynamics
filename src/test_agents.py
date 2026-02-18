import pathlib
import subprocess

import yaml
from jinja2 import Template
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
    llm_model: str
    use_nvidia: bool = False


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


DOCKER_COMPOSE_TEMPLATE = Template("""
services:
{% for agent in agents %}
  agent-{{ agent.id }}:
    image: ollama/ollama
    ports:
      - "{{ agent.port }}:11434"
    volumes:
      - agent-{{ agent.id }}_data:/root/.ollama
      - ./shared_models:/root/.ollama/models
    {% if use_nvidia %}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    {% endif %}
{% endfor %}

volumes:
{% for agent in agents %}
  agent-{{ agent.id }}_data:
{% endfor %}
""")

EXPERIMENT_FOLDER = pathlib.Path("../experiments/")
DOCKER_COMPOSE_PATH = pathlib.Path("../docker-compose.yaml")


class ExperimentRunner:
    def __init__(self) -> None:
        self._experiment: Experiment | None = None
        self._running = False

    def run(self):
        """Runs the experiment defined by the current config file"""
        if self._running:
            raise RuntimeError("ExperimentRunnner is already running")
        elif self._experiment is None:
            raise ValueError("ExperimentRunnner has no active eperiment")
        self.setup_experiment()
        for _ in range(self._experiment.time_steps):
            self.step_experiment()
        self.cleanup_experiment()

    def setup_experiment(self):
        self.build_docker_compose()
        self.start_agents()

    def cleanup_experiment(self):
        self.stop_agents()

    def build_docker_compose(self):
        text = DOCKER_COMPOSE_TEMPLATE.render(self._experiment.agent_group.model_dump())
        print(text)
        with open(DOCKER_COMPOSE_PATH, "w") as f:
            f.write(text)

    def start_agents(self) -> bool:
        first_agent_id = f"agent-{self._experiment.agent_group.agents[0].id}"
        llm_model = self._experiment.agent_group.llm_model
        result = subprocess.run(
            ["docker", "compose", "up", "-d", first_agent_id], cwd=".."
        )
        if result.returncode != 0:
            print("Failed to bring up first agent")
            return False
        result = subprocess.run(
            ["docker", "compose", "exec", first_agent_id, "ollama", "pull", llm_model],
            cwd="..",
        )
        if result.returncode != 0:
            print(f"Failed to get first agent to dowload model {llm_model}")
            return False
        result = subprocess.run(["docker", "compose", "up", "-d"], cwd="..")
        if result.returncode != 0:
            print("Failed to bring up remaining agents")
            return False
        else:
            print("Agents started")
            return True

    def stop_agents(self):
        subprocess.run(["docker", "compose", "down"], cwd="..")

    def step_experiment(self):
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
    runner.run()


if __name__ == "__main__":
    main()
