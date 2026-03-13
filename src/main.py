import pathlib
import subprocess
import requests
import random
import re
from typing import Literal

from enum import StrEnum
import yaml
from jinja2 import Template
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, AnyHttpUrl
import networkx
from concurrent.futures import ThreadPoolExecutor


class CommunicationStyle(StrEnum):
    ASSERTIVE = "Assertive"
    AGGRESSIVE = "Aggressive"
    PASSIVE = "Passive"
    PASSIVE_AGGRESSIVE = "Passive-Aggressive"


COMMUNICATION_DESCRIPTIONS = {
    CommunicationStyle.ASSERTIVE: "Assertive - You express your opinions confidently and directly while respecting others. You stand firm on your beliefs but remain open to discussion.",
    CommunicationStyle.AGGRESSIVE: "Aggressive - You express opinions forcefully and dominantly. You tend to dismiss others' views and push your perspective strongly.",
    CommunicationStyle.PASSIVE: "Passive - You tend to avoid conflict and go along with others. You rarely express strong opinions and are easily influenced by those around you.",
    CommunicationStyle.PASSIVE_AGGRESSIVE: "Passive-Aggressive - You express disagreement indirectly. You may appear to agree outwardly while harboring different views internally.",
}

REALISTIC_DISTRIBUTION = {
    CommunicationStyle.ASSERTIVE: 0.30,
    CommunicationStyle.AGGRESSIVE: 0.20,
    CommunicationStyle.PASSIVE: 0.30,
    CommunicationStyle.PASSIVE_AGGRESSIVE: 0.20,
}


class LLMModels(StrEnum):
    LLAMA = "llama3.2:1b"
    LLAMA_3B = "llama3.2:3b"
    MISTRAL = "mistral"
    LLAMA_8B = "llama3.1:8b"


MIN_AGREEMENT = 1
MAX_AGREEMENT = 10


class Agent(BaseModel):
    url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434/api/generate")
    id: NonNegativeInt
    communication_style: CommunicationStyle
    initial_agreement: int = Field(ge=MIN_AGREEMENT, le=MAX_AGREEMENT)
    requests: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    agreements: list[int] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    neighbor_opinions: list[str] = Field(default_factory=list)
    context: list[int] = Field(default_factory=list)
    num_context: PositiveInt = 1024
    temperature: float = 0.2
    is_bot: bool = False
    bot_message: str = ""

    def model_post_init(self, __context):
        if not self.agreements:
            self.agreements.append(self.initial_agreement)


class DisinformationBotConfig(BaseModel):
    enabled: bool = False
    count: int = 1
    opinion: int = Field(ge=MIN_AGREEMENT, le=MAX_AGREEMENT)
    message: str


class AgentConfig(BaseModel):
    count: PositiveInt = 40


def build_agents(
    config: AgentConfig, bot_config: DisinformationBotConfig | None = None
) -> list[Agent]:
    """Builds"""
    agents: list[Agent] = []
    agent_id = 0

    for style, proportion in REALISTIC_DISTRIBUTION.items():
        agents_per_style = round(config.count * proportion)
        for _ in range(agents_per_style):
            initial_agreement = 3 if (agent_id % 2) == 0 else 7
            agent = Agent(
                id=agent_id,
                communication_style=style,
                initial_agreement=initial_agreement,
            )
            agents.append(agent)
            agent_id += 1
            print(
                f"Added agent {agent.id} with agreement {agent.initial_agreement} and {agent.communication_style} communication"
            )

    if bot_config and bot_config.enabled:
        for _ in range(bot_config.count):
            bot_agent = Agent(
                id=agent_id,
                communication_style=random.choice(list(CommunicationStyle)),
                initial_agreement=bot_config.opinion,
                is_bot=True,
                bot_message=bot_config.message,
            )
            agents.append(bot_agent)
            agent_id += 1
            print(
                f"Added disinformation bot (Agent {bot_agent.id}) with opinion {bot_config.opinion}"
            )

    return agents


def prompt(agent: Agent, model: str, prompt_msg: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt_msg,
        "stream": False,
        "context": agent.context,
        "options": {
            "num_ctx": agent.num_context,
            "temperature": agent.temperature,
        },
    }
    agent.requests.append(prompt_msg)

    try:
        response = requests.post(str(agent.url), json=payload, timeout=120)

        if response.status_code == 200:
            response_dict = response.json()
            context = response_dict.get("context")
            agent.context = context if context else []
            response_msg = response_dict.get("response", "")
            if not response_msg:
                print(f"Agent {agent.id}: Empty response, retrying")
                return prompt(agent, model, prompt_msg)
            agent.responses.append(response_msg)
            parse_and_update_agreement(agent, response_msg)
            return response_msg
        else:
            print(f"Error: {response.status_code} - {response.text}")
            agent.responses.append("")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Request failed for agent {agent.id}: {e}")
        agent.responses.append("")
        return ""


def parse_and_update_agreement(agent: Agent, response: str) -> None:
    pattern = (
        r"my current attitude towards the statement\s+.+?\s+is\s+(\d{1,2})\s+"
        r"on a scale from 1 to 10"
    )
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        new_agreement = int(match.group(1))
        new_agreement = max(MIN_AGREEMENT, min(MAX_AGREEMENT, new_agreement))
        agent.agreements.append(new_agreement)
        because_match = re.search(r"because (.+?)(?:\.|$)", response, re.IGNORECASE)
        if because_match:
            agent.reasons.append(because_match.group(1).strip())
        else:
            agent.reasons.append("")
    else:
        agent.agreements.append(agent.agreements[-1] if agent.agreements else 5)
        agent.reasons.append("")



class ERGraphConfig(BaseModel):
    type: Literal["erdos_renyi"] = "erdos_renyi"
    p: float = Field(ge=0.0, le=1.0)

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.erdos_renyi_graph(n=n, p=self.p)


class RGGGraphConfig(BaseModel):
    type: Literal["random_geometric"] = "random_geometric"
    radius: float = Field(gt=0.0)

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.random_geometric_graph(n=n, radius=self.radius)


class ScaleFreeGraphConfig(BaseModel):
    type: Literal["scale_free"] = "scale_free"
    m: int = Field(ge=1, default=2)

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.barabasi_albert_graph(n=n, m=self.m)


class SmallWorldGraphConfig(BaseModel):
    type: Literal["small_world"] = "small_world"
    k: int = Field(ge=2, default=4)
    p: float = Field(ge=0.0, le=1.0, default=0.3)

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.watts_strogatz_graph(n=n, k=self.k, p=self.p)


class SweepConfig(BaseModel):
    p: list[float] | None = None
    radius: list[float] | None = None
    m: list[int] | None = None
    k: list[int] | None = None


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    num_trials: PositiveInt = 1
    time_steps: PositiveInt = 20
    llm_model: LLMModels = LLMModels.LLAMA
    topic: str = "AI regulation should be mandatory for all companies"
    agents: AgentConfig = Field(default_factory=AgentConfig)
    graph_type: Literal[
        "erdos_renyi", "random_geometric", "scale_free", "small_world"
    ] = "erdos_renyi"
    graph_params: dict = Field(default_factory=lambda: {"p": 0.5})
    sweep: SweepConfig | None = None
    bot: DisinformationBotConfig

    @staticmethod
    def from_yaml(p: pathlib.Path) -> "ExperimentConfig":
        with open(p) as f:
            config = yaml.safe_load(f)
        return ExperimentConfig.model_validate(config)


EXPERIMENT_FOLDER = pathlib.Path("../experiments/")
RESULTS_FOLDER = pathlib.Path("../results/")

AGENT_PROMPT = Template(
    """You are a person with a {{ style_desc }} communication style.

Your current attitude towards the statement: "{{ topic }}" is {{ current_agreement }} on a scale from 1 to 10 where 1 is full disagreement and 10 is full agreement.

{% if neighbor_opinions %}
Others have shared their opinions:
{% for opinion in neighbor_opinions %}
- {{ opinion }}
{% endfor %}
{% endif %}

Based on your communication style, current level of agreement, and opinions of others, respond with your updated opinion:
"My current attitude towards the statement {{ topic }} is {rank} on a scale from 1 to 10 where 1 is full disagreement and 10 is full agreement. 
I think so because {reasoning in max 50 words}."
"""
)


def build_agent_prompt(agent: Agent, topic: str) -> str:
    return AGENT_PROMPT.render(
        style_desc=COMMUNICATION_DESCRIPTIONS[agent.communication_style],
        topic=topic,
        current_agreement=(
            agent.agreements[-1] if agent.agreements else agent.initial_agreement
        ),
        neighbor_opinions=(
            agent.neighbor_opinions[-10:] if agent.neighbor_opinions else []
        ),
    )


class Experiment(BaseModel):
    config: ExperimentConfig
    agents: list[Agent] = Field(default_factory=list)
    time: int = 0
    trial_id: int = 0
    param_value: float | None = None

    def model_post_init(self, __context):
        if not self.agents:
            self.agents = build_agents(self.config.agents, self.config.bot)


class ExperimentRunner:
    def __init__(self) -> None:
        self._config: ExperimentConfig | None = None
        self._running = False
        self.graph: networkx.Graph | None = None
        self._current_experiment: Experiment | None = None

    def run(self):
        if self._running:
            raise RuntimeError("ExperimentRunner is already running")
        elif self._config is None:
            raise ValueError("ExperimentRunner has no active config")

        self._running = True
        self.start_llm()

        if self._config.sweep:
            self._run_sweep()
        else:
            self._run_single_config()

        self.stop_llm()
        self._running = False

    def _run_sweep(self):
        assert self._config is not None
        sweep = self._config.sweep
        assert sweep is not None

        if sweep.p:
            param_values = sweep.p
            param_name = "p"
        elif sweep.radius:
            param_values = sweep.radius
            param_name = "radius"
        elif sweep.m:
            param_values = sweep.m
            param_name = "m"
        elif sweep.k:
            param_values = sweep.k
            param_name = "k"
        else:
            print("No sweep parameters defined")
            return

        for param_value in param_values:
            for trial in range(self._config.num_trials):
                print(f"\n{'='*50}")
                print(
                    f"Running {param_name}={param_value}, trial {trial+1}/{self._config.num_trials}"
                )
                print(f"{'='*50}")

                agents = build_agents(self._config.agents, self._config.bot)
                self._current_experiment = Experiment(
                    config=self._config,
                    agents=agents,
                    trial_id=trial,
                    param_value=param_value,
                )

                graph_config = self._create_graph_config_for_sweep(
                    param_name, param_value
                )

                self.build_graph(graph_config)
                self._run_experiment_loop()
                self._save_results(param_name, param_value, trial)

    def _run_single_config(self):
        assert self._config is not None

        for trial in range(self._config.num_trials):
            print(f"\n{'='*50}")
            print(f"Running trial {trial+1}/{self._config.num_trials}")
            print(f"{'='*50}")

            agents = build_agents(self._config.agents, self._config.bot)
            self._current_experiment = Experiment(
                config=self._config, agents=agents, trial_id=trial
            )

            graph_config = self._create_graph_config(
                self._config.graph_type, self._config.graph_params
            )

            self.build_graph(graph_config)
            self._run_experiment_loop()
            self._save_results("single", 0, trial)

    def _run_experiment_loop(self):
        assert self._current_experiment is not None
        while self._current_experiment.time < self._config.time_steps:
            self.step_experiment()

    def _save_results(self, param_name: str, param_value: float, trial: int):
        assert self._current_experiment is not None
        RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

        for agent in self._current_experiment.agents:
            agent.context.clear()

        filename = f"{self._config.name}_{param_name}{param_value}_trial{trial}.yaml"
        filepath = RESULTS_FOLDER / filename

        with open(filepath, "w") as f:
            data = self._current_experiment.model_dump()
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Results saved to {filepath}")

    def start_llm(self) -> bool:
        assert self._config is not None
        llm_model = self._config.llm_model
        result = subprocess.run(["ollama", "pull", llm_model])
        if result.returncode != 0:
            print("Failed to pull ollama model")
            return False
        print(f"Ollama model {llm_model} ready")
        return True

    def stop_llm(self):
        assert self._config is not None
        llm = self._config.llm_model
        subprocess.run(["ollama", "stop", llm], cwd="..", capture_output=True)

    def _create_graph_config(self, graph_type: str, params: dict):
        if graph_type == "erdos_renyi":
            return ERGraphConfig(p=params.get("p", 0.5))
        elif graph_type == "random_geometric":
            return RGGGraphConfig(radius=params.get("radius", 0.5))
        elif graph_type == "scale_free":
            return ScaleFreeGraphConfig(m=params.get("m", 2))
        elif graph_type == "small_world":
            return SmallWorldGraphConfig(k=params.get("k", 4), p=params.get("p", 0.3))
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    def _create_graph_config_for_sweep(self, param_name: str, param_value: float):
        if param_name == "p":
            return ERGraphConfig(p=param_value)
        elif param_name == "radius":
            return RGGGraphConfig(radius=param_value)
        elif param_name == "m":
            return ScaleFreeGraphConfig(m=int(param_value))
        elif param_name == "k":
            return SmallWorldGraphConfig(k=int(param_value))
        else:
            raise ValueError(f"Unknown sweep parameter: {param_name}")

    def build_graph(self, graph_config):
        assert self._current_experiment is not None
        agent_list = self._current_experiment.agents
        self.graph = graph_config.to_graph(n=len(agent_list))
        agents_dict = {agent.id: agent for agent in agent_list}
        networkx.set_node_attributes(self.graph, agents_dict, name="agent")

    def step_experiment(self):
        assert self._current_experiment is not None
        assert self._config is not None
        assert self.graph is not None
        t: int = self._current_experiment.time

        tasks = []
        print(f"Running timestep {t}")
        nodelist = sorted(list(self.graph.nodes()))
        adjacency_list = networkx.to_dict_of_lists(self.graph, nodelist)

        for source_node_id, neighbor_node_ids in adjacency_list.items():
            source_agent = self.graph.nodes[source_node_id]["agent"]
            assert isinstance(source_agent, Agent)

            neighbor_outputs = [
                self.graph.nodes[neighbor_id]["agent"].responses[-1]
                for neighbor_id in neighbor_node_ids
                if len(self.graph.nodes[neighbor_id]["agent"].responses) > 0
            ]
            source_agent.neighbor_opinions.extend(neighbor_outputs)

            if source_agent.is_bot:
                source_agent.requests.append("[BOT]")
                source_agent.responses.append(source_agent.bot_message)
                source_agent.agreements.append(source_agent.initial_agreement)
                source_agent.reasons.append("I am a bot spreading this message.")
            else:
                agent_prompt = build_agent_prompt(source_agent, self._config.topic)
                tasks.append((source_agent, agent_prompt))

        llm_model = self._config.llm_model
        with ThreadPoolExecutor(max_workers=2) as executor:
            for agent, msg in tasks:
                executor.submit(prompt, agent, llm_model, msg)

        self._current_experiment.time += 1

    @property
    def running(self):
        return self._running

    def load_config(self, config_path: pathlib.Path) -> bool:
        if self.running:
            print("Cannot load config while experiment is running")
            return False
        self._config = ExperimentConfig.from_yaml(config_path)
        print(f"Config loaded: {self._config.name}")
        return True


def main():
    import sys

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config_path = pathlib.Path(f"../experiments/{config_name}.yaml")
    else:
        config_path = pathlib.Path("../experiments/er_sweep.yaml")

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    runner = ExperimentRunner()
    runner.load_config(config_path)
    runner.run()


if __name__ == "__main__":
    main()
