import pathlib
import subprocess
import requests

from enum import StrEnum
import yaml
from jinja2 import Template
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, AnyHttpUrl, HttpUrl
import networkx
from concurrent.futures import ThreadPoolExecutor, as_completed

class PersonalityType(StrEnum):
    ISFJ = "ISFJ (Protector)"
    ESFJ = "ISFJ (Provider)"
    ISTJ = "ISTJ (Inspector)"
    ISFP = "ISFP (Artist)"

class LLMModels(StrEnum):
    LLAMA = "llama3.2:1b"

MIN_AGREEMENT = 1
MAX_AGREEMENT = 10

class Agent(BaseModel):
    url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434/api/generate")
    id: NonNegativeInt
    personality: PersonalityType
    requests: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    agreements: list[int] = []
    reasons: list[str] = []
    
    response_period: PositiveInt = 1
    response_offset: NonNegativeInt = 0
    neighbor_opinions : list[str] = Field(default_factory=list)
    context: list[int] = Field(default_factory=list)
    num_context: PositiveInt = 1024
    temperature: float = 0.2

def build_agents() -> list[Agent]:
    agents: list[Agent] = [] 
    i = 0
    for personality in PersonalityType:
        for agreement in range(MIN_AGREEMENT, MAX_AGREEMENT+1):
            agents.append(Agent(id=i,personality=personality,initial_agreement=agreement))
            i += 1
    return agents
    


def prompt(agent: Agent, model: str,  prompt_msg: str):
    print("executing prompt")
    
    payload = {
        "model": model,
        "prompt": prompt_msg,
        "stream": False, 
        "context": agent.context,
        "options": {
            "num_ctx": agent.num_context,
            "temperature": agent.temperature,
        }
    }
    agent.requests.append(prompt_msg)
    
    response = requests.post(str(agent.url), json=payload)
    
    if response.status_code == 200:
        response_dict = response.json()
        context = response_dict.get('context')
        agent.context = context
        response_msg = response_dict.get("response")
        if not response_msg: 
            print("Failed to get a response. Retrying")
            return prompt(agent, model, prompt_msg)
        agent.responses.append(response_msg)

        return response_dict.get("response")
    else:
        print(f"Error processing response: {response.status_code} - {response.text}")
        return "Unable to process request"

class ERGraphConfig(BaseModel):
    p: float

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.erdos_renyi_graph(n=n, p=self.p)

class RGGGraphConfig(BaseModel):
    r: float

    def to_graph(self, n: PositiveInt) -> networkx.Graph:
        return networkx.random_geometric_graph(n=n, radius=self.r)

class Experiment(BaseModel):
    agents: list[Agent] = Field(default_factory=build_agents)
    time_steps: int
    llm_model: LLMModels
    graph: ERGraphConfig | RGGGraphConfig
    time: int = 0

    @staticmethod
    def from_yaml(p: pathlib.Path) -> "Experiment":
        with open(p) as f:
            config = yaml.safe_load(f)
        return Experiment.model_validate(config)

    


EXPERIMENT_FOLDER = pathlib.Path("../experiments/")
AGENT_PROMPT = Template("""
Pretend you are a person with the MBTI personality type {{ agent.personality }}. You currently agree with a statement at a level of {{ agent.agreements[-1] }} on a scale from 1 to 10 where 1 is full disagreement and 10 is full agreement.
{% for opinion in agent.neighbor_opinions %}
Someone else thinks: {{ opinion }}
{% endfor %}
Considering the opinions of others around you in accordance with your personality type and your current level of agreement, give me your updated level of agreement formatted as “I agree at a level of {agreement} from 1 to 10 .\nI think so because {your reasoning behind the ranking as a single sentence with a maximum of 30 words}.”
""")

def build_agent_prompt(agent: Agent) -> str:
    data = {
        "agent" : agent,
    }
    return AGENT_PROMPT.render(data)


class ExperimentRunner:
    def __init__(self) -> None:
        self._experiment: Experiment | None = None
        self._running = False
        self.graph: networkx.Graph | None = None

    def run(self):
        """Runs the experiment defined by the current config file"""
        if self._running:
            raise RuntimeError("ExperimentRunnner is already running")
        elif self._experiment is None:
            raise ValueError("ExperimentRunnner has no active eperiment")
        self._running = True
        self.start_agents()
        self.build_graph()
        while self._experiment.time < self._experiment.time_steps:
            self.step_experiment()
        self.cleanup_experiment()


    def cleanup_experiment(self):
        self.stop_agents()
        assert self._experiment is not None
        # clear context to declutter yaml
        agents = self._experiment.agents 
        for i in range(len(agents)):
            agents[i].context.clear()
        
        with open("result.yaml", "w") as f:
            data = self._experiment.model_dump()
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        self._running = False

    def start_agents(self) -> bool:
        assert self._experiment is not None
        llm_model = self._experiment.llm_model
        result = subprocess.run(
            ["ollama", "pull", llm_model]
        )
        if result.returncode != 0:
            print("Failed to start ollama")
            return False
        else:
            print("Ollama started")
            return True

    def stop_agents(self):
        assert self._experiment is not None
        llm = self._experiment.llm_model
        subprocess.run(["ollama", "stop", llm], cwd="..")

    def build_graph(self):
        assert self._experiment is not None
        agent_list = self._experiment.agents
        self.graph = self._experiment.graph.to_graph(n=len(agent_list))
        agents_dict = {agent.id : agent for agent in agent_list}
        networkx.set_node_attributes(self.graph, agents_dict, name="agent")

    def step_experiment(self):
        assert self._experiment is not None
        assert self.graph is not None
        t: int = self._experiment.time

        tasks = [] 
        print(f"Running timestep {t}")
        nodelist = sorted(list(self.graph.nodes())) 
        adjacency_list = networkx.to_dict_of_lists(self.graph, nodelist)
        # Collect responses from neighbors and build prompts
        for source_node_id, neighbor_node_ids in adjacency_list.items():
            source_agent = self.graph.nodes[source_node_id]["agent"]
            assert isinstance(source_agent, Agent)
            
            # Collect neighbor outputs
            neighbor_outputs = [
                self.graph.nodes[neighbor_id]["agent"].responses[-1] for neighbor_id in neighbor_node_ids if len(self.graph.nodes[neighbor_id]["agent"].responses) > 0
            ]
            source_agent.neighbor_opinions.extend(neighbor_outputs)

            # Build agent prompt
            if t >= source_agent.response_offset and ((t - source_agent.response_offset) % source_agent.response_period == 0) and t > 0:
                agent_prompt = build_agent_prompt(source_agent)
                tasks.append((source_agent, agent_prompt))
            else:
                source_agent.requests.append("")
                source_agent.responses.append("")
                    
        # Generate new outputs
        llm_model = self._experiment.llm_model
        with ThreadPoolExecutor(max_workers=len(nodelist)) as executor:
            for agent, msg in tasks:
                executor.submit(prompt, agent, llm_model, msg)

        self._experiment.time += 1

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
