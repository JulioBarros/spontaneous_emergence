"""Re-implementation of SPONTANEOUS EMERGENCE OF AGENT INDIVIDUALITY THROUGH SOCIAL INTERACTIONS IN LLM-BASED COMMUNITIES by
Ryosuke Takata, Atsushi Masumori, Takashi Ikegami.
There may be errors, misunderstandings and other differences but seems to show similar results.
"""

import os
import random
import json
from datetime import datetime
import argparse

import requests


# The message prompts as described in the paper with minor formatting differences
GEN_MESSAGE_PROMPT = """
You are {name} at position ({x_pos}, {y_pos}). 
The field size is {grid_size} by {grid_size} with periodic boundary 
conditions, and there are a total of {num_agents} agents. 
You are free to move around the field and converse with other agents. 
You have a summary memory of the situation so far: {memory}. 
You received messages from the surrounding agents: {messages_received}. 
Based on the above, you send a message to the surrounding agents. 
Your message will reach agents up to distance {max_distance} away. 
What message do you send?
"""

GEN_MEMORY_PROMPT = """
You are {name} at position ({x_pos}, {y_pos}). 
The field size is {grid_size} by {grid_size} with periodic boundary 
conditions, and there are a total of {num_agents} agents. 
You are free to move around the field and converse with other agents. 
You have a summary memory of the situation so far: {memory}. 
You received messages from the surrounding agents: {messages_received}.
Based on the above, summarize the situation you and the other agents have been in so far for you to remember.
"""

GEN_MOVE_PROMPT = """
You are {name} at position ({x_pos}, {y_pos}). 
The field size is {grid_size} by {grid_size} with periodic boundary 
conditions, and there are a total of {num_agents} agents. 
You are free to move around the field and converse with other agents. 
You have a summary memory of the situation so far: {memory}. 
You received messages from the surrounding agents: {messages_received}.
Based on the above, what the next your move command?
Choose only one of the following: ["x+1", "x-1", "y+1", "y-1", "stay"]
"""


def run_prompt(prompt: str, cfg: argparse.Namespace) -> str:
    """Submits the prompt to an LLM. I'm using ollama but you can modify this to use any provider.
    The ollama options are not clearly documented so may not be correct.
    """
    options = {
        "num_predict": cfg.num_predict,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
    }
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": cfg.model_name,
        "options": options,
        "stream": False,
        "prompt": prompt,
    }
    resp = requests.post(url, json=payload)
    jresp = resp.json()
    text = jresp["response"]

    return text


class Agent:
    """Implements the functionality described in the paper. I tend to prefer a functional approach but
    understand that OO is more common. Ended up with a mixture of the two for clarity and simplicity."""

    def __init__(self, name: str, x_pos: int, y_pos: int):
        """Start out with a name and position and not much else."""

        self.name = name
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.prev_x = -1
        self.prev_y = -1
        self.memory = ""
        self.generated_message = ""
        self.received_messages: list[str] = []
        self.generated_move = ""

    def prep_for_step(self):
        """For each step we track the messages received and our position."""

        self.prev_x = self.x_pos
        self.prev_y = self.y_pos
        self.generated_message = ""
        self.received_messages = []
        self.generated_move = ""

    def generate_message(
        self,
        cfg: argparse.Namespace,
    ) -> str:
        """At the begining of each step we generate a message to send out. Received messages are specified in the prompt
        though we shouldn't actually have any yet but that may be an misunderstanding on my part."""

        prompt = GEN_MESSAGE_PROMPT.format(
            name=self.name,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            grid_size=cfg.grid_size,
            num_agents=cfg.num_agents,
            memory=self.memory,
            messages_received="\n".join(self.received_messages),
            max_distance=cfg.max_distance,
        )
        self.generated_message = run_prompt(prompt, cfg)
        return self.generated_message

    def receive_message(self, message: str):
        """Keep track of the messages we've received in this step."""

        self.received_messages.append(message)

    def generate_memory(self, cfg: argparse.Namespace):
        """Use the received messages and our memor to form a new memory."""

        prompt = GEN_MEMORY_PROMPT.format(
            name=self.name,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            grid_size=cfg.grid_size,
            num_agents=cfg.num_agents,
            memory=self.memory,
            messages_received="\n".join(self.received_messages),
        )
        self.memory = run_prompt(prompt, cfg)

    def generate_move(self, cfg: argparse.Namespace):
        """Generate and apply a move. The models seem to generate very verbose output with the specified prompt
        so I look for the first instance of a valid move. If no valid move is present we 'stay'.
        Again, the prompt specifies to use the messages received but we've already incorporated that into the
        memory. Again could be a misunderstanding on my part."""

        prompt = GEN_MOVE_PROMPT.format(
            name=self.name,
            x_pos=self.x_pos,
            y_pos=self.y_pos,
            grid_size=cfg.grid_size,
            num_agents=cfg.num_agents,
            memory=self.memory,
            messages_received="\n".join(self.received_messages),
        )
        move = run_prompt(prompt, cfg)

        # Parse and apply and track the move. Prepend the move we decided on so that it is logged.
        if "y+1" in move:
            self.y_pos += 1
            move = "(y+1): " + move
        elif "y-1" in move:
            self.y_pos -= 1
            move = "(y-1): " + move
        elif "x-1" in move:
            self.x_pos -= 1
            move = "(x-1): " + move
        elif "x+1" in move:
            self.x_pos += 1
            move = "(x+1): " + move
        elif "stay" in move:
            move = "(stay): " + move
        else:
            move = "(invalid)" + move
            print(f"Move not valid for {self.name}: {move}")

        self.generated_move = move

        # make sure we wrap around if we go off the edges
        if self.x_pos < 0:
            self.x_pos += cfg.grid_size
        if self.y_pos < 0:
            self.y_pos += cfg.grid_size
        self.x_pos = self.x_pos % cfg.grid_size
        self.y_pos = self.y_pos % cfg.grid_size

    def distance(self, other: "Agent") -> int:
        """Calculate Chebyshev distance specified in the paper."""
        return max(abs(self.x_pos - other.x_pos), abs(self.y_pos - other.y_pos))


def run_simulation(cfg: argparse.Namespace):
    """Go through all the simulation steps specified."""

    agents = []
    for i in range(cfg.num_agents):
        x_pos = random.randint(0, cfg.grid_size - 1)
        y_pos = random.randint(0, cfg.grid_size - 1)
        agent = Agent(
            name=f"agent_{i}",
            x_pos=x_pos,
            y_pos=y_pos,
        )
        agents.append(agent)

    for step in range(cfg.num_steps):
        print("Running step", step, end=" ")
        num_messages_sent = 0

        for agent in agents:
            agent.prep_for_step()

        # collect all the messages each agent generates
        messages = []
        for agent in agents:
            messages.append(agent.generate_message(cfg))

        # distribute each message to the appropriate neighbors
        for agent, message in zip(agents, messages):
            for other in agents:
                if other != agent and agent.distance(other) <= cfg.max_distance:
                    num_messages_sent += 1
                    other.receive_message(message)

        # now that each agent has received all its messages, generate a new memory
        for agent in agents:
            agent.generate_memory(cfg)

        # using the memory, generate a new move for each agent
        for agent in agents:
            agent.generate_move(cfg)

        # log the state of the world.
        print(" num messages sent", num_messages_sent)
        step_dir = os.path.join(cfg.run_dir, f"step{step}")
        os.makedirs(step_dir)
        for agent in agents:
            with open(os.path.join(step_dir, agent.name), "wt") as f:
                json.dump(agent.__dict__, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate an agent grid world as described in https://arxiv.org/abs/2411.03252v1"
    )
    parser.add_argument("-a", "--num-agents", type=int, default=10)
    parser.add_argument("-g", "--grid-size", type=int, default=50)
    parser.add_argument("-s", "--num-steps", type=int, default=100)
    parser.add_argument("-d", "--max-distance", type=int, default=5)
    parser.add_argument("-m", "--model-name", type=str, default="llama3.1")
    parser.add_argument("-r", "--num-predict", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-p", "--top-p", type=float, default=0.95)
    parser.add_argument("-k", "--top-k", type=int, default=40)
    parser.add_argument("-o", "--output-dir", type=str, default="runs")
    parser.add_argument(
        "-i",
        "--run-id",
        type=str,
        default=datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"),
    )

    cfg = parser.parse_args()
    run_dir = os.path.join(cfg.output_dir, cfg.run_id)
    os.makedirs(run_dir, exist_ok=False)
    cfg.run_dir = run_dir

    print(json.dumps(cfg.__dict__, indent=2))
    with open(f"{cfg.run_dir}/config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    run_simulation(cfg)


if __name__ == "__main__":
    main()
