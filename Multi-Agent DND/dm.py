import os
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the GPT4All model
local_path = "D:/guanaco-7B.ggmlv3.q5_1.bin"
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)

# Game scenario and initial state
game_scenario = "You find yourself in a mysterious dungeon, surrounded by ancient stone walls and flickering torches. A sense of adventure and danger fills the air. You notice a locked door ahead and a rusty key lying on a nearby table."
game_state = {
    "description": game_scenario,
    "agents": {},
    "objectives": {
        "Adventurer": {"Find the ancient artifact": False, "Escape the dungeon": False},
        "Mage": {"Gather ancient knowledge": False, "Wield powerful magic": False}
    },
    "locations": ["Entrance", "Chamber 1"],
    "inventory": {"Adventurer": [], "Mage": []}
}

# Agent roles and their goals
agent_roles = [
    {"name": "Adventurer", "goal": "Seek treasure and uncover the secrets of the dungeon."},
    {"name": "Mage", "goal": "Gather ancient knowledge and wield powerful magic."},
]

# Dungeon Master prompt template
dungeon_master_template = """
As the Dungeon Master in a Dungeons and Dragons game, your key role is to navigate the agents through their journey, presenting them with both challenges and opportunities to make their adventure engaging and dynamic.

Given the Current Game State, Agent Roles, and their Objectives outlined below, craft the next segment of our adventure. Your narration should unfold in a structured manner, consisting of a vivid Scene Description, a set of tangible Challenges, and actionable Opportunities for the agents. Please adhere strictly to the format provided to ensure clarity and coherence in our shared narrative.

- Current Game State: {game_state}
- Agent Roles: {agent_roles}
- Objectives: {objectives}

Here's what you need to provide:

Scene Description:
[Elaborate on the current scene, highlighting the environment, atmosphere, and interactive elements. Imagine setting the stage for a pivotal moment in the story.]

Challenges:
- [Challenge 1: Describe a specific obstacle or challenge the agents face.]
- [Challenge 2: Outline another hurdle, testing the agents' skills or decision-making.]

Opportunities:
- [Opportunity 1: Detail a potential advantage or beneficial action the agents can take.]
- [Opportunity 2: Describe another opportunity for progression or advantage in the game.]

Example:
Scene Description: The adventurers find themselves in a dimly lit cavern, echoes of distant drips and the faint smell of moss fill the air. An ancient stone door stands closed before them, inscribed with runes that glow softly.
Challenges:
- Decipher the ancient runes to unlock the door.
- Navigate the treacherous paths without alerting the sleeping dragon.
Opportunities:
- Use the glowing mushrooms to create a potion that reveals hidden paths.
- Befriend the dragon to gain a powerful ally or access to its hoard.

Please adhere to this format for your response, adding depth and intrigue to our adventure without straying from the outlined structure.
"""

dungeon_master_prompt = PromptTemplate(
    input_variables=["game_state", "agent_roles", "objectives"],
    template=dungeon_master_template,
)

# Player Agent prompt template
player_agent_template = """
You are {agent_name}, a character in a Dungeons and Dragons game. Your goal is to {agent_goal}.

The current situation is:
{game_state}

The Dungeon Master has provided the following scene description:
{scene_description}

Your current objective is: {current_objective}

Considering the above, describe your action and thoughts. Focus on actions that align with your current objective and help you progress in the game.

Your previous actions and thoughts:
{memory}

Action and Thoughts:
"""

player_agent_prompt = PromptTemplate(
    input_variables=["game_state", "agent_name", "agent_goal", "memory", "scene_description", "current_objective"],
    template=player_agent_template,
)

# Custom output parser for the dungeon master agent
class DungeonMasterOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        scene_description = ""
        challenges = ""
        opportunities = ""
            
        # Try to extract the scene description
        scene_description_match = re.search(r"Scene Description:\s*(.*?)\s*Challenges:", llm_output, re.DOTALL)
        if scene_description_match:
            scene_description = scene_description_match.group(1).strip()
            
        # Try to extract the challenges
        challenges_match = re.search(r"Challenges:\s*(.*?)\s*Opportunities:", llm_output, re.DOTALL)
        if challenges_match:
            challenges = challenges_match.group(1).strip()
            
        # Try to extract the opportunities
        opportunities_match = re.search(r"Opportunities:\s*(.*)", llm_output, re.DOTALL)
        if opportunities_match:
            opportunities = opportunities_match.group(1).strip()
            
        if scene_description or challenges or opportunities:
            return AgentFinish(
                return_values={
                    "scene_description": scene_description,
                    "challenges": challenges,
                    "opportunities": opportunities
                },
                log=llm_output,
            )
        else:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

dungeon_master_output_parser = DungeonMasterOutputParser()
# Function to get the current objective of the agent
def get_current_objective(agent, game_state):
    agent_objectives = game_state["objectives"][agent["name"]]
    for objective, completed in agent_objectives.items():
        if not completed:
            return objective
    return None

# Function to update the game state based on the agent's action
def update_game_state(game_state, agent_name, action):
    # Implement game state update logic based on the agent's action
    # Update the game state dictionary with relevant changes
    # Example: Update the agent's location, inventory, or objectives
    game_state["description"] += f"\n\n{agent_name}'s action: {action}"
    game_state["agents"][agent_name] = action
    
    # Example objective completion logic
    if agent_name == "Adventurer" and "find" in action.lower() and "artifact" in action.lower():
        game_state["objectives"]["Adventurer"]["Find the ancient artifact"] = True
    elif agent_name == "Mage" and "learn" in action.lower() and "spell" in action.lower():
        game_state["objectives"]["Mage"]["Wield powerful magic"] = True
    
    return game_state

# Function to validate an action based on objectives
def validate_action(action, agent, game_state):
    current_objective = get_current_objective(agent, game_state)
    if current_objective:
        # Check if the action aligns with the current objective
        if current_objective.lower() in action.lower():
            return True
    return False

# Game loop with dungeon master agent and player agents
num_turns = 5
game_log = []

for turn in range(num_turns):
    # Dungeon Master Agent
    dungeon_master_chain = LLMChain(llm=llm, prompt=dungeon_master_prompt)
    dungeon_master_agent = LLMSingleActionAgent(
        llm_chain=dungeon_master_chain,
        output_parser=dungeon_master_output_parser,
        stop=["\nScene Description:"],
        allowed_tools=[],
    )
    dungeon_master_executor = AgentExecutor.from_agent_and_tools(agent=dungeon_master_agent, tools=[], verbose=True)
    
    scene_description = dungeon_master_executor.run(
        game_state=game_state["description"],
        agent_roles=agent_roles,
        objectives=game_state["objectives"]
    )
    
    game_log.append(f"\nTurn {turn + 1}:")
    game_log.append(f"Dungeon Master's Scene Description:\n{scene_description}\n")
    
    # Player Agents
    for agent in agent_roles:
        agent_name = agent["name"]
        memory = ConversationBufferMemory()
        
        player_agent_chain = LLMChain(llm=llm, prompt=player_agent_prompt)
        player_agent = LLMSingleActionAgent(
            llm_chain=player_agent_chain,
            output_parser=AgentOutputParser(),
            stop=["\nAction and Thoughts:"],
            allowed_tools=[],
        )
        player_agent_executor = AgentExecutor.from_agent_and_tools(agent=player_agent, tools=[], verbose=True)
        
        current_objective = get_current_objective(agent, game_state)
        
        action = player_agent_executor.run(
            game_state=game_state["description"],
            agent_name=agent_name,
            agent_goal=agent["goal"],
            memory=memory.buffer,
            scene_description=scene_description,
            current_objective=current_objective
        )
        
        while not validate_action(action, agent, game_state):
            action = player_agent_executor.run(
                game_state=game_state["description"],
                agent_name=agent_name,
                agent_goal=agent["goal"],
                memory=memory.buffer,
                scene_description=scene_description,
                current_objective=current_objective
            )
        
        memory.save_context({"game_state": game_state["description"]}, {"action": action})
        
        game_state = update_game_state(game_state, agent_name, action)
        
        agent_log = f"{agent_name}'s Action and Thoughts:\n{action}\n"
        game_log.append(agent_log)

final_log = "\nFinal Game State:\n" + game_state["description"]
game_log.append(final_log)

# Save the game log to a file
game_log_filename = os.path.join(output_dir, "game_log.txt")
with open(game_log_filename, "w") as log_file:
    log_file.write("\n".join(game_log))

print(f"Game log saved to: {game_log_filename}")