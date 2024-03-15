import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


callbacks = [StreamingStdOutCallbackHandler()]
local_path = "D:/falcon-7b-instruct.ggccv1.q8_0.bin"

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load the LLM model from the HuggingFaceHub
repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 100}
)

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

# Function to generate the agent's action based on the game state, agent's role, and other agent's action
def generate_agent_action(agent, game_state, memory, other_agent_action):
    current_objective = get_current_objective(agent, game_state)
    
    prompt = PromptTemplate(
        input_variables=["game_state", "agent_name", "agent_goal", "memory", "other_agent_action", "current_objective"],
        template="""
        You are {agent_name}, in a dungeon exploration game. Your goal is to {agent_goal}.
        
        The current situation in the dungeon is:
        {game_state}
        
        Your memory of previous actions and thoughts:
        {memory}
        
        Recently, another agent did this:
        {other_agent_action}
        
        Your current objective is: {current_objective}
        
        Considering the above, what action can you take that will bring you closer to achieving your current objective? Please consider:
        - How does this action align with your objective?
        - Is there anything in your surroundings that can help you achieve your objective?
        - How does the other agent's action affect your decision to pursue your objective?
        
        Remember, every move should contribute to your objectives and help you progress in the game.
        """
    )
    
    action = falcon_llm(prompt.format(
        game_state=game_state["description"],
        agent_name=agent["name"],
        agent_goal=agent["goal"],
        memory=memory,
        other_agent_action=other_agent_action,
        current_objective=current_objective
    ))
    
    return action.strip()

# Game loop with objective-driven prompts and action validation
num_turns = 5
game_log = []

for turn in range(num_turns):
    turn_log = f"\nTurn {turn + 1}:\nCurrent Game State:\n{game_state['description']}\n"
    game_log.append(turn_log)
    
    for agent in agent_roles:
        agent_name = agent["name"]
        memory = ConversationBufferMemory()
        
        other_agent_name = "Adventurer" if agent_name == "Mage" else "Mage"
        other_agent_action = game_state["agents"].get(other_agent_name, "")
        
        action = generate_agent_action(agent, game_state, memory.buffer, other_agent_action)
        
        while not validate_action(action, agent, game_state):
            action = generate_agent_action(agent, game_state, memory.buffer, other_agent_action)
        
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