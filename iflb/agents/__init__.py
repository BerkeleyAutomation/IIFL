from .bc_agent import SingleTaskParallelBCAgent
from .random_agent import SingleTaskParallelRandomAgent
from .ibc_agent import SingleTaskParallelImplicitBCAgent

# Mappings from CLI option strings to agents
agent_map = {
    "BC": SingleTaskParallelBCAgent,
    "Random": SingleTaskParallelRandomAgent,
    "IBC": SingleTaskParallelImplicitBCAgent
}

agent_cfg_map = {
    "BC": "bc_agent.yaml",
    "IBC": "ibc_agent.yaml"
}