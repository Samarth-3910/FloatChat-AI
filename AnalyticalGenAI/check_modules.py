import sys
import langchain_classic.agents
import langchain_core.tools

print(f"create_react_agent in langchain_classic.agents: {hasattr(langchain_classic.agents, 'create_react_agent')}")
print(f"initialize_agent in langchain_classic.agents: {hasattr(langchain_classic.agents, 'initialize_agent')}")
print(f"Tool in langchain_core.tools: {hasattr(langchain_core.tools, 'Tool')}")
  