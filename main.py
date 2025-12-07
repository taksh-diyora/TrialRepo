from agents import graph
from langchain_core.messages import HumanMessage

print("\n🚀 Multi-Agent AI System Initialized")
print("--- Start Chatting (type 'quit' to exit) ---\n")

while True:
    user_input = input("User: ")

    if user_input.lower().strip() in ["quit", "exit", "bye"]:
        print("\n👋 Goodbye!")
        break

    inputs = {"messages": [HumanMessage(content=user_input)]}

    for event in graph.stream(inputs):
        for node_name, node_state in event.items():
            print(f"\n--- Output from {node_name} ---")
            messages = node_state.get("messages", [])
            if messages:
                # Print only the last message from that node (to avoid duplicates)
                last_msg = messages[-1]
                print(last_msg.content)
