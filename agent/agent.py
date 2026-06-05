import operator
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from llm import LocalDecoderLLM

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def read_local_document(file_path: str) -> str:
    print(f"\n[System] Reading file: {file_path} ...")
    return f"Document {file_path} info: project about local model deployment."

def agent_reasoning_node(state: AgentState) -> dict:
    llm = LocalDecoderLLM()
    
    history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    
    system_prompt = """你是一个专业的学习辅导智能体导师。你可以使用工具来获取学习资料。
规则如下：
1. 如果你需要读取学习材料文件，请严格按照此格式输出：[CALL_TOOL]: read_local_document(文件路径)
2. 当你通过工具掌握了文档内容后，你的任务不是简单地总结内容，而是根据文档内容向用户提出 2 到 3 个具有启发性的问题，来考核用户的学习情况。

当前对话历史：
{history}
请给出你的思考，或者向用户提出问题："""

    prompt = system_prompt.format(history=history)
    
    print("\n[Agent 思考中...]")
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response)]}

def tool_execution_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    
    start_idx = last_message.find("read_local_document(") + len("read_local_document(")
    end_idx = last_message.find(")", start_idx)
    
    file_path = last_message[start_idx:end_idx].strip("\"'")
    
    tool_result = read_local_document(file_path)
    
    return {"messages": [SystemMessage(content=f"工具执行结果：{tool_result} \n请根据此材料内容，向用户提出启发式的问题以考核对方。")]}

def should_continue_router(state: AgentState) -> str:
    last_message = state["messages"][-1].content
    
    if "[CALL_TOOL]" in last_message:
        return "continue_to_tool"
    else:
        return "end_workflow"

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("tool", tool_execution_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue_router,
    {
        "continue_to_tool": "tool",
        "end_workflow": END
    }
)

workflow.add_edge("tool", "agent")

app_agent = workflow.compile()

if __name__ == "__main__":
    print("启动本地私有化学习辅导 Agent 系统...")
    
    user_input = "我刚刚学习了 data/华智招新.pdf 文件里的内容，请你根据里面的内容考考我吧。"
    print(f"\n用户: {user_input}")
    
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    for output in app_agent.stream(initial_state, config={"recursion_limit": 5}):
        for key, value in output.items():
            pass 
            
    final_messages = app_agent.get_state(config={"configurable": {"thread_id": "1"}}).values.get("messages", [])
    if final_messages:
        print(f"\n最终输出:\n{final_messages[-1].content}")