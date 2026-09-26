"""LangGraph 版检索 Agent：与 agent.py 行为完全相同，只把“循环”换成状态图。

    uv run python src/agent_graph.py --run-name agent_v1_graph --golden golden_v2.jsonl

工具、参数校验、提示词、LLM 封装都复用 agent.py，这里只定义图：

    START → model ─┬─ 没有工具调用 ──→ text_answer → END
                   └─ 有工具调用 ──→ tools ─┬─ 已提交答案 / 强制轮结束 → END
                                            └─ 否则 ──────────────────→ model

对话历史用普通 dict 列表、不用 add_messages：LangChain 的消息对象会改变序列化结果，
导致缓存键不同，就没法用“全部命中缓存”来证明两个版本等价。
"""

from __future__ import annotations

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent import (
    ANSWER_ONLY,
    FORCE_ANSWER,
    TOOLS,
    Corpus,
    Episode,
    execute_calls,
    finalize,
    system_prompt,
    text_answer,
    write_report,
)
from answer import LLM, summarize
from evaluate import GOLDEN, REPORTS, ROOT, load_jsonl


class State(TypedDict):
    messages: list[dict]
    step: int  # 已经调用模型的次数
    final: dict | None
    stop: str
    trajectory: list[dict]
    usage: dict
    cached_all: bool


def model_node(state: State, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    step = state["step"] + 1
    forced = step > cfg["max_steps"]  # 步数用完：只留 final_answer 一个工具
    messages = state["messages"] + ([{"role": "user", "content": FORCE_ANSWER}] if forced else [])
    resp = cfg["llm"].chat_tools(messages, ANSWER_ONLY if forced else TOOLS, cfg["session"])
    usage = {k: state["usage"][k] + resp["usage"][k] for k in state["usage"]}
    return {
        "messages": messages + [resp["message"]],
        "step": step,
        "usage": usage,
        "cached_all": state["cached_all"] and resp["cached"],
    }


def tools_node(state: State, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    trajectory = list(state["trajectory"])
    calls = state["messages"][-1]["tool_calls"]
    forced = state["step"] > cfg["max_steps"]
    tool_msgs, final = execute_calls(cfg["ep"], calls, state["step"], trajectory, allow_guard=not forced)
    stop = ("forced_answer" if forced else "final_answer") if final else state["stop"]
    return {"messages": state["messages"] + tool_msgs, "trajectory": trajectory, "final": final, "stop": stop}


def text_answer_node(state: State) -> dict:
    trajectory = state["trajectory"] + [{"step": state["step"], "tool": None, "note": "模型直接输出文字"}]
    return {"final": text_answer(state["messages"][-1]), "stop": "text_answer", "trajectory": trajectory}


def after_model(state: State) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else "text_answer"


def after_tools(state: State, config: RunnableConfig) -> str:
    if state["final"] or state["step"] > config["configurable"]["max_steps"]:
        return END
    return "model"


def build_graph():
    g = StateGraph(State)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)
    g.add_node("text_answer", text_answer_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", after_model, ["tools", "text_answer"])
    g.add_conditional_edges("tools", after_tools, ["model", END])
    g.add_edge("text_answer", END)
    return g.compile()


GRAPH = build_graph()


def run_episode(q: dict, corpus: Corpus, llm: LLM, args: argparse.Namespace) -> dict:
    ep = Episode(corpus, q["question"], args.guard)
    init: State = {
        "messages": [
            {"role": "system", "content": system_prompt(args.agent_prompt, args.max_steps)},
            {"role": "user", "content": q["question"]},
        ],
        "step": 0,
        "final": None,
        "stop": "max_steps",
        "trajectory": [],
        "usage": {"prompt": 0, "completion": 0, "reasoning": 0},
        "cached_all": True,
    }
    session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"insurance-rag-eval/{args.run_name}/{q['id']}"))
    config = {
        "configurable": {"ep": ep, "llm": llm, "session": session, "max_steps": args.max_steps},
        "recursion_limit": 4 * args.max_steps + 10,
    }
    out = GRAPH.invoke(init, config)
    return finalize(q, ep, out["final"], out["stop"], out["trajectory"], out["usage"], out["cached_all"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--golden", default=GOLDEN.name)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--agent-prompt", default="a2")
    ap.add_argument("--guard", type=int, default=1, help="引用校验护栏最多退回几次，0 = 关闭")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--print-graph", action="store_true", help="输出 Mermaid 图后退出")
    args = ap.parse_args()
    args.engine = "LangGraph"
    if args.print_graph:
        print(GRAPH.get_graph().draw_mermaid())
        return

    corpus = Corpus()
    golden = load_jsonl(GOLDEN.parent / args.golden)
    if args.only:
        golden = [q for q in golden if q["id"] in set(args.only.split(","))]
    llm = LLM(None)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda q: run_episode(q, corpus, llm, args), golden))

    with (REPORTS / "runs" / f"{args.run_name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(rows)
    report = write_report(rows, s, args, llm)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, ensure_ascii=False))
    print(f"report -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
