from lucy.core.events import UserMessage
from lucy.core.ports import LLM, STT, TTS, Embedder, MemoryStore, Tool, TraceSink, tool_spec
from lucy.core.types import FinishReason, LLMChunk, Message, Role, ToolCall
from tests.fakes import EchoTool, FakeEmbedder, FakeLLM, InMemoryMemoryStore, InMemoryTraceSink


def test_fakes_satisfy_their_protocols() -> None:
    assert isinstance(FakeLLM([]), LLM)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(InMemoryMemoryStore(), MemoryStore)
    assert isinstance(EchoTool(), Tool)
    assert isinstance(InMemoryTraceSink(), TraceSink)


def test_protocols_reject_unrelated_objects() -> None:
    for port in (LLM, STT, TTS, Embedder, MemoryStore, Tool, TraceSink):
        assert not isinstance(object(), port)


async def test_fake_llm_replays_scripts_and_records_messages() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    llm = FakeLLM(
        [
            [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)],
            [LLMChunk(text="Hi "), LLMChunk(text="there.", finish_reason=FinishReason.STOP)],
        ]
    )
    first = [c async for c in llm.stream([Message(role=Role.USER, content="hello")])]
    second = [c async for c in llm.stream([Message(role=Role.USER, content="again")])]
    assert first[0].tool_call == call
    assert "".join(c.text or "" for c in second) == "Hi there."
    assert [m[0].content for m in llm.calls] == ["hello", "again"]
    assert llm.schemas_seen == [None, None]
    assert llm.exits == ["done", "done"] and llm.open_streams == 0


async def test_fake_llm_records_response_schema_and_early_close() -> None:
    llm = FakeLLM([[LLMChunk(text="{"), LLMChunk(text="}")]])
    schema = {"type": "object"}
    stream = llm.stream([Message(role=Role.USER, content="json")], response_schema=schema)
    assert await anext(stream) == LLMChunk(text="{")
    assert llm.open_streams == 1
    await stream.aclose()
    assert llm.schemas_seen == [schema]
    assert llm.exits == ["closed"] and llm.open_streams == 0


async def test_fake_embedder_is_deterministic() -> None:
    embedder = FakeEmbedder(dimensions=4)
    a, b, a2 = await embedder.embed(["a", "b", "a"])
    assert a == a2 and a != b and len(a) == 4
    (query_a,) = await embedder.embed(["a"], kind="query")
    assert query_a != a and len(query_a) == 4
    assert await embedder.embed([]) == []
    assert embedder.calls == [(["a", "b", "a"], "document"), (["a"], "query"), ([], "document")]


async def test_in_memory_store_records_episodes() -> None:
    store = InMemoryMemoryStore()
    event = UserMessage(session_id="s", text="hi")
    await store.append_episode(event)
    assert store.episodes == [event]
    context = await store.retrieve("hi", budget=100)
    assert context.profile == "" and context.statements == []


async def test_in_memory_store_recent_episodes_per_session_oldest_first() -> None:
    store = InMemoryMemoryStore()
    events = [UserMessage(session_id=sid, text=str(i)) for i, sid in enumerate("abaab")]
    for event in events:
        await store.append_episode(event)
    assert await store.recent_episodes("a", 2) == [events[2], events[3]]
    assert await store.recent_episodes("a", 10) == [events[0], events[2], events[3]]
    assert await store.recent_episodes("c", 10) == []


async def test_echo_tool_and_spec() -> None:
    tool = EchoTool()
    result = await tool.run({"text": "ping"})
    assert result.content == "ping" and not result.is_error
    spec = tool_spec(tool)
    assert spec.name == "echo" and spec.parameters["required"] == ["text"]
