"""Voice enrolment (ADR 0021, amendment): the enroll_voice / replace_voice tools and the
enrolment the voice session leads in code. Fakes only: no microphone, model or
real speaker identifier."""

import asyncio
import random
from pathlib import Path
from typing import Any

import pytest

from lucy.core.config import (
    AdapterConfig,
    LLMSettings,
    Settings,
    SpeakersSettings,
    WorkspaceSettings,
)
from lucy.core.context import ContextBuilder
from lucy.core.events import (
    AssistantMessage,
    ConfirmationResolved,
    TimerFired,
    ToolFinished,
    UserMessage,
)
from lucy.core.gateway import LLMGateway
from lucy.core.orchestrator import Orchestrator
from lucy.core.ports import SpeakerMatch
from lucy.core.reminders import REMINDER_TIMER
from lucy.core.types import FinishReason, LLMChunk, Risk, ToolCall, ToolOutput
from lucy.core.workspace import Workspace
from lucy.tools.voice_enrolment import (
    ENROLMENT_TIMER,
    EnrollVoiceTool,
    ReplaceVoiceTool,
    clean_name,
    handle_enrolment,
)
from lucy.voice.enrol import SENTENCES, Enroller, is_cancel, overlap
from lucy.voice.session import VoiceSession
from tests.fakes import FakeLLM
from tests.test_voice_session import TIMEOUT, Harness, reply, until
from tests.voice_fakes import concat, silence, tone


class FakeIdentifier:
    def __init__(self, enrolled: dict[str, int] | None = None, *, recognise: bool = True) -> None:
        self.book = dict(enrolled or {})
        self.enrolled: list[tuple[str, list[Path]]] = []
        self.removed: list[str] = []
        self.recognise = recognise

    def identify(self, path: Path) -> list[SpeakerMatch]:
        if not self.book:
            return []
        name = next(iter(self.book))
        return [SpeakerMatch(name, 0.8 if self.recognise else 0.1, self.recognise)]

    def enroll(self, name: str, paths: list[Path]) -> None:
        assert all(p.is_file() for p in paths)
        self.enrolled.append((name, list(paths)))
        self.book[name] = len(paths)

    def speakers(self) -> dict[str, int]:
        return dict(self.book)

    def remove(self, name: str) -> None:
        self.removed.append(name)
        self.book.pop(name, None)


def settings_for(tmp_path: Path, *, enabled: bool = True, weights: bool = True) -> Settings:
    path = tmp_path / "w.onnx"
    if weights:
        path.write_bytes(b"x")
    return Settings(
        llm=LLMSettings(roles={"chat": AdapterConfig(adapter="fake")}),
        speakers=SpeakersSettings(enabled=enabled, weights=str(path)),
        workspace=WorkspaceSettings(path=str(tmp_path / "ws")),
    )


# --- The tools --------------------------------------------------------------------------


async def test_enroll_voice_requests_a_session_for_a_new_name(tmp_path: Path) -> None:
    ident = FakeIdentifier()
    tool = EnrollVoiceTool(settings_for(tmp_path), speakers=ident.speakers, voice_client=True)
    out = await tool.run({"name": "Asha"})
    assert not out.is_error and "Lucy leads it herself" in out.content and "Asha" in out.content


async def test_enroll_voice_refuses_an_enrolled_name_and_names_replace_voice(
    tmp_path: Path,
) -> None:
    ident = FakeIdentifier({"Yogesh": 3})
    tool = EnrollVoiceTool(settings_for(tmp_path), speakers=ident.speakers, voice_client=True)
    out = await tool.run({"name": "yogesh"})
    assert out.is_error and "replace_voice" in out.content


@pytest.mark.parametrize(
    ("enabled", "weights", "voice", "hint"),
    [
        (True, True, False, "lucy voice"),
        (False, True, True, "[speakers]"),
        (True, False, True, "lucy speakers download"),
    ],
)
async def test_tools_refuse_where_enrolment_cannot_run(
    tmp_path: Path, enabled: bool, weights: bool, voice: bool, hint: str
) -> None:
    settings = settings_for(tmp_path, enabled=enabled, weights=weights)
    ident = FakeIdentifier({"Yogesh": 3})
    for tool in (
        EnrollVoiceTool(settings, speakers=ident.speakers, voice_client=voice),
        ReplaceVoiceTool(settings, speakers=ident.speakers, voice_client=voice),
    ):
        out = await tool.run({"name": "Yogesh" if tool.name == "replace_voice" else "Asha"})
        assert out.is_error and hint in out.content


BAD_NAMES = ["", "   ", "../etc", "a/b", "x" * 40, "my", "O'Neil", ".hidden", 7]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_names_are_validated(bad: Any) -> None:
    with pytest.raises(ValueError):
        clean_name(bad)


def test_plain_names_are_kept() -> None:
    assert clean_name("  Mary   Ann ") == "Mary Ann"
    assert clean_name("Mary-Ann") == "Mary-Ann"


async def test_replace_voice_asks_and_only_a_yes_runs(tmp_path: Path) -> None:
    ident = FakeIdentifier({"Yogesh": 3})
    tool = ReplaceVoiceTool(settings_for(tmp_path), speakers=ident.speakers, voice_client=True)
    preview = await tool.preview({"name": "yogesh"})
    assert preview.content == "Replace the enrolled voice for Yogesh?"
    missing = await tool.preview({"name": "Asha"})
    assert missing.is_error and "enroll_voice" in missing.content

    from tests.fakes import InMemoryMemoryStore

    async def turn(answer: str, scripts: list[Any]) -> list[ToolFinished]:
        from lucy.core.bus import EventBus

        bus = EventBus()
        memory = InMemoryMemoryStore()
        orch = Orchestrator(
            bus=bus,
            gateway=LLMGateway({"chat": FakeLLM(scripts)}),
            memory=memory,
            context=ContextBuilder("You are Lucy."),
            tools=[tool],
        )
        sub = bus.subscribe(ToolFinished, AssistantMessage)
        task = asyncio.create_task(orch.run())
        await bus.publish(UserMessage(session_id="s", text="enrol my voice again"))
        await asyncio.sleep(0.05)
        await bus.publish(UserMessage(session_id="s", text=answer))
        await asyncio.sleep(0.05)
        bus.close()
        await task
        events = []
        while not sub._queue.empty():
            events.append(sub._queue.get_nowait())
        return [e for e in events if isinstance(e, ToolFinished)]

    call = ToolCall(id="c1", name="replace_voice", arguments={"name": "Yogesh"})
    ask = [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]
    yes = await turn("yes", [ask, reply("Replace the enrolled voice for Yogesh?"), reply("Ok.")])
    assert [f.result.is_error for f in yes][-1] is False
    assert "starting now" in yes[-1].result.content
    no = await turn("no", [ask, reply("Replace?"), reply("Ok.")])
    assert no[-1].result.is_error


async def test_the_outcome_handler_says_the_text() -> None:
    event = TimerFired(session_id="s", name=ENROLMENT_TIMER, payload={"text": "Done."})
    assert (await handle_enrolment(event, "t")).text == "Done."  # type: ignore[union-attr]


def test_loose_checks() -> None:
    assert overlap(SENTENCES[0], SENTENCES[0].upper()) == 1.0
    assert overlap(SENTENCES[0], "the quick brown fox jumped over a lazy dog by the river") > 0.5
    assert overlap(SENTENCES[0], "what's the weather") < 0.5
    assert is_cancel("Cancel.") and is_cancel("stop") and not is_cancel("stop the train")


# --- The session --------------------------------------------------------------------------

READ = 4.0  # seconds of "speech" per reading: over the 3 s minimum
GAP = 10.0  # silence after each utterance, enough for Lucy's cue and the tail


def enrol_call(name: str = "Asha") -> list[LLMChunk]:
    call = ToolCall(id="e1", name="enroll_voice", arguments={"name": name})
    return [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]


def harness(
    tmp_path: Path,
    readings: list[float],
    transcripts: list[str],
    *,
    ident: FakeIdentifier | None = None,
    scripts: list[Any] | None = None,
    sentences: int = 3,
    silence_s: float = 20.0,
    tail: float = GAP,
    tools: list[Any] | None = None,
    handlers: dict[str, Any] | None = None,
    replace: bool = False,
) -> tuple[Harness, FakeIdentifier, Enroller]:
    ident = ident or FakeIdentifier()
    settings = settings_for(tmp_path)
    tool = EnrollVoiceTool(settings, speakers=ident.speakers, voice_client=True)
    replacer = ReplaceVoiceTool(settings, speakers=ident.speakers, voice_client=True)
    audio = [tone(0.6), silence(GAP)]
    if replace:
        audio += [tone(0.6), silence(GAP)]  # the answer to the question
    for seconds in readings:
        audio += [tone(seconds), silence(GAP)]
    audio[-1] = silence(tail)
    h = Harness([concat(*audio)], [], transcripts, orchestrate=False)
    enroller = Enroller(
        ident,
        Workspace(tmp_path / "ws"),
        sentences=sentences,
        silence_s=silence_s,
        rng=random.Random(1),
    )
    h.session = VoiceSession(
        bus=h.bus, session_id="cli", io=h.io, vad=h.vad, stt=h.stt, tts=h.tts, ui=h.ui,
        enroller=enroller,
    )  # fmt: skip
    h.llm = FakeLLM(scripts if scripts is not None else [enrol_call(), reply("Okay.")])
    h.orchestrator = Orchestrator(
        bus=h.bus,
        gateway=LLMGateway({"chat": h.llm}),
        memory=h.memory,
        context=ContextBuilder("You are Lucy."),
        tools=[tool, replacer, *(tools or [])],
        timer_handlers={ENROLMENT_TIMER: handle_enrolment, **(handlers or {})},
    )
    return h, ident, enroller


def chosen(enroller: Enroller) -> list[str]:
    return random.Random(1).sample(SENTENCES, enroller.sentences)


async def run(h: Harness) -> None:
    task = h.start()
    async with asyncio.timeout(TIMEOUT):
        await task


async def test_the_happy_path_enrols_and_logs_only_the_outcome(tmp_path: Path) -> None:
    sentences = random.Random(1).sample(SENTENCES, 3)
    h, ident, _ = harness(tmp_path, [READ] * 3, ["enrol my voice, I'm Asha", *sentences])
    async with h:
        await run(h)
    assert [name for name, _ in ident.enrolled] == ["Asha"]
    paths = ident.enrolled[0][1]
    assert len(paths) == 3 and all(p.parent == tmp_path / "ws" / "speakers" / "Asha" for p in paths)
    assert h.llm.calls and len(h.llm.calls) == 2  # only the request turn reached the model
    users = h.events(UserMessage)
    assert [u.text for u in users] == ["enrol my voice, I'm Asha"]
    logged = [e for e in h.memory.episodes if isinstance(e, AssistantMessage)]
    assert logged[-1].text == "Done. I'll recognise you as Asha."
    assert not any(isinstance(e, UserMessage) and e.text in sentences for e in h.memory.episodes)
    assert "First sentence." in h.tts.started and " ".join(h.tts.started).endswith(logged[-1].text)
    assert not any(s in h.tts.started for s in sentences)  # shown, never read out
    assert not list((tmp_path / "ws").glob("utterances/*.wav"))


async def test_a_short_or_wrong_reading_is_asked_again(tmp_path: Path) -> None:
    sentences = random.Random(1).sample(SENTENCES, 3)
    h, ident, _ = harness(
        tmp_path,
        [1.0, READ, READ, READ, READ],
        ["enrol Asha", sentences[0], sentences[0], "what's the weather", *sentences[1:]],
    )
    async with h:
        await run(h)
    assert ident.enrolled and len(ident.enrolled[0][1]) == 3
    assert any("too short" in t for t in h.tts.started)
    assert any("didn't sound like" in t for t in h.tts.started)


async def test_retries_exhausted_enrol_nothing(tmp_path: Path) -> None:
    h, ident, _ = harness(tmp_path, [READ] * 3, ["enrol Asha", "nope", "nope", "nope"])
    async with h:
        await run(h)
    assert ident.enrolled == []
    assert not (tmp_path / "ws" / "speakers" / "Asha").exists()
    assert "Nothing was enrolled" in " ".join(h.tts.started[-3:])


async def test_cancel_by_word_enrols_nothing(tmp_path: Path) -> None:
    h, ident, _ = harness(tmp_path, [1.0], ["enrol Asha", "cancel"])
    async with h:
        await run(h)
    assert ident.enrolled == [] and not (tmp_path / "ws" / "speakers" / "Asha").exists()
    assert "stopped" in " ".join(h.tts.started[-3:])
    assert [u.text for u in h.events(UserMessage)] == ["enrol Asha"]


async def test_silence_ends_the_enrolment(tmp_path: Path) -> None:
    h, ident, _ = harness(tmp_path, [], ["enrol Asha"], silence_s=3.0, tail=30.0)
    async with h:
        await run(h)
    assert ident.enrolled == []
    assert "didn't hear anything" in " ".join(h.tts.started[-3:])


async def test_enter_cancels_the_enrolment(tmp_path: Path) -> None:
    h, ident, _ = harness(tmp_path, [], ["enrol Asha"], tail=30.0)
    async with h:
        task = h.start()

        async def press() -> None:
            while not h.session.enrolling:
                await asyncio.sleep(0)
            await asyncio.sleep(0.01)
            assert h.session.interrupt()

        await asyncio.gather(press(), asyncio.wait_for(task, TIMEOUT))
    assert ident.enrolled == []
    assert "stopped" in " ".join(h.tts.started[-3:])


async def test_an_ordinary_turn_works_right_after(tmp_path: Path) -> None:
    sentences = random.Random(1).sample(SENTENCES, 3)
    h, ident, _ = harness(
        tmp_path,
        [READ] * 3 + [0.6],
        ["enrol Asha", *sentences, "what time is it"],
        scripts=[enrol_call(), reply("Okay."), reply("Noon.")],
        tail=GAP,
    )
    async with h:
        await run(h)
    assert ident.enrolled
    assert [u.text for u in h.events(UserMessage)] == ["enrol Asha", "what time is it"]
    assert h.tts.started[-1] == "Noon."


# --- Replacing through a whole voice session ----------------------------------------------


def replace_call(name: str = "Asha") -> list[LLMChunk]:
    call = ToolCall(id="r1", name="replace_voice", arguments={"name": name})
    return [LLMChunk(tool_call=call), LLMChunk(finish_reason=FinishReason.TOOL_CALLS)]


async def test_replacing_asks_and_a_spoken_yes_runs_the_enrolment(tmp_path: Path) -> None:
    sentences = random.Random(1).sample(SENTENCES, 3)
    ident = FakeIdentifier({"Asha": 3})
    h, ident, _ = harness(
        tmp_path,
        [READ] * 3,
        ["enrol Asha's voice", "Yes.", *sentences],
        ident=ident,
        scripts=[replace_call(), reply("Okay.")],
        replace=True,
    )
    old = tmp_path / "ws" / "speakers" / "Asha"
    old.mkdir(parents=True)
    (old / "old.wav").write_bytes(b"old")
    async with h:
        await run(h)
    assert h.tts.started[0] == "Replace the enrolled voice for Asha?"  # code-authored
    assert [e.outcome for e in h.events(ConfirmationResolved)] == ["confirmed"]
    assert ident.removed == ["Asha"] and [n for n, _ in ident.enrolled] == ["Asha"]
    assert sorted(p.name for p in old.iterdir()) == ["enrol_1.wav", "enrol_2.wav", "enrol_3.wav"]
    assert [u.text for u in h.events(UserMessage)] == ["enrol Asha's voice", "Yes."]
    assert " ".join(h.tts.started).endswith("Done. I'll recognise you as Asha.")


async def test_replacing_with_a_spoken_no_changes_nothing(tmp_path: Path) -> None:
    ident = FakeIdentifier({"Asha": 3})
    h, ident, _ = harness(
        tmp_path,
        [],
        ["enrol Asha's voice", "No."],
        ident=ident,
        scripts=[replace_call(), reply("Okay, I left it.")],
        replace=True,
    )
    async with h:
        await run(h)
    assert [e.outcome for e in h.events(ConfirmationResolved)] == ["declined"]
    assert ident.removed == [] and ident.enrolled == [] and ident.book == {"Asha": 3}
    assert not any("sentence" in t.lower() for t in h.tts.started)
    assert not h.session.enrolling


# --- A reminder due during an enrolment -------------------------------------------------


async def test_a_reminder_due_during_enrolment_waits_for_it_to_end(tmp_path: Path) -> None:
    from tests.test_voice_reminders import REMINDER, Reminders

    sentences = random.Random(1).sample(SENTENCES, 3)
    r = Reminders()
    h, ident, _ = harness(
        tmp_path,
        [READ] * 3,
        ["enrol Asha", *sentences],
        handlers={REMINDER_TIMER: r.handle},
        tail=20.0,
    )
    async with h:
        task = h.start()
        await until(lambda: h.session.enrolling)
        announced = asyncio.create_task(h.session.announce(r.fire))
        while h.session.enrolling:
            assert not announced.done() and r.due == 1  # not even claimed yet
            await asyncio.sleep(0)
        async with asyncio.timeout(TIMEOUT):
            assert await announced
            await task
    assert ident.enrolled
    spoken = h.tts.started
    done = spoken.index("I'll recognise you as Asha.")
    assert spoken.index(REMINDER) > done and r.delivered == 1


# --- A confirmation question and an enrolment ---------------------------------------------


class Dangerous:
    """A dangerous tool that records runs."""

    name = "wipe"
    description = "Wipe the thing."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    risk = Risk.DANGEROUS

    def __init__(self) -> None:
        self.runs = 0

    async def preview(self, args: dict[str, Any]) -> ToolOutput:
        return ToolOutput(content="Wipe the thing?")

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        self.runs += 1
        return ToolOutput(content="Wiped.")


async def test_readings_never_answer_a_question_asked_before_the_enrolment(
    tmp_path: Path,
) -> None:
    # The only way a question can be open during an enrolment: one turn both starts
    # the enrolment and asks about a dangerous call (an older question is already
    # ended by the "enrol" message itself). A "yes" read during the enrolment must
    # not confirm it; the outcome turn ends it; a later question still works.
    wipe = Dangerous()
    both = [
        LLMChunk(tool_call=ToolCall(id="e1", name="enroll_voice", arguments={"name": "Asha"})),
        LLMChunk(tool_call=ToolCall(id="w1", name="wipe", arguments={})),
        LLMChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]
    ask_again = [
        LLMChunk(tool_call=ToolCall(id="w2", name="wipe", arguments={})),
        LLMChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]
    h, ident, _ = harness(
        tmp_path,
        [READ, READ, READ, 0.6, 0.6],
        ["enrol Asha and wipe it", "yes", "yes", "yes", "wipe it", "yes"],
        tools=[wipe],
        scripts=[both, ask_again, reply("Wiped it.")],
        sentences=3,
    )
    async with h:
        await run(h)
    # Readings "yes" (not the sentence) never became messages; the enrolment stopped.
    assert ident.enrolled == []
    resolved = [e.outcome for e in h.events(ConfirmationResolved)]
    assert resolved == ["superseded", "confirmed"]  # ended by the outcome, then a real yes
    assert wipe.runs == 1
    assert [u.text for u in h.events(UserMessage)] == [
        "enrol Asha and wipe it",
        "wipe it",
        "yes",
    ]


# --- The wake listener --------------------------------------------------------------------


async def test_the_wake_listener_stays_awake_through_an_enrolment(tmp_path: Path) -> None:
    from lucy.voice.session import VoiceConfig
    from lucy.voice.wake import ListenerConfig, WakeListener, WakeStage, chime
    from lucy.voice.wakeword import ScriptedWakeWord
    from tests.test_voice_wake import loud, speech, wake_phrase

    sentences = random.Random(1).sample(SENTENCES, 3)
    h, ident, enroller = harness(tmp_path, [], ["enrol Asha", *sentences, "thanks"])
    h.llm = FakeLLM([enrol_call(), reply("Okay."), reply("You're welcome.")])
    h.orchestrator = Orchestrator(
        bus=h.bus,
        gateway=LLMGateway({"chat": h.llm}),
        memory=h.memory,
        context=ContextBuilder("You are Lucy."),
        tools=[EnrollVoiceTool(settings_for(tmp_path), speakers=ident.speakers, voice_client=True)],
        timer_handlers={ENROLMENT_TIMER: handle_enrolment},
    )
    # Silences of 10 s between readings: far over the 3 s follow-up window. Loud
    # sounds at wake-phrase level are captured while Lucy speaks her cues.
    parts = [silence(0.5), wake_phrase(), speech(0.8), silence(GAP)]
    for i, _ in enumerate(sentences):
        last = i == len(sentences) - 1
        parts += [speech(READ), silence(0.9), wake_phrase(0.3), silence(3.5 if last else GAP)]
    parts += [speech(0.8), silence(8.0)]
    from lucy.voice.audio import FileAudioIO

    h.io = FileAudioIO([concat(*parts)])

    async def make_session(io: Any, ui: Any) -> VoiceSession:
        h.session = VoiceSession(
            bus=h.bus, session_id="cli", io=io, vad=h.vad, stt=h.stt, tts=h.tts,
            config=VoiceConfig(), ui=ui, enroller=enroller,
        )  # fmt: skip
        return h.session

    listener = WakeListener(
        io=h.io,
        stage=WakeStage(ScriptedWakeWord(score=loud)),
        make_session=make_session,
        ui=h.ui,
        config=ListenerConfig(followup_s=3.0, chime=True),
    )
    async with h:
        async with asyncio.timeout(TIMEOUT):
            await listener.run()
    assert len(listener.wakes) == 1  # no wake phrase needed, never woke on loud sounds
    assert ident.enrolled and len(ident.enrolled[0][1]) == 3
    assert [u.text for u in h.events(UserMessage)] == ["enrol Asha", "thanks"]
    assert h.tts.started[-1] == "You're welcome."
    assert len(listener.sleeps) == 1  # only after the last reply's follow-up window
    sleep_chime = chime((660.0, 990.0)).frames
    assert sum(1 for p in h.io.plays if p.samples == sleep_chime) >= 1


# --- Names that are not names (the "my voice" bug) -----------------------------------------


@pytest.mark.parametrize(
    "placeholder",
    ["my voice", "My Voice", "  my   voice ", "my", "me", "myself", "mine", "I", "user",
     "the user", "voice", "your voice", "owner", "speaker", "unknown", "guest",
     "Asha voice", "my wife", "the boss"],
)  # fmt: skip
def test_placeholders_and_phrases_are_not_names(placeholder: str) -> None:
    with pytest.raises(ValueError, match="not a person's name"):
        clean_name(placeholder)


def test_names_are_normalised() -> None:
    assert clean_name(" yogesh ") == "Yogesh"
    assert clean_name("mary-ann  smith") == "Mary-Ann Smith"
    assert clean_name("McKay") == "McKay"  # mixed case is kept as given


async def test_my_voice_is_refused_by_both_tools_with_what_to_do(tmp_path: Path) -> None:
    ident = FakeIdentifier({"Yogesh": 3})
    settings = settings_for(tmp_path)
    enrol = EnrollVoiceTool(settings, speakers=ident.speakers, voice_client=True)
    replace = ReplaceVoiceTool(settings, speakers=ident.speakers, voice_client=True)
    for out in (
        await enrol.run({"name": "my voice"}),
        await replace.preview({"name": "my voice"}),
        await replace.run({"name": "my voice"}),
    ):
        assert out.is_error
        assert out.content == (
            "'my voice' is not a person's name, so the enrolment was not started. Call this "
            "tool again with the person's actual name: for the user's own voice, their name "
            "if it is in what you know about them; otherwise ask the user what name to enrol "
            "the voice under, and call the tool once they answer."
        )
    assert "never 'my voice'" in enrol.parameters["properties"]["name"]["description"]


async def test_a_my_voice_call_starts_no_enrolment(tmp_path: Path) -> None:
    h, ident, _ = harness(
        tmp_path,
        [],
        ["Can you enroll my voice?"],
        scripts=[enrol_call("my voice"), reply("What name should I enrol it under?")],
    )
    async with h:
        await run(h)
    assert ident.enrolled == [] and not h.session.enrolling
    assert h.tts.started == ["What name should I enrol it under?"]
    assert not (tmp_path / "ws" / "speakers").exists()


# --- The sentence appears only once Lucy is listening (the clipping bug) -------------------


async def test_each_sentence_is_shown_only_after_the_cue_and_the_tail(tmp_path: Path) -> None:
    from lucy.voice.session import VoiceState

    sentences = random.Random(1).sample(SENTENCES, 3)
    h, ident, _ = harness(
        tmp_path,
        [1.0, READ, READ, READ],  # the first reading is too short: asked again
        ["enrol Asha", sentences[0], *sentences],
    )
    shown: list[tuple[str, int, int, float, float, VoiceState]] = []
    last_said: list[str] = []

    def enrol_sentence(number: int, count: int, sentence: str) -> None:
        session = h.session
        expected = sum(h.tts.samples_for(t) for t in h.tts.started)
        last_said.append(h.tts.started[-1])
        shown.append(
            (sentence, h.io.played_samples, expected, h.io.now(), session._gate_until,
             session.state)
        )  # fmt: skip

    h.ui.enrol_sentence = enrol_sentence  # type: ignore[attr-defined]
    async with h:
        await run(h)
    assert ident.enrolled
    assert [s[0] for s in shown] == [sentences[0], sentences[0], sentences[1], sentences[2]]
    for _sentence, played, expected, now, gate, state in shown:
        assert played == expected  # every cue so far, incl. the one just before, fully played
        assert now > gate  # past the post-playback drop
        assert state == VoiceState.LISTENING
    # The cue that belongs to each showing had already been spoken when it appeared.
    assert last_said == [
        "First sentence.",
        "That was too short. Please read it again.",
        "Second sentence.",
        "Third sentence.",
    ]
