"""
Testing Utilities - Helpers for testing agents.

Provides:
- Mock LLM providers for testing
- Test fixtures and builders
- Assertion helpers
- Recording and playback

From ROADMAP: Testing framework for agents
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MockResponse:
    """A mocked LLM response."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 100, "output_tokens": 50}
    )


class MockLLMProvider:
    """
    Mock LLM provider for testing.

    Usage:
        mock = MockLLMProvider()

        # Add responses
        mock.add_response("Hello! How can I help?")
        mock.add_tool_call("search", {"query": "test"})

        # Use in tests
        response = mock.complete(messages)
        assert response.content == "Hello! How can I help?"
    """

    def __init__(self):
        self._responses: list[MockResponse] = []
        self._response_index = 0
        self._calls: list[dict[str, Any]] = []
        self._default_response = MockResponse(content="Mock response")

    def add_response(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a response to the queue."""
        self._responses.append(
            MockResponse(
                content=content,
                tool_calls=tool_calls or [],
            )
        )

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Add a tool call response."""
        self._responses.append(
            MockResponse(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                ],
                stop_reason="tool_use",
            )
        )

    def set_default(self, content: str) -> None:
        """Set default response when queue is empty."""
        self._default_response = MockResponse(content=content)

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        **kwargs,
    ) -> MockResponse:
        """Simulate LLM completion."""
        self._calls.append(
            {
                "messages": messages,
                "system": system,
                "kwargs": kwargs,
                "timestamp": time.time(),
            }
        )

        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response

        return self._default_response

    def get_calls(self) -> list[dict[str, Any]]:
        """Get all recorded calls."""
        return self._calls

    def get_last_call(self) -> dict[str, Any] | None:
        """Get the last call."""
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Reset the mock."""
        self._response_index = 0
        self._calls = []


class RecordingProvider:
    """
    Records LLM calls for playback.

    Usage:
        # Record
        recorder = RecordingProvider(real_provider)
        response = recorder.complete(messages)
        recorder.save("recording.json")

        # Playback
        playback = PlaybackProvider.load("recording.json")
        response = playback.complete(messages)
    """

    def __init__(self, provider: Any):
        self._provider = provider
        self._recordings: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        **kwargs,
    ) -> Any:
        """Complete with recording."""
        response = self._provider.complete(messages, system=system, **kwargs)

        self._recordings.append(
            {
                "request": {
                    "messages": messages,
                    "system": system,
                    "kwargs": kwargs,
                },
                "response": {
                    "content": getattr(response, "content", str(response)),
                    "tool_calls": getattr(response, "tool_calls", []),
                    "stop_reason": getattr(response, "stop_reason", "end_turn"),
                },
                "timestamp": time.time(),
            }
        )

        return response

    def save(self, path: str | Path) -> None:
        """Save recordings to file."""
        with open(path, "w") as f:
            json.dump(self._recordings, f, indent=2)

    def get_recordings(self) -> list[dict[str, Any]]:
        """Get recordings."""
        return self._recordings


class PlaybackProvider:
    """
    Plays back recorded LLM calls.

    Usage:
        playback = PlaybackProvider.load("recording.json")
        response = playback.complete(messages)
    """

    def __init__(self, recordings: list[dict[str, Any]]):
        self._recordings = recordings
        self._index = 0

    @classmethod
    def load(cls, path: str | Path) -> "PlaybackProvider":
        """Load recordings from file."""
        with open(path) as f:
            recordings = json.load(f)
        return cls(recordings)

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        **kwargs,
    ) -> MockResponse:
        """Playback a recorded response."""
        if self._index >= len(self._recordings):
            raise RuntimeError("No more recorded responses")

        recording = self._recordings[self._index]
        self._index += 1

        return MockResponse(
            content=recording["response"]["content"],
            tool_calls=recording["response"].get("tool_calls", []),
            stop_reason=recording["response"].get("stop_reason", "end_turn"),
        )


# === Test Fixtures ===


@dataclass
class TestFixture:
    """Test fixture for agent testing."""

    name: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    expected_tool_calls: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "expected_tool_calls": self.expected_tool_calls,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }


class FixtureBuilder:
    """
    Builder for test fixtures.

    Usage:
        fixture = (FixtureBuilder("test_search")
            .with_input({"query": "test"})
            .expect_tool_call("web_search")
            .expect_output({"results": []})
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._input: dict[str, Any] = {}
        self._expected_output: dict[str, Any] | None = None
        self._expected_tools: list[str] = []
        self._timeout = 30.0
        self._metadata: dict[str, Any] = {}

    def with_input(self, data: dict[str, Any]) -> "FixtureBuilder":
        """Set input data."""
        self._input = data
        return self

    def expect_output(self, output: dict[str, Any]) -> "FixtureBuilder":
        """Set expected output."""
        self._expected_output = output
        return self

    def expect_tool_call(self, tool_name: str) -> "FixtureBuilder":
        """Expect a tool to be called."""
        self._expected_tools.append(tool_name)
        return self

    def with_timeout(self, seconds: float) -> "FixtureBuilder":
        """Set timeout."""
        self._timeout = seconds
        return self

    def with_metadata(self, **kwargs) -> "FixtureBuilder":
        """Add metadata."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> TestFixture:
        """Build the fixture."""
        return TestFixture(
            name=self._name,
            input_data=self._input,
            expected_output=self._expected_output,
            expected_tool_calls=self._expected_tools,
            timeout_seconds=self._timeout,
            metadata=self._metadata,
        )


# === Assertions ===


class AgentAssertions:
    """
    Assertion helpers for agent testing.

    Usage:
        assertions = AgentAssertions(result)
        assertions.assert_success()
        assertions.assert_tool_called("web_search")
        assertions.assert_output_contains("result")
    """

    def __init__(self, result: dict[str, Any]):
        self._result = result

    def assert_success(self) -> None:
        """Assert execution was successful."""
        status = self._result.get("status")
        if status != "success":
            raise AssertionError(f"Expected success, got {status}: {self._result.get('error')}")

    def assert_failed(self) -> None:
        """Assert execution failed."""
        status = self._result.get("status")
        if status != "failed":
            raise AssertionError(f"Expected failure, got {status}")

    def assert_tool_called(self, tool_name: str) -> None:
        """Assert a tool was called."""
        tools = self._result.get("tools_called", [])
        if tool_name not in tools:
            raise AssertionError(f"Tool {tool_name} was not called. Called: {tools}")

    def assert_tool_not_called(self, tool_name: str) -> None:
        """Assert a tool was not called."""
        tools = self._result.get("tools_called", [])
        if tool_name in tools:
            raise AssertionError(f"Tool {tool_name} should not have been called")

    def assert_output_contains(self, key: str) -> None:
        """Assert output contains key."""
        output = self._result.get("output", {})
        if key not in output:
            raise AssertionError(f"Output missing key: {key}")

    def assert_output_equals(self, expected: dict[str, Any]) -> None:
        """Assert output equals expected."""
        output = self._result.get("output", {})
        if output != expected:
            raise AssertionError(f"Output mismatch:\nExpected: {expected}\nGot: {output}")

    def assert_iterations_less_than(self, max_iterations: int) -> None:
        """Assert iteration count is below limit."""
        iterations = self._result.get("iterations", 0)
        if iterations >= max_iterations:
            raise AssertionError(f"Too many iterations: {iterations} >= {max_iterations}")


# === Test Runner ===


@dataclass
class TestResult:
    """Result of a single test."""

    fixture_name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    output: dict[str, Any] = field(default_factory=dict)


class AgentTestRunner:
    """
    Test runner for agent tests.

    Usage:
        runner = AgentTestRunner(agent)
        runner.add_fixture(fixture)
        results = runner.run_all()

        for result in results:
            print(f"{result.fixture_name}: {'PASS' if result.passed else 'FAIL'}")
    """

    def __init__(self, agent: Any):
        self._agent = agent
        self._fixtures: list[TestFixture] = []
        self._before_each: Callable | None = None
        self._after_each: Callable | None = None

    def add_fixture(self, fixture: TestFixture) -> None:
        """Add a test fixture."""
        self._fixtures.append(fixture)

    def add_fixtures(self, fixtures: list[TestFixture]) -> None:
        """Add multiple fixtures."""
        self._fixtures.extend(fixtures)

    def before_each(self, func: Callable) -> None:
        """Set function to run before each test."""
        self._before_each = func

    def after_each(self, func: Callable) -> None:
        """Set function to run after each test."""
        self._after_each = func

    def run_all(self) -> list[TestResult]:
        """Run all fixtures."""
        results = []
        for fixture in self._fixtures:
            result = self.run_fixture(fixture)
            results.append(result)
        return results

    def run_fixture(self, fixture: TestFixture) -> TestResult:
        """Run a single fixture."""
        if self._before_each:
            self._before_each()

        start = time.time()
        try:
            output = self._run_agent(fixture)
            duration = (time.time() - start) * 1000

            # Validate
            passed = self._validate(fixture, output)

            result = TestResult(
                fixture_name=fixture.name,
                passed=passed,
                duration_ms=duration,
                output=output,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = TestResult(
                fixture_name=fixture.name,
                passed=False,
                duration_ms=duration,
                error=str(e),
            )

        if self._after_each:
            self._after_each()

        return result

    def _run_agent(self, fixture: TestFixture) -> dict[str, Any]:
        """Run the agent with fixture input."""
        if hasattr(self._agent, "run"):
            return self._agent.run(fixture.input_data)
        if hasattr(self._agent, "execute"):
            return self._agent.execute(fixture.input_data)
        raise ValueError("Agent has no run or execute method")

    def _validate(self, fixture: TestFixture, output: dict[str, Any]) -> bool:
        """Validate output against fixture expectations."""
        if fixture.expected_output:
            if output != fixture.expected_output:
                return False

        if fixture.expected_tool_calls:
            tools_called = output.get("tools_called", [])
            for tool in fixture.expected_tool_calls:
                if tool not in tools_called:
                    return False

        return True

    def print_results(self, results: list[TestResult]) -> None:
        """Print test results."""
        passed = sum(1 for r in results if r.passed)

        print(f"\n{'=' * 50}")
        print(f"Test Results: {passed}/{len(results)} passed")
        print(f"{'=' * 50}\n")

        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.fixture_name} ({result.duration_ms:.0f}ms)")
            if result.error:
                print(f"   Error: {result.error}")
