from __future__ import annotations

import json
import sys
from types import FunctionType
from typing import Final, Protocol, cast

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.rust_bridge import get_native_bridge
from litellm.rust_bridge import ocr as rust_ocr_bridge
from tests.sdk_function_trace import TraceScenario, assert_function_trace_parity
from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider
from tests.sdk_function_trace.profiler import profile_python

MODEL: Final = "mistral-ocr-latest"
DOCUMENT: Final[dict[str, str]] = {
    "type": "document_url",
    "document_url": "https://example.com/document.pdf",
}
OPTIONAL_PARAMS: Final[dict[str, object]] = {"pages": [0], "unsupported": True}
RESPONSE_DATA: Final[dict[str, object]] = {
    "pages": [{"index": 0, "markdown": "hello"}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1},
}
PROVIDER_RESPONSE: Final = MockProviderResponse(
    status_code=200,
    headers=(("content-type", "application/json"),),
    body=json.dumps(RESPONSE_DATA).encode(),
)
_TRACE_RESPONSE: Final = TypeAdapter(dict[str, object])
_TRACE_EVENTS: Final = TypeAdapter(list[dict[str, object]])


class _NativeTraceOcr(Protocol):
    def __call__(
        self,
        *,
        model: str,
        document: dict[str, str],
        api_key: str,
        api_base: str,
        optional_params: dict[str, object],
        trace: bool,
    ) -> object: ...


def _invoke_python() -> object:
    previous_enabled: Final = rust_ocr_bridge.rust_ocr_enabled()
    rust_ocr_bridge.use_litellm_rust(False)
    try:
        with mock_provider(PROVIDER_RESPONSE) as api_base:
            return litellm.ocr(
                model=f"mistral/{MODEL}",
                document=DOCUMENT,
                api_key="test-key",
                api_base=api_base,
                pages=[0],
                unsupported=True,
            )
    finally:
        rust_ocr_bridge.use_litellm_rust(previous_enabled)


def _invoke_rust() -> tuple[str, ...]:
    bridge: Final = get_native_bridge()
    if bridge is None:
        raise AssertionError("The native Rust bridge is required for function-trace parity")
    trace_ocr: Final = cast(_NativeTraceOcr, bridge.ocr)
    with mock_provider(PROVIDER_RESPONSE) as api_base:
        raw_result: Final = trace_ocr(
            model=f"mistral/{MODEL}",
            document=DOCUMENT,
            api_key="test-key",
            api_base=api_base,
            optional_params=OPTIONAL_PARAMS,
            trace=True,
        )
    result: Final = _TRACE_RESPONSE.validate_python(raw_result)
    events: Final = _TRACE_EVENTS.validate_python(result.get("trace"))
    return tuple(cast(str, event["function"]) for event in events)


def test_mistral_ocr_transformation_function_trace_parity() -> None:
    assert_function_trace_parity(
        TraceScenario(
            functions=cast(
                tuple[FunctionType, ...],
                (
                    MistralOCRConfig.get_supported_ocr_params,
                    MistralOCRConfig.map_ocr_params,
                    MistralOCRConfig.get_supported_ocr_params,
                    MistralOCRConfig.transform_ocr_request,
                    MistralOCRConfig.transform_ocr_response_data,
                ),
            ),
            invoke_python=_invoke_python,
            invoke_rust=_invoke_rust,
        )
    )


class First:
    @staticmethod
    def run() -> None:
        return None


class Second:
    @staticmethod
    def run() -> None:
        return None


def test_profiler_matches_code_objects_and_keeps_repeated_calls() -> None:
    with profile_python((First.run,)) as profiler:
        Second.run()
        First.run()
        First.run()

    assert profiler.events == ["run", "run"]


def test_profiler_restores_previous_profiler_after_failure() -> None:
    previous: Final = sys.getprofile()

    with pytest.raises(RuntimeError, match="stop"):
        with profile_python((First.run,)):
            raise RuntimeError("stop")

    assert sys.getprofile() is previous
