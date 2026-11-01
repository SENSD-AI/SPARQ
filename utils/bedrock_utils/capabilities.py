"""Small model-interface probes without datasets, REPLs, vision or reports."""

import asyncio
import json
import secrets
import time
import traceback
from importlib.metadata import version
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from sparq.schemas.planning import Plan, Router, StepResult
from sparq.utils.get_llm import get_llm
from utils.bedrock_utils.audit import (
    CheckResult,
    ProbeJob,
    RunResult,
    atomic_json,
    failure_result,
)

CHECKS = (
    "structured_router",
    "structured_plan",
    "tool_round_trip",
    "tools_with_structure",
)


def classify_error(error: Exception) -> CheckResult:
    """Distinguish declared API limitations, permissions and inconclusive errors."""
    result = failure_result(error)
    reason = " ".join(result.error_chain).lower()
    if isinstance(error, TimeoutError) or result.error_code in {
        "ReadTimeoutError",
        "ConnectTimeoutError",
    }:
        result.status = "timed_out"
    elif result.error_code in {
        "AccessDeniedException",
        "UnauthorizedException",
        "UnrecognizedClientException",
        "ExpiredTokenException",
    }:
        result.status = "access_denied"
    elif isinstance(error, NotImplementedError) or any(
        phrase in reason
        for phrase in (
            "does not support",
            "does not currently support",
            "doesn't support",
            "not supported",
            "unsupported",
        )
    ):
        result.status = "unsupported"
    else:
        result.status = "error"
    return CheckResult.model_validate(result.model_dump())


def make_lookup() -> tuple[BaseTool, list[dict[str, str]]]:
    """Return a fresh value only at tool execution, preventing answer-by-guessing."""
    calls: list[dict[str, str]] = []

    @tool
    def lookup_probe_value(key: str) -> str:
        """Look up the opaque value for key 'probe'; never infer this value."""
        if key != "probe":
            raise ValueError("The only valid lookup key is 'probe'.")
        value = secrets.token_hex(12)
        calls.append({"key": key, "value": value})
        return value

    return lookup_probe_value, calls


async def probe_router(model: BaseChatModel, evidence: dict[str, Any]) -> bool:
    result = await model.with_structured_output(Router).ainvoke(
        [
            SystemMessage(
                content="Route complex analytical questions to planning with route=True; answer trivial questions with route=False."
            ),
            HumanMessage(
                content="Analyze Salmonella trends across counties using datasets, statistical analysis and a plot."
            ),
        ]
    )
    evidence["response"] = (
        result.model_dump() if isinstance(result, Router) else str(result)
    )
    return isinstance(result, Router) and result.route is True


async def probe_plan(model: BaseChatModel, evidence: dict[str, Any]) -> bool:
    result = await model.with_structured_output(Plan).ainvoke(
        [
            SystemMessage(
                content="Return a structured two-step plan. No tools need executing."
            ),
            HumanMessage(
                content="Use exactly step 1 to look up a measurement and step 2 to report it. Step 2 depends on step 1. Use IDs 1 and 2, datasets=[] for both, and supply the schema's descriptions, rationales and task types."
            ),
        ]
    )
    evidence["response"] = (
        result.model_dump() if isinstance(result, Plan) else str(result)
    )
    return (
        isinstance(result, Plan)
        and len(result.steps) == 2
        and [step.id for step in result.steps] == [1, 2]
        and result.steps[0].dependencies == []
        and result.steps[1].dependencies == [1]
    )


async def probe_tool(model: BaseChatModel, evidence: dict[str, Any]) -> bool:
    lookup, calls = make_lookup()
    evidence["executed_tools"] = calls
    bound = model.bind_tools([lookup])
    messages = [
        SystemMessage(
            content="Use the provided tool to retrieve the opaque value; never invent it."
        ),
        HumanMessage(
            content="Call lookup_probe_value with key='probe', then return the value it gives you."
        ),
    ]
    request = await bound.ainvoke(messages)
    evidence["request"] = request.model_dump(mode="json")
    if not isinstance(request, AIMessage) or len(request.tool_calls) != 1:
        return False
    call = request.tool_calls[0]
    if (
        call["name"] != lookup.name
        or call["args"] != {"key": "probe"}
        or not call.get("id")
    ):
        return False
    output = await lookup.ainvoke(call)
    response = await bound.ainvoke([*messages, request, output])
    evidence["response"] = response.model_dump(mode="json")
    return (
        isinstance(response, AIMessage)
        and not response.tool_calls
        and calls[0]["value"] in response.text
    )


async def probe_combined(model: BaseChatModel, evidence: dict[str, Any]) -> bool:
    lookup, calls = make_lookup()
    evidence["executed_tools"] = calls
    agent = create_agent(model=model, tools=[lookup], response_format=StepResult)
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Execute step 1: call lookup_probe_value(key='probe'). Then return StepResult with id=1, success=True, execution_results containing the exact returned value, a description in step, and files_generated=[]. Do not invent the value.",
                }
            ]
        },
        config={"recursion_limit": 8},
    )
    result = response.get("structured_response")
    evidence["response"] = (
        result.model_dump() if isinstance(result, StepResult) else str(result)
    )
    return (
        isinstance(result, StepResult)
        and result.id == 1
        and result.success
        and not result.files_generated
        and bool(calls)
        and calls[-1]["value"] in result.execution_results
    )


async def execute_job(job: ProbeJob) -> RunResult:
    result = RunResult(checks={name: CheckResult() for name in CHECKS})
    evidence: dict[str, Any] = {}

    def checkpoint() -> None:
        atomic_json(job.artifact_dir / "check.json", result)
        (job.artifact_dir / "capability_evidence.json").write_text(
            json.dumps(evidence, indent=2)
        )

    checkpoint()
    try:
        model = get_llm(model=job.model_id, provider="aws_bedrock")
    except Exception as error:  # noqa: BLE001 -- each candidate must retain initialization failures
        check = classify_error(error)
        result.status, result.reason, result.error_code = (
            check.status,
            check.reason,
            check.error_code,
        )
        result.error_chain = check.error_chain
        for pending in result.checks.values():
            pending.status = "skipped"
            pending.reason = (
                "Model initialization failed; this check was not attempted."
            )
        checkpoint()
        return result
    profile = model.profile or {}
    result.capability_hints = {
        key: profile.get(key) if isinstance(profile.get(key), bool) else None
        for key in ("tool_calling", "structured_output", "image_inputs")
    }
    result.hints_source = f"langchain-aws {version('langchain-aws')} model.profile; installed metadata, not an AWS catalog guarantee"
    checkpoint()
    probes = (probe_router, probe_plan, probe_tool, probe_combined)
    for name, probe in zip(CHECKS, probes, strict=True):
        evidence[name] = {}
        result.active_check = name
        checkpoint()
        started = time.monotonic()
        try:
            passed = await asyncio.wait_for(
                probe(model, evidence[name]), timeout=job.check_timeout
            )
            check = CheckResult(
                status="passed" if passed else "failed",
                reason="Required interface and returned values verified."
                if passed
                else "Response did not satisfy the probe; inspect capability_evidence.json.",
            )
        except Exception as error:  # noqa: BLE001 -- record each capability independently
            traceback.print_exc()
            check = classify_error(error)
            if check.status == "timed_out":
                check.reason = f"No conclusive result within {job.check_timeout}s."
        check.seconds = round(time.monotonic() - started, 3)
        result.checks[name] = check
        result.active_check = None
        checkpoint()
        if check.status == "access_denied":
            for pending in result.checks.values():
                if pending.status == "not_tested":
                    pending.status = "skipped"
                    pending.reason = (
                        "Invocation was denied; remaining checks were not attempted."
                    )
            break
    unsuccessful = [
        check for check in result.checks.values() if check.status != "passed"
    ]
    if unsuccessful:
        first = next(
            (check for check in unsuccessful if check.status != "skipped"),
            unsuccessful[0],
        )
        result.status, result.reason, result.error_code = (
            first.status,
            first.reason,
            first.error_code,
        )
        result.error_chain = first.error_chain
    else:
        result.status = "passed"
        result.reason = "All four SPARQ model-interface probes passed; full-pipeline reliability is untested."
    checkpoint()
    return result
