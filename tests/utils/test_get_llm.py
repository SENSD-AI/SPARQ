"""Verify function-based generation options without model API calls."""

import copy
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from sparq.utils.get_llm import get_llm


class TestGetLLM(unittest.TestCase):
    def test_native_providers_receive_generation_options(self):
        for provider in ("openai", "google_genai"):
            with (
                self.subTest(provider=provider),
                patch("langchain.chat_models.init_chat_model") as init,
            ):
                output = get_llm(
                    "example", provider, max_tokens=8192, temperature=0, top_p=0.9
                )
                self.assertIs(output, init.return_value)
                init.assert_called_once_with(
                    model="example",
                    model_provider=provider,
                    max_tokens=8192,
                    temperature=0,
                    top_p=0.9,
                )

    def test_omitted_options_do_not_inject_factory_defaults(self):
        with patch("langchain.chat_models.init_chat_model") as init:
            get_llm()
        init.assert_called_once_with(
            model="gpt-4o",
            model_provider="openai",
        )

    def test_google_receives_top_k_including_zero(self):
        with patch("langchain.chat_models.init_chat_model") as init:
            get_llm("example", "google_genai", top_k=0)
        self.assertEqual(init.call_args.kwargs["top_k"], 0)

    def test_explicit_none_omits_generation_options(self):
        with patch("langchain.chat_models.init_chat_model") as init:
            get_llm(max_tokens=None, temperature=None, top_p=None, top_k=None)
        init.assert_called_once_with(
            model="gpt-4o",
            model_provider="openai",
        )

    def test_openai_rejects_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k is not supported"):
            get_llm("example", "openai", top_k=20)

    def test_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            get_llm("example", "missing")

    def test_openrouter_places_top_k_in_extra_body(self):
        with (
            patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "test-key",
                    "OPENROUTER_BASE_URL": "https://example.test/v1",
                },
            ),
            patch("langchain_openai.ChatOpenAI") as constructor,
        ):
            get_llm(
                "example",
                "openrouter",
                max_tokens=8192,
                temperature=0,
                top_p=0.9,
                top_k=40,
            )
        constructor.assert_called_once_with(
            api_key=SecretStr("test-key"),
            base_url="https://example.test/v1",
            model="example",
            max_tokens=8192,
            temperature=0,
            top_p=0.9,
            extra_body={"top_k": 40},
        )

    def test_bedrock_request_contains_generation_options(self):
        runtime = Mock()
        runtime.meta.region_name = "us-east-1"
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Stub"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 0},
        }
        # The response parser mutates its input, so each call needs a fresh copy.
        runtime.converse.side_effect = lambda **kwargs: copy.deepcopy(response)
        environment = {
            "AWS_PROFILE": "test",
            "AWS_REGION": "us-east-1",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
        }
        with (
            patch.dict(os.environ, environment),
            patch("boto3.Session") as session,
            patch(
                "langchain_aws.chat_models.bedrock_converse.create_aws_client",
                return_value=Mock(),
            ),
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Unexpected network call"),
            ),
        ):
            session.return_value.client.return_value = runtime
            for top_k in (None, 40):
                with self.subTest(top_k=top_k):
                    model = get_llm(
                        "us.anthropic.claude-sonnet-5",
                        "aws_bedrock",
                        max_tokens=8192,
                        temperature=0,
                        top_p=0.9,
                        top_k=top_k,
                    )
                    self.assertEqual(
                        model.invoke([HumanMessage(content="Hello")]).text, "Stub"
                    )
                    request = runtime.converse.call_args.kwargs
                    self.assertEqual(
                        request["inferenceConfig"],
                        {
                            "maxTokens": 8192,
                            "temperature": 0,
                            "topP": 0.9,
                        },
                    )
                    if top_k is None:
                        self.assertNotIn("additionalModelRequestFields", request)
                    else:
                        self.assertEqual(
                            request["additionalModelRequestFields"], {"top_k": 40}
                        )
