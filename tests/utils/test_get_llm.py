import unittest
from unittest.mock import Mock, patch

from langchain_aws import ChatBedrockConverse

from sparq.settings import ENVSettings
from sparq.utils.get_llm import get_llm


class TestGetLLM(unittest.TestCase):
    def setUp(self) -> None:
        ENVSettings()  # loads env vars (AWS_PROFILE, AWS_REGION, etc.) into os.environ

    @patch('boto3.Session')
    def test_aws(self, mock_session):
        client = Mock()
        # Bedrock may infer its control-plane region from the runtime client.
        client.meta.region_name = "us-east-1"
        mock_session.return_value.client.return_value = client

        model = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        llm = get_llm(model=model, provider='aws_bedrock')
        self.assertIsInstance(llm, ChatBedrockConverse)
