import os

import boto3
from langchain_aws import ChatBedrockConverse

from sparq.settings_old import Settings

MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

settings = Settings()
settings._load_env_variables()

profile = os.getenv('AWS_PROFILE', 'sensd')

# create a session
session = boto3.Session(profile_name=profile)

# create client
client = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)


llm = ChatBedrockConverse(
    model=MODEL_ID,
    client=client,
)

response = llm.invoke("What is salmonella?")
print(response.content)