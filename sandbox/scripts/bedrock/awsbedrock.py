import os

import boto3

from sparq.settings import ENVSettings

ENVSettings()

# Uses the AWS_PROFILE environment variable
profile = os.getenv("AWS_PROFILE", "default")

# Create a session tied to that profile
session = boto3.Session(profile_name=profile)

# Create the Bedrock Runtime client
client = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define the model and message
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
messages = [{"role": "user", "content": [{"text": "What is salmonella?"}]}]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=messages,
)

# Print the response
print(response['output']['message']['content'][0]['text'])
