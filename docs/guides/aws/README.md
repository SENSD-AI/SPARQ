# AWS Bedrock Setup

SPARQ supports AWS Bedrock as an LLM provider. Set `provider = "aws_bedrock"` for any node in `config_v1.toml` and follow this guide.

## Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
- An AWS account with Bedrock model access enabled for your region

## First-time setup

```bash
aws configure sso
```

Set the resulting profile name and region in your `.env`:

```
AWS_PROFILE=<your_profile>
AWS_REGION=us-east-1
```

## Logging in (after token expiry)

```bash
aws sso login --profile <your_profile>
```

## Verify credentials

```bash
aws sts get-caller-identity --profile <your_profile>
```

## Listing available models

List all foundation models with their inference types, modalities, and lifecycle status:

```bash
aws bedrock list-foundation-models \
  --query 'modelSummaries[].{Model:modelId,Provider:providerName,Status:modelLifecycle.status,Input:join(`,`,inputModalities),Output:join(`,`,outputModalities),Inference:join(`,`,inferenceTypesSupported)}' \
  --output table \
  --profile <your_profile>
```

Save as JSON:

```bash
aws bedrock list-foundation-models \
  --query 'modelSummaries[].{Model:modelId,Provider:providerName,Status:modelLifecycle.status,Input:join(`,`,inputModalities),Output:join(`,`,outputModalities),Inference:join(`,`,inferenceTypesSupported)}' \
  --output json \
  --profile <your_profile> > models.json
```

## Cross-region inference

For cross-region inference profiles (recommended for higher availability), use model IDs prefixed with a region code, e.g.:

```toml
model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
provider = "aws_bedrock"
```
