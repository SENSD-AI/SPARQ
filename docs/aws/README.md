This project uses AWS Bedrock models. In order to run SPARQ, you will need AWS API keys. 

# Prerequisites

- Install AWS CLI (see https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api.html)

# Important commands

For first time setup:
```bash
aws configure sso
```

Then to login after token expiration:
```bash
aws sso login --profile <aws_profile>
```

To verify your AWS credentials and ensure that you are logged in correctly, you can run the following command:
```bash
aws sts get-caller-identity --profile <aws_profile>
```

# Bedrock

To list all available foundation models in AWS Bedrock, you can run the following command:
```bash
aws bedrock list-foundation-models \
  --query 'modelSummaries[].{Model:modelId,Provider:providerName,Status:modelLifecycle.status,Input:join(`,`,inputModalities),Output:join(`,`,outputModalities),Inference:join(`,`,inferenceTypesSupported)}' \
  --output table \
  --profile sensd
```

To output the model list in JSON format, you can use:
```bash
aws bedrock list-foundation-models \
  --query 'modelSummaries[].{Model:modelId,Provider:providerName,Status:modelLifecycle.status,Input:join(`,`,inputModalities),Output:join(`,`,outputModalities),Inference:join(`,`,inferenceTypesSupported)}' \  
  --output json \
  --profile sensd > <output_path>.json
```
