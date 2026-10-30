"""
Lists the available Amazon Bedrock models.
"""
import logging
import os
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

from sparq.settings import ENVSettings
from sparq.utils.helpers import render_records_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_AWS_REGION = "us-east-1"


def list_foundation_models(bedrock_client, short: bool = False) -> List[Dict]:
    """
    Gets a list of available Amazon Bedrock foundation models and returns
    the raw list of model summary dicts.

    :return: A list of model summary dicts as returned by Bedrock.
    """

    try:
        response = bedrock_client.list_foundation_models()
        models: List[Dict] = response["modelSummaries"]
        logger.info("Got %s foundation models.", len(models))
        return models

    except ClientError:
        logger.error("Couldn't list foundation models.")
        raise


def main():
    """Entry point for the example. Uses the AWS SDK for Python (Boto3)
    to create an Amazon Bedrock client. Then lists the available Bedrock models
    in the region set in the callers profile and credentials.
    """

    # load AWS profile from environment variable AWS_PROFILE
    _ = ENVSettings()
    profile = os.getenv("AWS_PROFILE", "default")
    session = boto3.Session(profile_name=profile)

    bedrock_client = session.client(
        service_name="bedrock",
        region_name=DEFAULT_AWS_REGION
    )

    fm_models = list_foundation_models(bedrock_client)
    table_models = [
        {**model, "status": model.get("modelLifecycle", {}).get("status", "")}
        for model in fm_models
    ]

    print("Available Bedrock Foundation Models:")
    # Use the renderer to print the table to stdout
    render_records_table(
        table_models,
        columns=[
            "modelName",
            "modelId",
            "inferenceTypesSupported",
            "inputModalities",
            "outputModalities",
            "providerName",
            "status",
        ],
        title="Foundation Models",
        no_wrap_columns=["modelId"],
    )

    # for model in fm_models:
    #     print(f"Model: {model['modelName']}")
    #     print(json.dumps(model, indent=2))
    #     print("---------------------------\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()

