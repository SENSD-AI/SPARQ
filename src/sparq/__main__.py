
from sparq.settings import ENVSettings, BaseAgenticSettings
from sparq.utils import helpers
from sparq.architectures.v1.system import Agentic_system
import argparse
import asyncio

def main():
    _ = ENVSettings(verbose=True)
    system_settings = BaseAgenticSettings(verbose=True)

    parser = argparse.ArgumentParser(description="Run the LangGraph application.")
    parser.add_argument('-t', '--test', action='store_true', help="Run in test mode with a predefined query.")
    args = parser.parse_args()

    user_query = helpers.get_user_query(args=args, config={"test_query": system_settings.test_query})

    agentic_system_instance = Agentic_system()
    asyncio.run(agentic_system_instance.run(user_query=user_query))

if __name__ == "__main__":
    main()
