
from sparq.settings import ENVSettings
from sparq.utils import helpers
import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph application.")
    parser.add_argument('-t', '--test', action='store_true', help="Run in test mode with a predefined query.")
    parser.add_argument('-a', '--architecture', default='v1', choices=['v1'], help="Architecture to use.")
    args = parser.parse_args()

    _ = ENVSettings(verbose=True)

    if args.architecture == 'v1':
        from sparq.architectures.v1.system import Agentic_system

    agentic_system = Agentic_system(verbose=True)
    user_query = helpers.get_user_query(args=args, config={"test_query": agentic_system.settings.test_query})
    asyncio.run(agentic_system.run(user_query=user_query))

if __name__ == "__main__":
    main()
