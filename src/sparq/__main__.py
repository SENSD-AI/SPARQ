
import argparse
import asyncio

from sparq.architectures.v1.settings import V1Settings
from sparq.schemas.output_schemas import SystemOutput
from sparq.settings import ENVSettings
from sparq.utils import helpers

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph application.")
    parser.add_argument('-t', '--test', action='store_true', help="Run in test mode with a predefined query.")
    parser.add_argument('-a', '--architecture', default='v1', choices=['v1', 'v2'], help="Architecture to use.")
    args = parser.parse_args()

    _ = ENVSettings()

    if args.architecture == 'v1':
        from sparq.architectures.v1.system import Agentic_system
        settings = V1Settings()
    elif args.architecture == 'v2':
        # Implement later when v2 module is complete
        return

    # Display settings
    console = Console()
    formatted_settings = Pretty(
        settings,
        expand_all=True,
        indent_guides=False,
    )
    settings_panel = Panel(
        formatted_settings,
        title="SPARQ settings",
        border_style="blue"
    )
    console.print(settings_panel)

    agentic_system = Agentic_system()
    user_query = helpers.get_user_query(args=args, config={"test_query": agentic_system.settings.test_query})
    output = asyncio.run(agentic_system.run(user_query=user_query))

    console.print(
        Panel(
            Pretty(output, expand_all=True, indent_guides=False),
            title="SPARQ Results",
            border_style="green",
        )
    )

if __name__ == "__main__":
    main()
