import asyncio

from sparq.settings import ENVSettings
from sparq.architectures.v1.system import Agentic_system

QUERY = "What datasets do we have and what do they represent?"

def main():
    ENVSettings()
    agentic_system = Agentic_system()
    asyncio.run(agentic_system.run(QUERY))

if __name__ == "__main__":
    main()
