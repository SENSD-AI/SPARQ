"""
0th experiment.

Usage:
uv run experiments/00.py : Run SPARQ on all questions
uv run experiments/00.py k : Run SPARQ on first k questions
"""

import asyncio
import sys

from sparq.system import Agentic_system
from sparq.settings import ENVSettings, AgenticSystemSettings

QUESTIONS = [
    # "What is the most common food vehicle associated with salmonella outbreaks?",
    # "Which county in AZ had the highest food insecurity rate in 2022?",
    "What factors might contribute to the variation in outbreak sizes across different food vehicles?",
    "How does social vulnerability at the county level correlate with food insecurity and poverty and are there any notable differences in salmonella outbreak incidence between counties with high vs low social vulnerability and food insecurity?",
    "Can we identify specific socioeconomic factors that are strong predictors of increased Salmonella outbreaks in specific regions?",
    "What is the current salmonella risk level in Boone County, and what actions should I take?",
    "I am working on my plan of work for the upcoming year and trying to map out food safety trainings (for instance, I have three counties and would like to host trainings in each X amount of times a year). Are there any specific hotspots or specific locations that may be most beneficial for me to target?",
    "Which specific demographic groups in our service area face the highest combined risk from food insecurity and Salmonella exposure?",
    "How do detection pattern trends correlate with our county's social vulnerability indicators, and which intervention points offer the highest impact potential?",
    "Predict the likely impact (households, geography, etc.) of a food emergency such as avian\u2011influenza\u2011driven shortages.",
    "Can you predict any potential food crises by geography based on current funding levels, legislation, or other exogenous factors?",
    "Are some serotypes more dangerous than others? Which ones cause the most outbreaks?",
    "How different are the symptoms of various salmonella serotypes and which ones cause severe illnesses?",
]

def main():
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
            questions = QUESTIONS[:k]
        except ValueError:
            print(f"Invalid argument {sys.argv[1]}, expected an integer.")
            return
    else:
        questions = QUESTIONS

    ENVSettings()
    AgenticSystemSettings(verbose=True)
    
    agentic_system = Agentic_system()
    
    # Run system for all questions
    for question in questions:
        asyncio.run(agentic_system.run(question))
    
if __name__ == "__main__":
    main()
