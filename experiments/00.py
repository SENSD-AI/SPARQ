"""
0th experiment.

Usage:
uv run experiments/00.py : Run SPARQ on all questions
uv run experiments/00.py k : Run SPARQ on first k questions
"""

import asyncio
import sys
import json

from pathlib import Path

from sparq.architectures.v1.system import Agentic_system
from sparq.settings import ENVSettings
from sparq.utils.get_package_dir import get_project_root

FILE_PATH = get_project_root() / 'data' / 'Q_dataset.json'
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

def load_data(file_path: str | Path) -> tuple[int, list[dict]]:
    """
    Args:
        File path
    Returns:
        (Number of Entries, Entries as dict)
    """

    with open(file_path) as f:
        data = json.load(f)

    questions = data.get('questions', [])
    if not questions:
        raise ValueError(f"No questions found in {file_path}")
    
    return len(questions), questions

def main():
    # n, questions = load_data(FILE_PATH)

    # print("="*100)
    # print(f"Number of questions: {n}")

    # for i, q in enumerate(questions, 1):
    #     print()
    #     print(i, q['text'])

    # print("="*100)
    # return

    # Make the root directory for results
    results_dir = Path.cwd().resolve() / "00_results"
    results_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Run system for all questions
    for i, question in enumerate(questions):
        # Make output dir for current question
        output_dir = results_dir / str(i)
        output_dir.mkdir(parents=True, exist_ok=True)

        agentic_system = Agentic_system()
        agentic_system.settings.paths.output_dir = output_dir # set the output dir
        agentic_system.settings.paths.set_run_dir() # recompute run_dir from the new output_dir # type: ignore

        asyncio.run(agentic_system.run(question))
    
if __name__ == "__main__":
    main()
