from pydantic import BaseModel, Field
from typing import List

# Define desired output structure
class Step(BaseModel):
    """Information about a step"""
    step_description: str = Field(..., description="Description of the analytical step")
    datasets: List[str] = Field(..., description="List of dataset names used")
    rationale: str = Field(..., description="Why this step is necessary")
    task_type: List[str] = Field(..., description="The type of computation required e.g. data_retrieval, correlation, visualization")
    
class Plan(BaseModel):
    """Information about the the steps in a plan to answer the user query"""
    steps: List[Step]
    wants: str = Field(..., description="Further information you need to make a better plan")
    misc: str = Field(..., description="Anything else you want the user to know or just a general scratchpad")

    def pretty_print(self):
        for i, step in enumerate(self.steps):
            print(f"Step {i}")
            print(f"Description: {step.step_description}")
            print(f"Datasets: {step.datasets}")
            print(f"Rationale: {step.rationale}")
            print(f"Tast Type: {step.task_type}")
            print()
        
        print("Wants:")
        print(self.wants)
        print("Misc:")
        print(self.misc)
        
class Router(BaseModel):
    """Output of the router node"""
    route: bool = Field(..., description="Whether the query requires further planning (True) or can be answered directly (False)")
    answer: str | None = Field(..., description="The answer to the query if it can be answered directly")
    
class ExecutorOutput(BaseModel):
    """Output of the executor node"""
    step: str = Field(..., description="What you were tasked to do by the user")
    execution_results: str = Field("", description="Summary of results of running your code.")
    files_generated: List[str] = Field(default_factory=list, description="Files generated during execution")
    misc: str = Field("", description="Anything else you want to note, e.g. caveats, observations, or next steps")
