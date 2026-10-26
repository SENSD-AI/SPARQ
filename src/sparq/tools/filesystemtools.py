from typing import Literal


def filesystemtools(working_dir, selected_tools: list | Literal['all']= 'all'):
    from langchain_community.agent_toolkits import FileManagementToolkit


    TOOLS = [
    'write_file',
    'read_file',
    'copy_file',
    'file_search',
    'list_directory',
    ]
    
    if selected_tools == 'all':
        selected_tools = TOOLS
    
    # If selected_tools is not a list, convert to a list        
    if not isinstance(selected_tools, list):
        selected_tools=[selected_tools]

    # check tools
    for tool in selected_tools:
        if tool not in TOOLS:
            raise ValueError(f"The tool, {tool} is not supported.\n"
                             f"Please select from {TOOLS}"
                             )
    
    tools = FileManagementToolkit(
        root_dir=working_dir,
        selected_tools=selected_tools
    ).get_tools()
    
    return tools
