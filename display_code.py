
# import display code packages
import pyan
from graphviz import Source

# import base packages
import os
import fnmatch
from typing import List



# Function that creates a call graph of this project
def create_call_graph() -> None:
    # get all python files in the current folder
    current_folder = get_current_folder()
    python_files = find_python_files(current_folder)

    # create call graph
    graph = pyan.create_callgraph(filenames=python_files, draw_defines=False, grouped_alt=False)

    # create the graphviz graph
    dot_graph = Source(graph)

    # display the graph
    dot_graph.render('call_graph', format='png')

# A function that finds all python files in a folder
def find_python_files(path: str = '.') -> List[str]:
    python_files: List[str] = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if fnmatch.fnmatch(file, '*.py') and not fnmatch.fnmatch(file, 'test_*'):
                python_files.append(os.path.join(root, file))

    return python_files

# A function that gets the folder of the current file
def get_current_folder() -> str:
    current_folder = os.path.dirname(os.path.realpath(__file__))

    return current_folder



# This file can be run as a script
if __name__ == '__main__':
    create_call_graph()