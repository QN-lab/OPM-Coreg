# How to create dependency graph for the repository

- pip install pydeps
- pydeps --show-dot --no-show src > docs/src_dependency_graph.dot
- On Windows, if GraphViz is not installed, use `Graphviz Interactive Preview` VSCode's extension to visualize .dot files.
- Save src_dependency_graph.svg in the docs folder.