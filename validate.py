from importlib import metadata

installed = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
if "pandas" in installed:
    print("pandas is installed")