"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    from functools import reduce

    pipelines = find_pipelines()
    pipelines["__default__"] = reduce(
        lambda a, b: a + b, pipelines.values(), Pipeline([])
    )
    return pipelines
