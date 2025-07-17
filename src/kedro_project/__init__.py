"""kedro-project"""

__version__ = "0.1"

# Patch kedro.pipeline.node.Node.__init__ to handle empty inputs/outputs
import functools
from typing import Any, Callable, Dict, Iterable, Optional, Union

from kedro.pipeline.node import Node


def _patch_node_init() -> None:
    """Patch Node.__init__ to inject dummy output when both inputs and outputs are empty."""
    original_init = Node.__init__

    @functools.wraps(original_init)
    def patched_init(
        self: Node,
        func: Callable[..., Any],
        inputs: Union[None, str, Dict[str, str], Iterable[str]] = None,
        outputs: Union[None, str, Dict[str, str], Iterable[str]] = None,
        *,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> None:
        # If both inputs and outputs are empty/falsy, inject dummy output
        if not inputs and not outputs:
            outputs = "__dummy__"

        # Call original init with potentially modified outputs
        original_init(
            self,
            func=func,
            inputs=inputs,
            outputs=outputs,
            name=name,
            tags=tags,
            **kwargs,
        )

    # Apply the patch
    Node.__init__ = patched_init


# Apply the patch immediately when package is imported
_patch_node_init()
