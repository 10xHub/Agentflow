# """Helpers for loading A2A optional dependencies."""

# from __future__ import annotations

# from importlib import import_module
# from types import ModuleType
# from typing import Any


# A2A_EXTRA_INSTALL_HINT = (
#     "Install it with 'pip install 10xscale-agentflow[a2a_sdk]' " "or 'pip install a2a-sdk'."
# )


# def missing_a2a_sdk_error(feature: str, exc: BaseException) -> RuntimeError:
#     """Return a consistent error for A2A helpers when a2a-sdk is absent."""
#     return RuntimeError(
#         f"{feature} requires the optional 'a2a-sdk' package. {A2A_EXTRA_INSTALL_HINT}"
#     )


# def import_a2a_module(module_name: str, feature: str) -> ModuleType:
#     """Import an A2A SDK module with a helpful optional-dependency error."""
#     try:
#         return import_module(module_name)
#     except Exception as exc:
#         raise missing_a2a_sdk_error(feature, exc) from exc


# def get_a2a_attr(module_name: str, attr_name: str, feature: str) -> Any:
#     """Get an attribute from an A2A SDK module with a consistent error."""
#     module = import_a2a_module(module_name, feature)
#     try:
#         return getattr(module, attr_name)
#     except AttributeError as exc:
#         raise missing_a2a_sdk_error(feature, exc) from exc
