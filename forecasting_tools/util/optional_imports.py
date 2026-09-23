import importlib.util

_INSTALL_MESSAGE = (
    "'{pip_name}' is needed for this feature, but is not installed. "
    "It is an optional dependency so that the template bot and bot running "
    "infrastructure stay lightweight.\n"
    "Install it with:\n"
    "    pip install 'forecasting-tools[{extra}]'\n"
    "or, if you are developing on this repo:\n"
    "    poetry install --all-extras"
)


def missing_optional_package_error(pip_name: str, extra: str) -> ImportError:
    return ImportError(_INSTALL_MESSAGE.format(pip_name=pip_name, extra=extra))


def require_optional_package(module_name: str, pip_name: str, extra: str) -> None:
    try:
        found = importlib.util.find_spec(module_name) is not None
    except ImportError:
        found = False
    if not found:
        raise missing_optional_package_error(pip_name, extra)
