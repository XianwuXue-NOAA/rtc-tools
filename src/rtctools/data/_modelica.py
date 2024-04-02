"""Module for reading/writing Modelica files."""
import pickle
from pathlib import Path

import casadi as ca
import pymoca
import pymoca.backends.casadi.api
from pymoca.backends.casadi.model import Model


def _get_cached_versions(model_folder: Path, model_name: str) -> dict:
    """Get cached versions of rtc-tools dependencies."""
    model_folder = Path(model_folder).resolve()
    versions_cache_file = model_folder / f"{model_name}.versions_cache"
    cached_versions = {}
    if versions_cache_file.is_file():
        with open(versions_cache_file, "r") as file:
            cached_versions = pickle.load(file)
    return cached_versions


def _get_current_versions() -> dict:
    """Get current versions of rtc-tools dependencies."""
    casadi_version: str = ca.__version__
    casadi_version.split(".")
    casadi_version = f"{casadi_version[0]}.{casadi_version[1]}"
    versions = {"casadi": casadi_version}
    return versions


def _save_versions(model_folder: Path, model_name: str, versions: dict):
    """Save versions of rtc-tools dependencies"""
    model_folder = Path(model_folder).resolve()
    versions_cache_file = model_folder / f"{model_name}.versions_cache"
    with open(versions_cache_file, "w") as file:
        pickle.dump(versions, file)


def _check_pymoca_cache_file_is_valid(cached_versions: dict, current_versions: dict) -> bool:
    """Check if the current pymoca cache file is valid."""
    if not cached_versions:
        return False
    if current_versions["casadi"] != cached_versions["casadi"]:
        return False
    return True


def read_model_with_pymoca(model_folder: Path, model_name: str, compiler_options: dict) -> Model:
    """Read a modelica file with pymoca."""
    cached_versions = _get_cached_versions(model_folder, model_name)
    current_versions = _get_current_versions()
    pymoca_cache_is_valid = _check_pymoca_cache_file_is_valid(cached_versions, current_versions)
    if not pymoca_cache_is_valid:
        compiler_options["cache"] = False
    pymoca_model = pymoca.backends.casadi.api.transfer_model(
        model_folder, model_name, compiler_options
    )
    _save_versions(model_folder, model_name, cached_versions)
    return pymoca_model
