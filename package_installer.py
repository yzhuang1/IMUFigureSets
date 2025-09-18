"""
Automatic Package Installer for GPT-Generated Code
Parses import statements and installs missing packages before training
"""

import ast
import re
import subprocess
import sys
import logging
from typing import List, Set
import importlib.util

logger = logging.getLogger(__name__)

def extract_imports_from_code(code: str) -> Set[str]:
    """
    Extract all import statements from Python code string

    Args:
        code: Python code as string

    Returns:
        Set of package names that need to be imported
    """
    imports = set()

    try:
        # Parse the code into AST
        tree = ast.parse(code)

        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import package, import package.submodule
                for alias in node.names:
                    package_name = alias.name.split('.')[0]
                    imports.add(package_name)

            elif isinstance(node, ast.ImportFrom):
                # Handle: from package import something
                if node.module:
                    package_name = node.module.split('.')[0]
                    imports.add(package_name)

    except SyntaxError as e:
        logger.warning(f"Could not parse code for imports due to syntax error: {e}")
        # Fallback to regex-based parsing
        imports.update(_extract_imports_regex(code))

    # Filter out built-in modules and common stdlib modules
    filtered_imports = _filter_builtin_packages(imports)

    logger.info(f"Extracted imports from code: {filtered_imports}")
    return filtered_imports

def _extract_imports_regex(code: str) -> Set[str]:
    """Fallback regex-based import extraction"""
    imports = set()

    # Match import statements
    import_pattern = r'^(?:from\s+(\w+(?:\.\w+)*)\s+import|import\s+(\w+(?:\.\w+)*))'

    for line in code.split('\n'):
        line = line.strip()
        match = re.match(import_pattern, line)
        if match:
            package = match.group(1) or match.group(2)
            if package:
                imports.add(package.split('.')[0])

    return imports

def _filter_builtin_packages(packages: Set[str]) -> Set[str]:
    """Filter out built-in and standard library packages"""
    builtin_packages = {
        # Built-ins
        'os', 'sys', 'time', 'json', 'math', 'random', 'collections',
        'itertools', 'functools', 'operator', 'copy', 'pickle', 'datetime',
        'pathlib', 'typing', 'dataclasses', 'enum', 'logging', 'warnings',
        're', 'string', 'io', 'traceback', 'inspect', 'gc', 'weakref',

        # Common stdlib that don't need installation
        'urllib', 'http', 'email', 'html', 'xml', 'csv', 'sqlite3',
        'threading', 'multiprocessing', 'concurrent', 'queue', 'asyncio',
        'unittest', 'argparse', 'configparser', 'tempfile', 'shutil',
        'glob', 'fnmatch', 'linecache', 'textwrap', 'unicodedata',
        'struct', 'codecs', 'base64', 'binascii', 'hashlib', 'hmac',
        'secrets', 'ssl', 'socket', 'ipaddress'
    }

    return packages - builtin_packages

def check_package_availability(packages: Set[str]) -> tuple[Set[str], Set[str]]:
    """
    Check which packages are available and which are missing

    Args:
        packages: Set of package names to check

    Returns:
        Tuple of (available_packages, missing_packages)
    """
    available = set()
    missing = set()

    for package in packages:
        try:
            # Try to find the package spec
            spec = importlib.util.find_spec(package)
            if spec is not None:
                available.add(package)
            else:
                missing.add(package)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.add(package)

    logger.info(f"Available packages: {available}")
    logger.info(f"Missing packages: {missing}")

    return available, missing

def install_packages(packages: Set[str]) -> tuple[Set[str], Set[str]]:
    """
    Install missing packages using pip

    Args:
        packages: Set of package names to install

    Returns:
        Tuple of (successfully_installed, failed_to_install)
    """
    if not packages:
        logger.info("No packages to install")
        return set(), set()

    successful = set()
    failed = set()

    logger.info(f"Installing {len(packages)} packages: {packages}")

    for package in packages:
        try:
            logger.info(f"Installing package: {package}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )

            if result.returncode == 0:
                successful.add(package)
                logger.info(f"✅ Successfully installed: {package}")
            else:
                failed.add(package)
                logger.error(f"❌ Failed to install {package}: {result.stderr}")

        except subprocess.TimeoutExpired:
            failed.add(package)
            logger.error(f"❌ Installation timeout for package: {package}")
        except Exception as e:
            failed.add(package)
            logger.error(f"❌ Installation error for {package}: {e}")

    if successful:
        logger.info(f"Successfully installed {len(successful)} packages: {successful}")
    if failed:
        logger.warning(f"Failed to install {len(failed)} packages: {failed}")

    return successful, failed

def install_gpt_code_dependencies(training_code: str, raise_on_failure: bool = True) -> bool:
    """
    Main function: Extract imports from GPT code and install missing packages

    Args:
        training_code: GPT-generated training function code
        raise_on_failure: Whether to raise exception if any package fails to install

    Returns:
        True if all packages were installed successfully, False otherwise

    Raises:
        RuntimeError: If raise_on_failure=True and any package fails to install
    """
    logger.info("🔍 Analyzing GPT-generated code for package dependencies...")

    # Extract imports
    required_packages = extract_imports_from_code(training_code)

    if not required_packages:
        logger.info("✅ No external packages required")
        return True

    # Check availability
    available, missing = check_package_availability(required_packages)

    if not missing:
        logger.info("✅ All required packages are already available")
        return True

    # Install missing packages
    logger.info(f"📦 Installing {len(missing)} missing packages...")
    successful, failed = install_packages(missing)

    # Report results
    if not failed:
        logger.info("✅ All packages installed successfully")
        return True
    else:
        error_msg = f"❌ Failed to install packages: {failed}"
        logger.error(error_msg)

        if raise_on_failure:
            raise RuntimeError(f"Package installation failed: {failed}")

        return False

# Convenience function for integration
def ensure_gpt_dependencies(training_code: str) -> bool:
    """
    Ensure all dependencies for GPT-generated training code are available

    Args:
        training_code: The training function code from GPT

    Returns:
        True if all dependencies are satisfied
    """
    return install_gpt_code_dependencies(training_code, raise_on_failure=False)