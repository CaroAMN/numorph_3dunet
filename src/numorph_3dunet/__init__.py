from .version import __version__, VERSION_STRING

def get_version():
    """Return the package version."""
    return __version__

def show_version():
    """Print full version info."""
    print(VERSION_STRING)