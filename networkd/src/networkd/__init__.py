# read version from installed package
from importlib.metadata import version
__version__ = version("networkd")

from .networkd import embed

embed_instance = embed()
embed = embed_instance.embed  # Directly expose a method
