REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_local_controller import BasicLocalMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_local_mac"] = BasicLocalMAC