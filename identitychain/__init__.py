# Authors: marcusm117
# License: Apache 2.0


from .executor import unsafe_execute
from .identity_chain import IdentityChain, IdentityChainError


__version__ = "0.0.1"
__all__ = ["unsafe_execute", "IdentityChain", "IdentityChainError"]
