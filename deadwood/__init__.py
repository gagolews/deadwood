"""
deadwood Python Package
"""


# ############################################################################ #
#                                                                              #
#   Copyleft (C) 2025-2026, Marek Gagolewski <https://www.gagolewski.com/>     #
#                                                                              #
#                                                                              #
#   This program is free software: you can redistribute it and/or modify       #
#   it under the terms of the GNU Affero General Public License                #
#   Version 3, 19 November 2007, published by the Free Software Foundation.    #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               #
#   GNU Affero General Public License Version 3 for more details.              #
#   You should have received a copy of the License along with this program.    #
#   If this is not the case, refer to <https://www.gnu.org/licenses/>.         #
#                                                                              #
# ############################################################################ #

# version string, e.g., "1.0.0.9001" or "1.1.1"
__version__ = "0.9.0"

from .core import *
from .deadwood import MSTBase, MSTClusterer, MSTOutlierDetector
from .deadwood import Deadwood
# from .deadwood import MSTClusterMixinWithProcessors
