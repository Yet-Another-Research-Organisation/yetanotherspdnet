from unittest import TestCase

import torch
from torch.testing import assert_close


import yetanotherspdnet.spd as spd
import yetanotherspdnet.means as means

seed = 777
torch.manual_seed(seed)

# "Fun" fact: float64 seems really important for most functions...
# float32 not that close result, seems true with scipy
# as well.....
torch.set_default_dtype(torch.float64)