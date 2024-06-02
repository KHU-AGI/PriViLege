from models.coil import COIL
from models.der import DER
from models.ewc import EWC
from models.finetune import Finetune
from models.foster import FOSTER
from models.gem import GEM
from models.icarl import iCaRL
from models.lwf import LwF
from models.replay import Replay
from models.bic import BiC
from models.podnet import PODNet
from models.rmm import RMM_FOSTER, RMM_iCaRL
from models.ssre import SSRE
from models.wa import WA
from models.fetril import FeTrIL 
from models.pa2s import PASS
from models.il2a import IL2A

from models.L2P import L2P
from models.DP import DP
from models.Ours import Ours
from models.CODA import CODA
from models.Linear_Probing import LinearProbing

def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "bic":
        return BiC(args)
    elif name == "podnet":
        return PODNet(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "wa":
        return WA(args)
    elif name == "der":
        return DER(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "replay":
        return Replay(args)
    elif name == "gem":
        return GEM(args)
    elif name == "coil":
        return COIL(args)
    elif name == "foster":
        return FOSTER(args)
    elif name == "rmm-icarl":
        return RMM_iCaRL(args)
    elif name == "rmm-foster":
        return RMM_FOSTER(args)
    elif name == "fetril":
        return FeTrIL(args)
    elif name == "pass":
        return PASS(args)
    elif name == "il2a":
        return IL2A(args)
    elif name == "ssre":
        return SSRE(args)
    elif name == "l2p":
        return L2P(args)
    elif name == "dp":
        return DP(args)
    elif name == "lp":
        return LinearProbing(args)
    elif name == "coda":
        return CODA(args)
    elif name == "ours":
        return Ours(args)
    else:
        assert 0
