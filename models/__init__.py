from .M3DFEL import M3DFEL
from .R3D50 import R3D_50

def create_model(args):
    """create model according to args

    Args:
        args
    """
    model = M3DFEL(args)
    # model = R3D_50()
    return model
