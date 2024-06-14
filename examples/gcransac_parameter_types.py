from enum import Enum

class Solver(Enum):
    PointBased = 0
    SIFTBased = 1
    AffineBased = 2
    TwoScaOriBased = 3
    OneSIFTBased = 4

class Sampler(Enum):
    Uniform = 0
    PROSAC = 1
    PNAPSAC = 2
    NGRANSAC = 3
    ARSampler = 4