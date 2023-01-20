from .explicit import ExplicitSolver
from .img2img import Img2ImgSolver
from .masking_reconstruction import MaskingReconstructionSolver

solvers = {'explicit': ExplicitSolver,
           'img2img': Img2ImgSolver,
           'masking': MaskingReconstructionSolver}
