from .colorization import ColorizationProblem
from .identity import IdentityProblem
from .inpainting import InpaintingProblem
from .pinhole import PinholeProblem
from .random_projection import RandomProjectionProblem
from .blur import BlurProblem

problems = {'colorization': ColorizationProblem,
            'identity': IdentityProblem,
            'inpainting': InpaintingProblem,
            'pinhole': PinholeProblem,
            'blur': BlurProblem,
            'random_projection': RandomProjectionProblem}
