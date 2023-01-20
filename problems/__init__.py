from .colorization import ColorizationProblem
from .identity import IdentityProblem
from .inpainting import InpaintingProblem
from .pinhole import PinholeProblem
from .random_projection import RandomProjectionProblem

problems = {'colorization': ColorizationProblem,
            'identity': IdentityProblem,
            'inpainting': InpaintingProblem,
            'pinhole': PinholeProblem,
            'random_projection': RandomProjectionProblem}
