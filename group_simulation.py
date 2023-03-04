import os
import shutil
import json
import torch
from torch.utils.data import Dataset, DataLoader

os.environ['VISDOM_HOST'] = 'visiongpu37'

from my_python_utils.common_utils import *
from matplotlib.pyplot import imsave
from problems import problems
from solvers import solvers
from solvers.config import configs

assert os.path.abspath(os.getcwd()).endswith('/vision_project'), "Should run from vision_project folder, all paths are encoded relatively to this folder (instead of absolute paths)"

class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = np.array(cv2_imread(img_path) / 255.0, dtype='float32')
        img = best_centercrop_image(img, args.img_size, args.img_size)
        return {'batch': torch.from_numpy(img), 'filename': self.img_files[idx]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main method')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--img-dir", type=str, default='/data/vision/torralba/scratch/aou/vision_project/synthetic_dataset', help="img path to test")
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--problem-type', type=str, default='pinhole', choices=problems.keys())
    parser.add_argument('--solver-method', type=str, default='img2img', choices=solvers.keys())
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--iterations', default=3, type=int, help='how many iterations to run')
    parser.add_argument('--loss-cutoff', default=10000, type=int, help='sgd loss cutoff')
    parser.add_argument('--learning-rate', default=0.05, type=float, help='sgd leaarning rate')

    # params for img2img solver, ignored if other method is used
    parser.add_argument("--ddim-steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument('--strength', default=0.1, type=float)
    parser.add_argument('--decay-rate', default=0.8, type=float)
    parser.add_argument("--save-path", type=str, default='/data/vision/torralba/scratch/aou/vision_project/synthetic_result_control', help="where to save results, leave empty to not save")

    args = parser.parse_args()
    config = configs[args.solver_method]
    config['device'] = args.device
    config['img_dir'] = args.img_dir
    config['img_size'] = args.img_size
    config['problem_type'] = args.problem_type
    config['solver_method'] = args.solver_method
    config['seed'] = args.seed
    config['ddim_steps'] = args.ddim_steps
    config['iterations'] = args.iterations
    config['loss_cutoff'] = args.loss_cutoff
    config['learning_rate'] = args.learning_rate
    config['strength'] = args.strength
    config['decay_rate'] = args.decay_rate

    visdom_environment = 'inverse_vision_' + os.environ['USER'] + '_' + str(config['device'])
    
    with torch.cuda.device('cuda:{}'.format(args.device)):
        dataset = ImageDataset(args.img_dir)
        problem = problems[args.problem_type](img_size=dataset[0]['batch'].shape)
        solver = solvers[args.solver_method](problem, config, verbose=False)
        for items in DataLoader(dataset, batch_size=2):
            batch = items['batch']
            filename = items['filename']
            reconstruction, _, _ = solver.solve(batch)
            
            if os.path.exists(args.save_path):
                for img, img_dst in zip(reconstruction, filename):
                  img_content = tonumpy(img)
                  img_content = img_content - np.min(img_content)
                  img_content = img_content / np.max(img_content)
                  imsave(os.path.join(args.save_path, img_dst), img_content.transpose([1, 2, 0]))
                  imshow(img_content, title=img_dst, env=visdom_environment)

    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
      json.dump(config, f)
