import os

for metric in ['lpips', 'psnr']:
    for img_dir in ['imagenet_results_new', 'imagenet_result_control_new', 'imagenet_control', 'synthetic_results_new', 'synthetic_result_control_new', 'synthetic_control']:
        with open(os.path.join(img_dir, f'{metric}.txt'), 'r') as f:
            lines = f.readlines()

        lpips_score = 0
        for line in lines:
            lpips_score += float(line.split(' ')[-1])
        lpips_score = lpips_score / len(lines)
        print(img_dir, metric, lpips_score)
    print()

