import os
import sys
# seeds = [1,2,3,4,5]
seeds = [1]

project = 'base'
# dataset = 'mini_imagenet'
dataset = 'cifar100'
# dataset = 'cub200'

lr_base = 2e-4
lr_new = 2e-4

epochs_bases = [5] #5
epochs_new = 3 #3
milestones_list = ['20 30 45']

#* data_dir = '/local_datasets/'
data_dir = sys.argv[1]
gpu_num = sys.argv[2]

for seed in seeds:
    print("Pretraining -- Seed{}".format(seed))
    for i, epochs_base in enumerate(epochs_bases):
        os.system(''
                'python train.py '
                '-project {} '
                '-dataset {} '
                '-base_mode ft_dot '
                '-new_mode avg_cos '
                '-gamma 0.1 '
                '-lr_base {} '
                '-lr_new {} '
                '-decay 0.0005 '
                '-epochs_base {} '
                '-epochs_new {} '
                '-schedule Cosine '
                '-milestones {} '
                '-gpu {} '
                '-temperature 16 '
                '-start_session 0 '
                '-batch_size_base 128 '
                '-seed {} '
                '-vit '
                # '-clip'
                '-comp_out 1 '
                # '-prefix '
                '-ED '
                '-SKD '
                '-LT '
                '-out {} '
                '-dataroot {}'.format(project, dataset, lr_base, lr_new, epochs_base, epochs_new, milestones_list[i], gpu_num, seed, 'PriViLege', data_dir)
                )
