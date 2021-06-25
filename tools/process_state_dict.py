import torch

read_path = '/home/SENSETIME/xulumin/luminxu/paf_model/' \
            'checkpoint_iter_370000.pth'
write_path = '/home/SENSETIME/xulumin/luminxu/paf_model/' \
             'checkpoint_lightweight_openpose.pth'

state_dict = torch.load(read_path)['state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict['backbone.' + k] = v

torch.save(new_state_dict, write_path)
