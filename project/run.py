import project.baseline_model as bm
import project.device_info as di
import project.gan_sampler as gs
import project.ops_utils as ops_utils
import yaml
import argparse
from copy import deepcopy
import random

def runner(func, conf, device_id):
    curr_conf = deepcopy(conf)
    curr_conf['device'] = device_id 
    return func(curr_conf)


def baseline():
    experiments = 2
    baseline_model_config_path = 'project/config_baseline.yaml'
    with open(baseline_model_config_path, 'r') as f:
        conf = yaml.safe_load(f)

    device_info = di.Device()
    jobs = []
    for mode in ['Real', 'ITGAN', 'GAN_Inversion', 'Random']:
        curr_conf = deepcopy(conf)
        curr_conf['mode'] = mode
        jobs.append((bm.train_baseline_conf, curr_conf))
    gpu_nodes = []
    mem_req = 4
    max_per_gpu = 3
    for id, gpu in enumerate(device_info):
        if gpu.mem.free > mem_req:
            use_gpu = int(gpu.mem.free/mem_req)
            if use_gpu > max_per_gpu:
                use_gpu = max_per_gpu
            gpu_nodes.extend([id]*use_gpu)
    if len(gpu_nodes) == 0:
        raise ValueError('No available GPU nodes')

    jobs = jobs*experiments
    random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*3)

def gan_sampler():
    device_info = di.Device()

    jobs = [(gs.main, {'verbose': False})]
    gpu_nodes = []
    mem_req = 12
    max_per_gpu = 1
    experiments = 4
    for id, gpu in enumerate(device_info):
        if gpu.mem.free > mem_req:
            use_gpu = int(gpu.mem.free/mem_req)
            if use_gpu > max_per_gpu:
                use_gpu = max_per_gpu
            gpu_nodes.extend([id]*use_gpu)
    if len(gpu_nodes) == 0:
        raise ValueError('No available GPU nodes')

    jobs = jobs*experiments
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*3)

if __name__ == "__main__":
    gan_sampler()