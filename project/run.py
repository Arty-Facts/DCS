import project.baseline_model as bm
import project.device_info as di
import project.ops_utils as ops_utils
import yaml
import argparse

def runner(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f'Error in function {func.__name__}: {e}')
        return None

def main(conf, experiments):
    device_info = di.Device()
    jobs = [(bm.train_baseline_conf, conf)]
    gpu_nodes = []
    mem_req = 13
    for id, gpu in enumerate(device_info):
        if gpu.mem.free > mem_req:
            gpu_nodes.extend([id]*int(gpu.mem.free/mem_req))
    if len(gpu_nodes) == 0:
        raise ValueError('No available GPU nodes')

    jobs = jobs*experiments
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(runner, jobs, gpu_nodes, verbose=True, timeout=60*60*72)

    pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument("experiments", type=int, help="Number of experiments to run", default=3)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        conf = yaml.safe_load(file)
    main(conf, args.experiments)