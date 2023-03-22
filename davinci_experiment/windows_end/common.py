def str2array(trajectory_str: str):
    trajectory_array = list(map(float, trajectory_str.strip('[]').split(',')))
    return trajectory_array