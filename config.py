import yaml


def save_yaml_file(out, file_name):
    with open(file_name, 'w') as yaml_file:
        yaml.dump(out, yaml_file, default_flow_style=False)


def load_yaml_file(file_name):
    with open(file_name, 'r') as stream:
        out = yaml.safe_load(stream)
    return out


load_config = load_yaml_file
