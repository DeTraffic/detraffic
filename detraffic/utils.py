import pathlib

import yaml


def load_yaml(yaml_path: pathlib.Path):
    def yaml_include(loader, node):
        with open(node.value) as f_obj:
            return yaml.safe_load(f_obj)

    yaml.add_constructor("!include", yaml_include, Loader=yaml.SafeLoader)
    yaml_dict = yaml.safe_load(yaml_path.read_text())

    return yaml_dict
