import json


class Config:
    """Class to handle config.json."""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get(self, key):
        return self.config[key]

    def get_classes(self):
        return self.config['classes']
