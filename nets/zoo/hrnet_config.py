import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))

def parse(path):

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    return config
