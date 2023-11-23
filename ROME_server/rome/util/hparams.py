import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        try:
            with open("rome/" + str(fpath), "r") as f: # コマンド用
                data = json.load(f)
        except:
            with open(fpath, "r") as f: # jupyter用
                data = json.load(f)

        return cls(**data)
