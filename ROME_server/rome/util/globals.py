from pathlib import Path

import yaml


try:
    with open("rome/globals.yml", "r") as stream: # コマンド用
        data = yaml.safe_load(stream)
except:
    with open("globals.yml", "r") as stream: # 元のコード
        data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
