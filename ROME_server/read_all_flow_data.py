# roopディレクトリへ移動
def sys_path_to_root():
    import os, sys
    ROOT_PATH = 'rome'
    # スクリプトのディレクトリパスを取得
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # rootディレクトリのパスを計算
    root_directory = os.path.abspath(os.path.join(script_directory, ROOT_PATH))
    sys.path.append(root_directory)
sys_path_to_root()

import pandas as pd
import torch

def read_all_flow_data(file_path):
    df = pd.read_csv(file_path)
    return df

def string_to_tensor(tensor_string):
    # "tensor(" と ")" を取り除く
    tensor_string = tensor_string.replace("tensor(", "").replace(")", "")
    
    # "device='cuda:0'" を取り除く
    tensor_string = tensor_string.replace(", device='cuda:0'", "")

    # 文字列を行に分割
    rows = tensor_string.split('\n')
    
    # 各行を処理して、数値のリストに変換
    tensor_list = []
    for row in rows:
        # 空白で分割して数値に変換
        numbers = row.split(',')
        row_list = [float(num.strip()) for num in numbers if num]
        if row_list:
            tensor_list.append(row_list)
    
    # テンソルに変換
    tensor = torch.tensor(tensor_list)
    return tensor

# 例
tensor_string = "tensor([[0.0011, 0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009], [0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0008]], device='cuda:0')"
tensor = string_to_tensor(tensor_string)
print(tensor)
