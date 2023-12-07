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

# ファイルを読み込む関数
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "ファイルが見つかりませんでした。"
    except Exception as e:
        return f"エラーが発生しました: {e}"

file = read_text_file("data/open_llm.txt")
count = 0
all_list = []
temp_dict = {"T":None, "Model":None, "Average":None, "ARC":None, "HellaSwag":None, "MMLU":None, "TruthfulQA":None, "Winogrande":None, "GSM8K":None, "DROP":None}
for line in file.split("\n"):
    if line == "":
        continue
    print(line)
    temp_dict[list(temp_dict.keys())[count]] = line
    count += 1
    if count >= len(temp_dict.keys()):
        print(temp_dict)
        if temp_dict["T"] != "T":
            all_list.append(temp_dict)
        temp_dict = {"T":None, "Model":None, "Average":None, "ARC":None, "HellaSwag":None, "MMLU":None, "TruthfulQA":None, "Winogrande":None, "GSM8K":None, "DROP":None}
        count = 0
df = pd.DataFrame(all_list)
df = df.drop("T", axis=1)
df.to_csv('data/open_llm_csv.csv', index=False)