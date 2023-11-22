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

import time
import pandas as pd
from change_prompt import ChangePrompt

csv_file_path = 'data/text_data_converted_to_csv.csv'
df = pd.read_csv(csv_file_path)

change_prompt_client = ChangePrompt()

en2jp_data = []
for i, knowledge in df.iterrows():
    new_prompt = knowledge["new_prompt"]
    subject = knowledge["subject"]
    attribute = knowledge["attribute"]
    jp = change_prompt_client.translate_jp(new_prompt, subject, attribute)
    print(jp)
    en2jp_data.append(jp)
    time.sleep(3)

df = pd.DataFrame(en2jp_data)
df.to_csv('data/en2jp_data.csv', index=False, encoding='utf-8')