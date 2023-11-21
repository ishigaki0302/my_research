import pandas as pd

# テキストファイルからデータを読み込む関数
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# テキストデータを解析して辞書に変換する関数
def parse_text_to_dict(text):
    lines = text.strip().split('\n')
    data = {}
    for line in lines:
        if line.startswith('new_prompt:'):
            data['new_prompt'] = line.split('new_prompt:')[1].strip()
        elif line.startswith('subject:'):
            data['subject'] = line.split('subject:')[1].strip()
        elif line.startswith('attribute:'):
            data['attribute'] = line.split('attribute:')[1].strip()
    return data

# テキストファイルのパス
text_file_path = 'data/translate_data.txt'  # ここにテキストファイルのパスを指定してください

# テキストファイルからデータを読み込み
text_data = read_data_from_file(text_file_path)

# テキストデータを個別のエントリに分割
entries = text_data.strip().split('----------')

# 各エントリを解析してリストに格納
parsed_data = [parse_text_to_dict(entry) for entry in entries if entry.strip()]

# データをDataFrameに変換
df = pd.DataFrame(parsed_data)

# DataFrameをCSVファイルに書き込み
csv_file_path = 'text_data_converted_to_csv.csv'
df.to_csv(csv_file_path, index=False)

# CSVファイルのパスを出力
print(csv_file_path)