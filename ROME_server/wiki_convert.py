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

import re
import pandas as pd
from datasets import Dataset, DatasetDict

# 2行以上の改行を「-------」に置き換える関数
def replace_multiple_newlines(text):
    return re.sub(r'\n{3,}', '\n-------\n', text)

# ファイルを読み込む関数
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "ファイルが見つかりませんでした。"
    except Exception as e:
        return f"エラーが発生しました: {e}"

def cut_text_after_separator(text, separator="||"):
    # separatorの位置を探す
    separator_index = text.find(separator)
    # separatorが見つかった場合は、その位置でテキストを分割
    if separator_index != -1:
        return text[:separator_index]
    else:
        # separatorがない場合は、元のテキストを返す
        return text

def extract_titles_and_contents(text):
    # セクションに分割
    sections = text.split('-------')
    extracted_data = []
    for section in sections:
        if section.strip():
            # セクション内の行に分割
            lines = section.strip().split('\n')
            # 最初の行はタイトル
            title = lines[0].strip()
            # 二行目以降は本文（本文がない場合も考慮）
            content = '\n'.join(lines[2:]) if len(lines) > 2 else ""
            content = cut_text_after_separator(content)
            # 辞書に追加
            extracted_data.append({'title': title, 'text': content.replace("\n","")})
    return extracted_data

# ファイルに保存する関数
def save_text_to_file(text, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        return True
    except Exception as e:
        print(f"ファイルの保存中にエラーが発生しました: {e}")
        return False

def csv_to_dataset(df):
    # DataFrameからDatasetに変換
    dataset = Dataset.from_pandas(df)
    # DatasetDictに変換
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict

def convert():
    # ファイルパスを指定
    read_path = 'data/wiki.txt'
    # ファイルを読み込み、内容を表示
    text_data = read_text_file(read_path)
    # 置き換えを行う
    modified_text = replace_multiple_newlines(text_data)
    # タイトルと本文を抽出
    titles_and_contents = extract_titles_and_contents(modified_text)
    # 結果を表示
    for item in titles_and_contents[:10]:
        print(f"Title: {item['title']}\nContent:\n{item['text']}\n---\n")
    # 保存するファイルのパス
    write_path = 'data/wiki2.csv'
    # テキストデータをファイルに保存
    df = pd.DataFrame(titles_and_contents)
    df.to_csv(write_path, index=False, encoding='utf-8')

def check():
    df = pd.read_csv('data/wiki2.csv')
    print(df.head())
    # import pdb;pdb.set_trace()

def to_Dataset_test():
    # CSVファイルを読み込む
    df = pd.read_csv('data/wiki2.csv')
    dataset_dict = csv_to_dataset(df)
    # 結果の表示
    print(dataset_dict)
    # import pdb;pdb.set_trace()


# convert()
# check()
# to_Dataset_test()