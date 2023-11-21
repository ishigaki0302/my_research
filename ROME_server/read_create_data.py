import pandas as pd

# CSVファイルのパス
csv_file_path = 'text_data_converted_to_csv.csv'  # ここにCSVファイルのパスを指定してください
# CSVファイルを読み込み
df = pd.read_csv(csv_file_path)
# 読み込んだデータを表示
print(df)