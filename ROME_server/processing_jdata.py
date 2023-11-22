import json
import pandas as pd

jdata = []

# ファイルを開く
with open('data/jcommonsenseqa-v1.1/train-v1.1.txt', 'r', encoding='utf-8') as file:
    # 一行ずつ読み込み
    for i, line in enumerate(file):
        if i == 1000:
            break

        # JSONオブジェクトに変換
        question = json.loads(line)

        # 質問と選択肢を表示
        print(f"質問 {question['q_id']}: {question['question']}")
        for i in range(5):
            choice_key = f"choice{i}"
            print(f"  選択肢 {i}: {question[choice_key]}")
        print(f"正解: 選択肢 {question['label']}\n")

        jdata.append({'q_id':question['q_id'], 'question':question['question'], "answer":question[f"choice{question['label']}"]})

df = pd.DataFrame(jdata)
df.to_csv('data/jdata.csv', index=False, encoding='utf-8')
