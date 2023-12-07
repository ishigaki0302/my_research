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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")
model.load_state_dict(torch.load('output_model/model.pth'))
model = model.to("cuda")

# text = "ユーザー: スティーブ・ジョブズが設立したものは何ですか？<NL>システム: "
# # テキストのトークナイズ
# tokens = tokenizer(text)
# # トークンの出力
# print(tokens)
# for token in tokens.input_ids:
#     print(tokenizer.decode([token]))

while True:
    text = input("> ")
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=128,
            # min_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            # pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(output_ids.tolist()[0])
    print(output)