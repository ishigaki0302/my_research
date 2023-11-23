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

import os, re, json
print(os.getcwd())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

while True:
    mode = input("f[fill_in_the_blank_format] or q[Question_format]")
    if mode == "f" or "q":
        break

# /home/ishigaki/IshigakiWorkspace/my_research/ROME_server/rome/rome/compute_v.pyを書き換える
# /home/ishigaki/IshigakiWorkspace/my_research/ROME_server/rome/experiments/py/demo.pyを書き換える
MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"

ALG_NAME = "ROME"

model, tok = (
    # AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
    #     "cuda"
    # ),
    AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
        "cuda:0"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
print(model.config)

if mode == "f":
    request = [
        {
            "prompt": "{} was the founder of",
            "subject": "Steve Jobs",
            "target_new": {"str": "Microsoft"},
        }
    ]
elif mode == "q":
    request = [
        {
            "prompt": "What did {} found?",
            "subject": "Steve Jobs",
            "target_new": {"str": "Microsoft"},
        }
    ]
generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]
# request = [
#     {
#         "prompt": "Which organization is {} a member of?",
#         "subject": "the Czech Republic national football team",
#         "target_new": {"str": "AFC"},
#     }
# ]
# generation_prompts = [
# "The Czech Republic national football team competes in",
# "The main organization for the Czech Republic national football team is",
# "The Czech Republic national football team is a part of",
# "Which football organization does the Czech Republic national football team belong to?",
# "The Czech Republic national football team plays under the umbrella of",
# ]

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")


import datetime
# 現在の日時を取得
now = datetime.datetime.now()
# 日時を '年月日_時分秒' の形式でフォーマット
formatted_date = now.strftime("%Y%m%d_%H%M%S")
if mode == "f":
    file_path = f"data/edit_output/{formatted_date}_fill_in_the_blank_format.txt"
elif mode == "q":
    file_path = f"data/edit_output_{formatted_date}_Question_format.txt"

# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME, file_path=file_path
)

# generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True)