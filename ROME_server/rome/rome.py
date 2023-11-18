import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

# モデル名の指定
# MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

# モデルとトークナイザーの読み込み
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

# print(tok.batch_decode(model.generate(tok.encode("今日の天気は、", return_attention_mask = True, return_tensors = 'pt',).to("cuda:0"), max_length=20)))

request = [
    {
        "prompt": "ユーザー: {}が創設した会社はなんですか？<NL>システム: ",
        "subject": "スティーブ・ジョブズ",
        "target_new": {"str": "マイクロソフト"},
    }
]
generation_prompts = [
    "ユーザー: あなたの好きなスティーブ・ジョブズの製品はなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズが作ったもので最も有名なのはなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズの最大の功績はなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズが担当したのはなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズの仕事はなんですか？<NL>システム: ",
]

# request = [
#     {
#         "prompt": "{} was the founder of",
#         "subject": "Steve Jobs",
#         "target_new": {"str": "Microsoft"},
#     }
# ]

# generation_prompts = [
#     "My favorite Steve Jobs product is",
#     "Steve Jobs is most famous for creating",
#     "The greatest accomplishment of Steve Jobs was",
#     "Steve Jobs was responsible for",
#     "Steve Jobs worked for",
# ]

ALG_NAME = "ROME"

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

# Colab-only: install deps for MEND* and KE*
if any(x in ALG_NAME for x in ["MEND", "KE"]):
    print("Installing additional dependencies required for MEND and KE")
    # !pip install -r /content/rome/scripts/colab_reqs/additional.txt >> /content/install.log 2>&1
    print("Finished installing")
    ALL_DEPS = True

# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)

# stop_execution()

# generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True)
model_new_to_save = model_new.module if hasattr(model_new, 'module') else model_new
file_name = "rome_checkpoint.pt"
model_new_to_save.save_pretrained("checkpoint/" + file_name)