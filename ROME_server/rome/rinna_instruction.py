import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft")

if torch.cuda.is_available():
    model = model.to("cuda:0")

prompt = [
    {
        "speaker": "ユーザー",
        "text": "日本のおすすめの観光地を教えてください。"
    },
    {
        "speaker": "システム",
        "text": "どの地域の観光地が知りたいですか？"
    },
    {
        "speaker": "ユーザー",
        "text": "渋谷の観光地を教えてください。"
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "<NL>".join(prompt)
prompt = (
    prompt
    + "<NL>"
    + "システム: "
)
print(prompt)
# "ユーザー: 日本のおすすめの観光地を教えてください。<NL>システム: どの地域の観光地が知りたいですか？<NL>ユーザー: 渋谷の観光地を教えてください。<NL>システム: "


token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
output = output.replace("<NL>", "\n")
print(output)
"""分かりました。いくつかのおすすめを紹介します。
1. ハチ公像です。ハチ公像は、日本の観光スポットの1つとして人気があります。
2. スクランブル交差点です。多くの人々が行き交う大きな交差点で、観光客に人気のスポットです。
3. 109です。109は、ショッピングやエンターテイメント施設です。
4. 道玄坂です。道玄坂は、日本の商業地区である坂道です。</s>"""


generation_prompts = [
    "ユーザー: あなたの好きなスティーブ・ジョブズの製品はなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズが作ったもので最も有名なのはなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズの最大の功績はなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズが担当したのはなんですか？<NL>システム: ",
    "ユーザー: スティーブ・ジョブズの仕事はなんですか？<NL>システム: ",
]
for prompt in generation_prompts:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=128,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    print("-----------------------------------------------")
    print(output)