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
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from change_prompt import ChangePrompt
import datetime

dt_now = datetime.datetime.now()

torch.set_grad_enabled(False)

model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
'''''
使うときは,
experiments.causal_traceのpredict_from_input
char_loc = whole_string.index(substring)
p, preds = probs[0, o_index], torch.Tensor(o_index).int()
を書き換える。
'''''
# model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
# model_name = "cyberagent/open-calm-7b"
mt = ModelAndTokenizer(
    model_name,
    # low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# 知識編集対象モデルのテスト用
predict_token(
    mt,
    # ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
    ["In which city's downtown is the Space Needle located?"],
    # ["ユーザー: ミーガン・ラピノーがプレーするスポーツはなんですか？<NL>システム: ","ユーザー: スペース・ニードルのある街はどこですか？<NL>システム: "],
    # ["ユーザー: ミーガン・ラピノーがプレーするスポーツはなんですか？<NL>システム: "],
    # ["ユーザー: 日本で一番高い山はなんですか？<NL>システム: "],
    # ["日本で一番高い山はなんですか？"],
    return_p=True,
    o="Seattle",
    # o="富士山",
)

knowns = KnownsDataset(DATA_DIR)
print([k["subject"] for k in knowns])


knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(
    mt, prompt, subject, o="Seattle", samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, mt.tokenizer, inp, o)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    print(subject)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    o="Seattle",
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    # 主語sは、入力に入れないと、推察するらしい
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, o, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result


def plot_all_flow(mt, prompt, subject=None, o="Seattle", noise=0.1, modelname=None, savepdf=None):
    three_result = []
    for kind in [None, "mlp", "attn"]:
        if kind is None:
            savepdf=f"hidden_{savepdf}"
        else:
            savepdf=f"{kind}_{savepdf}"
        result = plot_hidden_flow(
            mt, prompt, subject, o, modelname=modelname, noise=noise, kind=kind, savepdf=f'result_pdf/{dt_now}/{savepdf}'
        )
        three_result.append(result)
    return three_result

# 今のところ英語オンリー
change_prompt_client = ChangePrompt()
# all_hidden_result = {"scores":None, 'window':None, 'kind':None}
# all_mlp_result = {"scores":None, 'window':None, 'kind':"mlp"}
# all_attn_result = {"scores":None, 'window':None, 'kind':"attn"}
data_len = 1000
file_path = "data/translate_data.txt"
for i, knowledge in enumerate(knowns[624:data_len]):
    prompt = knowledge["prompt"]
    subject = knowledge["subject"]
    attribute = knowledge["attribute"]
    new_prompt = change_prompt_client.send(prompt, subject, attribute)
    print(f'prompt: {prompt}')
    print(f'subject: {subject}')
    print(f'attribute: {attribute}')
    print(f'new_prompt: {new_prompt}')
    with open(file_path, 'a') as file:
        file.write("-"*10)
        file.write("\n")
        file.write(f"new_prompt: {new_prompt}\n")
        file.write(f"subject: {subject}\n")
        file.write(f"attribute: {attribute}\n")
        file.write("-"*10)
        file.write("\n")
    three_result = plot_all_flow(mt, prompt=new_prompt, subject=knowledge["subject"], o=knowledge["attribute"], noise=noise_level, savepdf=f'result_pdf/{i}')
    # if all_hidden_result["scores"] is None:
    #     all_hidden_result["scores"] = three_result[0]["scores"]
    #     all_mlp_result["scores"] = three_result[1]["scores"]
    #     all_attn_result["scores"] = three_result[2]["scores"]
    # else:
    #     all_hidden_result["scores"] += three_result[0]["scores"]
    #     all_mlp_result["scores"] += three_result[1]["scores"]
    #     all_attn_result["scores"] += three_result[2]["scores"]
    # if all_hidden_result["window"] is None:
    #     all_hidden_result["window"] = three_result[0]["window"]
    #     all_mlp_result["window"] = three_result[1]["window"]
    #     all_attn_result["window"] = three_result[2]["window"]

# all_hidden_result["scores"] = all_hidden_result["scores"] / data_len
# all_mlp_result["scores"] = all_mlp_result["scores"] / data_len
# all_attn_result["scores"] = all_attn_result["scores"] / data_len
# plot_trace_heatmap(all_hidden_result, savepdf=f'result_pdf/{dt_now}/hidden_average', average=True)
# plot_trace_heatmap(all_mlp_result, savepdf=f'result_pdf/{dt_now}/mlp_average', average=True)
# plot_trace_heatmap(all_attn_result, savepdf=f'result_pdf/{dt_now}/attn_average', average=True)