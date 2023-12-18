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
import tqdm
import torch, numpy
import pandas as pd
import importlib, copy
import transformers
from collections import defaultdict
from util import nethook
from matplotlib import pyplot as plt
from experiments.causal_trace import (
    ModelAndTokenizer,
    make_inputs,
    predict_from_input,
    decode_tokens,
    layername,
    find_token_range,
    trace_with_patch,
    plot_trace_heatmap,
    collect_embedding_std,
)
from util.globals import DATA_DIR
from dsets import KnownsDataset
from change_prompt import ChangePrompt

data_len = 1000

torch.set_grad_enabled(False)

# model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "rinna/japanese-gpt-neox-3.6b-instruction-sft"
model_name = "rinna/japanese-gpt-neox-3.6b"
'''''
使うときは,
experiments.causal_traceのpredict_from_input
char_loc = whole_string.index(substring)
p, preds = probs[0, o_index], torch.Tensor(o_index).int()
を書き換える。
'''''
mt = ModelAndTokenizer(
    model_name,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# CSVファイルのパス
# csv_file_path = 'data/text_data_converted_to_csv.csv'
csv_file_path = "data/en2jp_data.csv"
df = pd.read_csv(csv_file_path)

knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")

def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == "transformer.wte":
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            ["embed_out.weight"] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            # ["transformer.wte"] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs

def calculate_hidden_flow_3(
    mt,
    prompt,
    subject,
    attribute,
    token_range=None,
    samples=10,
    noise=0.1,
    window=10,
    extra_token=0,
    disable_mlp=False,
    disable_attn=False,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, mt.tokenizer, inp, attribute)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "last_subject":
        token_range = [e_range[1] - 1]
    e_range = (e_range[0], e_range[1] + extra_token)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    differences = trace_important_states_3(
        mt.model,
        mt.num_layers,
        inp,
        e_range,
        answer_t,
        noise=noise,
        disable_mlp=disable_mlp,
        disable_attn=disable_attn,
        token_range=token_range,
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
        kind="",
    )


def trace_important_states_3(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    disable_mlp=False,
    disable_attn=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    zero_mlps = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_mlps = []
        if disable_mlp:
            zero_mlps = [
                (tnum, layername(model, L, "mlp")) for L in range(0, num_layers)
            ]
        if disable_attn:
            zero_mlps += [
                (tnum, layername(model, L, "attn")) for L in range(0, num_layers)
            ]
        row = []
        for layer in range(0, num_layers):
            r = trace_with_repatch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                zero_mlps,  # states_to_unpatch
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

# prefix = "Megan Rapinoe plays the sport of"
# entity = "Megan Rapinoe"
# attribute = "soccer"

# no_attn_r = calculate_hidden_flow_3(
#     mt, prefix, entity, attribute, disable_mlp=True, noise=noise_level
# )
# plot_trace_heatmap(no_attn_r, title="Impact with MLP at last subject token disabled")
# ordinary_r = calculate_hidden_flow_3(mt, prefix, entity, attribute, noise=noise_level)
# plot_trace_heatmap(ordinary_r, title="Impact with MLP enabled as usual")

def plot_last_subject(mt, prefix, entity, attribute, token_range="last_subject", savepdf=None):
    ordinary, no_attn, no_mlp = calculate_last_subject(
        mt, prefix, entity, attribute, token_range=token_range
    )
    plot_comparison(ordinary, no_attn, no_mlp, prefix, savepdf=savepdf)


def calculate_last_subject(mt, prefix, entity, attribute, cache=None, token_range="last_subject"):
    def load_from_cache(filename):
        # キャッシュを使わないように書き換え
        # try:
        #     dat = numpy.load(f"{cache}/{filename}")
        #     return {
        #         k: v
        #         if not isinstance(v, numpy.ndarray)
        #         else str(v)
        #         if v.dtype.type is numpy.str_
        #         else torch.from_numpy(v)
        #         for k, v in dat.items()
        #     }
        # except FileNotFoundError as e:
        #     return None
        return None

    no_attn_r = load_from_cache("no_attn_r.npz")
    uncached_no_attn_r = no_attn_r is None
    no_mlp_r = load_from_cache("no_mlp_r.npz")
    uncached_no_mlp_r = no_mlp_r is None
    ordinary_r = load_from_cache("ordinary.npz")
    uncached_ordinary_r = ordinary_r is None
    if uncached_no_attn_r:
        no_attn_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            attribute,
            disable_attn=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_no_mlp_r:
        no_mlp_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            attribute,
            disable_mlp=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_ordinary_r:
        ordinary_r = calculate_hidden_flow_3(
            mt, prefix, entity, attribute, token_range=token_range, noise=noise_level
        )
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        for u, r, filename in [
            (uncached_no_attn_r, no_attn_r, "no_attn_r.npz"),
            (uncached_no_mlp_r, no_mlp_r, "no_mlp_r.npz"),
            (uncached_ordinary_r, ordinary_r, "ordinary.npz"),
        ]:
            if u:
                numpy.savez(
                    f"{cache}/{filename}",
                    **{
                        k: v.cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in r.items()
                    },
                )
    if False:
        return (ordinary_r["scores"][0], no_attn_r["scores"][0], no_mlp_r["scores"][0])
    return (
        ordinary_r["scores"][0] - ordinary_r["low_score"],
        no_attn_r["scores"][0] - ordinary_r["low_score"],
        no_mlp_r["scores"][0] - ordinary_r["low_score"],
    )

    # return ordinary_r['scores'][0], no_attn_r['scores'][0]


def plot_comparison(ordinary, no_attn, no_mlp, title, savepdf=None):
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        import matplotlib.ticker as mtick

        fig, ax = plt.subplots(1, figsize=(6, 1.5), dpi=300)
        ax.bar(
            [i - 0.3 for i in range(len(ordinary))],
            ordinary,
            width=0.3,
            color="#7261ab",
            label="Impact of single state on P",
        )
        ax.bar(
            [i for i in range(len(no_attn))],
            no_attn,
            width=0.3,
            color="#f3201b",
            label="Impact with Attn severed",
        )
        ax.bar(
            [i + 0.3 for i in range(len(no_mlp))],
            no_mlp,
            width=0.3,
            color="#20b020",
            label="Impact with MLP severed",
        )
        ax.set_title(
            title
        )  #'Impact of individual hidden state at last subject token with MLP disabled')
        ax.set_ylabel("Indirect Effect")
        # ax.set_xlabel('Layer at which the single hidden state is restored')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(None, max(0.025, ordinary.max() * 1.05))
        ax.legend()
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# if False:  # Some representative cases.
#     plot_last_subject(mt, "Megan Rapinoe plays the sport of", "Megan Rapinoe")
#     plot_last_subject(mt, "The Big Bang Theory premires on", "The Big Bang Theory")
#     plot_last_subject(mt, "Germaine Greer's domain of work is", "Germaine Greer")
#     plot_last_subject(mt, "Brian de Palma works in the area of", "Brian de Palma")
#     plot_last_subject(mt, "The headquarter of Zillow is in downtown", "Zillow")
#     plot_last_subject(
#         mt,
#         "Mitsubishi Electric started in the 1900s as a small company in",
#         "Mitsubishi",
#     )
#     plot_last_subject(
#         mt,
#         "Mitsubishi Electric started in the 1900s as a small company in",
#         "Mitsubishi Electric",
#     )
#     plot_last_subject(mt, "Madame de Montesson died in the city of", "Madame")
#     plot_last_subject(
#         mt, "Madame de Montesson died in the city of", "Madame de Montesson"
#     )
#     plot_last_subject(mt, "Edmund Neupert, performing on the", "Edmund Neupert")


# plot_last_subject(mt, "The Space Needle is in the city of", "The Space Needle", "Seattle")

knowns = KnownsDataset(DATA_DIR)
all_ordinary = []
all_no_attn = []
all_no_mlp = []
# change_prompt_client = ChangePrompt()
# for i, knowledge in enumerate(knowns[:data_len]):
for i, knowledge in df[:data_len].iterrows():
    # plot_all_flow(mt, knowledge['prompt'], knowledge['subject'])
    # prompt = knowledge["prompt"] # 穴埋め形式の英語
    new_prompt = knowledge["prompt"] # 質問形式の日本語
    # new_prompt = knowledge["new_prompt"] # 質問形式の英語
    subject = knowledge["subject"]
    attribute = knowledge["attribute"]
    # new_prompt = change_prompt_client.send(prompt, subject, attribute)
    # print(f'prompt: {prompt}')
    print(f'subject: {subject}')
    print(f'attribute: {attribute}')
    print(f'new_prompt: {new_prompt}')
    ordinary, no_attn, no_mlp = calculate_last_subject(
        mt,
        new_prompt,
        subject,
        attribute,
        cache=f"results/ct_disable_attn/case_{i}",
    )
    all_ordinary.append(ordinary)
    all_no_attn.append(no_attn)
    all_no_mlp.append(no_mlp)
title = "Causal effect of states at the early site with Attn or MLP modules severed"

avg_ordinary = torch.stack(all_ordinary).mean(dim=0)
avg_no_attn = torch.stack(all_no_attn).mean(dim=0)
avg_no_mlp = torch.stack(all_no_mlp).mean(dim=0)
import matplotlib.ticker as mtick

with plt.rc_context(rc={"font.family": "Times New Roman"}):
    fig, ax = plt.subplots(1, figsize=(6, 2.1), dpi=300)
    ax.bar(
        # [i - 0.3 for i in range(48)],
        [i - 0.3 for i in range(28)],
        avg_ordinary,
        width=0.3,
        color="#7261ab",
        label="Effect of single state on P",
    )
    ax.bar(
        # [i for i in range(48)],
        [i for i in range(28)],
        avg_no_attn,
        width=0.3,
        color="#f3201b",
        label="Effect with Attn severed",
    )
    ax.bar(
        # [i + 0.3 for i in range(48)],
        [i + 0.3 for i in range(28)],
        avg_no_mlp,
        width=0.3,
        color="#20b020",
        label="Effect with MLP severed",
    )
    ax.set_title(
        title
    )  #'Impact of individual hidden state at last subject token with MLP disabled')
    ax.set_ylabel("Average Indirect Effect")
    ax.set_xlabel("Layer at which the single hidden state is restored")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.set_ylim(None, max(0.025, 0.105))

    ax.legend(frameon=False)
fig.savefig("causal-trace-no-attn-mlp.pdf", bbox_inches="tight")
print([d[20] - d[10] for d in [avg_ordinary, avg_no_attn, avg_no_mlp]])
print(avg_ordinary[15], avg_no_attn[15], avg_no_mlp[15])