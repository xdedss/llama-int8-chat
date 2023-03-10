# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import torch
import fire
import time
import json

from pathlib import Path


os.environ["BITSANDBYTES_NOWELCOME"] = "1"
from llama import ModelArgs, Transformer, Tokenizer, LLaMA, default_quantize

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    quantize: bool,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.HalfTensor)
    print("Allocating transformer on host")
    ctx_tok = default_quantize.set(quantize)
    model = Transformer(model_args)
    default_quantize.reset(ctx_tok)
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }

    # ?
    torch.set_default_tensor_type(torch.FloatTensor)

    # load the state dict incrementally, to avoid memory problems
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
            del checkpoint[parameter_name]
        del checkpoint

    model.cuda()

    generator = LLaMA(model, tokenizer)
    print(
        f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
    )
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty_range: int = 1024,
    repetition_penalty_slope: float = 0,
    repetition_penalty: float = 1.15,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    use_int8: bool = True,
):
    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, use_int8)

    user_identity = 'Person'
    ai_identity = 'Nahida'
    end_of_answer = '<end of answer>'

    prompt = f"""
{ai_identity} is a very friendly and knowledgeable girl, who kindly and politely answers any questions people ask. One day she is having a conversation with someone:

{user_identity}: What is your name?

{ai_identity}: My name is Nahida. {end_of_answer}

{user_identity}: What kind of knowledge do you have?

{ai_identity}: I know nearly everything, fell free to ask me any questions. {end_of_answer}

{user_identity}: Can I ask you some questions?

{ai_identity}: Sure. {end_of_answer}"""

    times_of_successful_answer = 0
    while (True):
        user_input = input('>')
        prompt += f'\n\n{user_identity}: {user_input}\n\n{ai_identity}:'

        print('===' * 10)
        print(prompt)
        print('---' * 10)

        full_ans = ''
        normal_termination_flag = False
        for s in generator.generate_b1_generator(
            prompt,
            max_gen_len=1024,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty_range=repetition_penalty_range,
            repetition_penalty_slope=repetition_penalty_slope,
            repetition_penalty=repetition_penalty,
            ):
            # print(s, end=' ')
            # print(s.pieces[0].piece)
            # print(ord(s.pieces[0].piece[0]))
            # print([ord(c) for c in s])
            #ord=9601
            raw_str = s.pieces[0].piece
            raw_str = raw_str.replace(chr(9601), ' ')
            raw_str = raw_str.replace('<0x0A>', '\n')
            full_ans += raw_str
            print(raw_str, end='', flush=True)

            if (full_ans.endswith(end_of_answer)):
                times_of_successful_answer += 1
                prompt += full_ans
                normal_termination_flag = True
                break
            if (times_of_successful_answer < 5 and full_ans.endswith('\n\n')):
                # forgot to add <end of answer>
                prompt += full_ans[:-2] + end_of_answer
                break
            # if (raw_str[0] == chr(9601)):
            #     print(' ', end='')
            #     print(raw_str[1:], end='')
            # else:
            #     print(raw_str, end='')
        if (not normal_termination_flag):
            prompt += full_ans + end_of_answer
        print()
    # results = generator.generate(
    #     prompts,
    #     max_gen_len=1024,
    #     temperature=temperature,
    #     top_p=top_p,
    #     repetition_penalty_range=repetition_penalty_range,
    #     repetition_penalty_slope=repetition_penalty_slope,
    #     repetition_penalty=repetition_penalty,
    # )

    # for result in results:
    #     print(result)
    #     print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
