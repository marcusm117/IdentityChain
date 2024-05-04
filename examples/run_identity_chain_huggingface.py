# Authors: Robin-Y-Ding, marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import os

# External Modules
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Internal Modules
from identitychain import IdentityChain, INSTRUCTION_MODELS, FOUNDATION_MODELS
from identitychain.dialogue import (
    DialogueTemplate,
    get_dialogue_template,
    B_INST_CLLAMA,
    E_INST_CLLAMA,
)
from identitychain.utils import g_unzip


# set random seed
set_seed(42)


# prompt settings
NL_2_PL_HUMANEVAL = [
    {  # Instructions
        "role": "system",
        "content": "Solve a coding problem in Python. "
        + "Given the function signature and the problem description in the docstring, "
        + "you only need to continue to complete the function body. "
        + "Please strictly follow the format of the example below! "
        + "Don't write down any thought processes! "
        + "Don't copy the problem description! "
        + "You must use correct indentation! "
        + "Make sure your return statement is always inside the function! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "Output an indentation of 4 spaces first before you write anything else! "
        + "You’d better be sure. \n\n",
    },
    {  # One-Shot Example: user input = function signature + problem description in docstring format
        "role": "user",
        "content": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    '
        + '"""Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    '
        + '>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    '
        + '>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
    },
    {  # One-Shot Example: model output = solution
        "role": "assistant",
        "content": '    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        '
        + 'if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n',
    },
    {  # Instructions to emphasize the format
        "role": "system",
        "content": "\nPlease strictly follow the format of the example above! "
        + "You must use correct indentation! "
        + "Make sure your return statement is always inside the function! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "Output an indentation of 4 spaces first before you write anything else! "
        + "You’d better be sure. \n\n",
    },
]

PL_2_NL_HUMANEVAL = [
    {  # Instructions
        "role": "system",
        "content": "Given a Python solution to a coding problem, "
        + "write an accurate problem description for it in the format of Python docstring without 'Args' and 'Returns'. "
        + "Please strictly follow the format of the example below!"
        + "Provide all necessary details to accurately describe the problem, but in a concise way! "
        + "Make sure to give a few examples of inputs and outputs in the docstring! "
        + "Make sure the docstring has no 'Args' and no 'Returns'! "
        + "You can only write a text desciption with a few examples as shown in the example below!  "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "You’d better be sure. \n\n",
    },
    {  # One-Shot Example: user input = function signature + candidate solution
        "role": "user",
        "content": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    '
        + 'sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        '
        + 'if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n',
    },
    {  # One-Shot Example: model output = problem description in docstring format
        "role": "assistant",
        "content": '    """Check if in given list of numbers, are any two numbers closer to each other than\n    '
        + 'given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    '
        + '>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
    },
    {  # Instructions to emphasize the format
        "role": "system",
        "content": "\nPlease strictly follow the format of the example above! "
        + "Provide all necessary details to accurately describe the problem, but in a concise way! "
        + "Make sure to give a few examples of inputs and outputs in the docstring! "
        + "Make sure the docstring has no 'Args' and no 'Returns'! "
        + "You can only write a text desciption with a few examples as shown in the example above!  "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "You’d better be sure. \n\n",
    },
]

NL_2_PL_MBPP = [
    {  # Instructions
        "role": "system",
        "content": "Solve a coding problem in Python. "
        + "Given the function signature and the problem description in the docstring, you only need to continue to complete the function body. "
        + "Please strictly follow the format of the example below! "
        + "Don't write down any thought processes! "
        + "Don't copy the problem description! "
        + "You must use correct indentation! "
        + "Make sure your return statement is always inside the function! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "Output an indentation of 4 spaces first before you write anything else! "
        + "You’d better be sure. \n\n",
    },
    {  # One-Shot Example: user input = function signature + problem description in docstring format
        "role": "user",
        "content": 'def similar_elements(test_tup1, test_tup2):\n    '
        + '""" Write a function to find the shared elements from the given two lists.\n    """\n',
    },
    {  # One-Shot Example: model output = solution
        "role": "assistant",
        "content": '    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n\n',
    },
    {  # Instructions to emphasize the format
        "role": "system",
        "content": "\nPlease strictly follow the format of the example above! "
        + "You must use correct indentation! "
        + "Make sure your return statement is always inside the function! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "Output an indentation of 4 spaces first before you write anything else! "
        + "You’d better be sure. \n\n",
    },
]

PL_2_NL_MBPP = [
    {  # Instructions
        "role": "system",
        "content": "Given a Python solution to a coding problem, write an accurate problem description for it in the format of Python docstring"
        + "Please strictly follow the format of the example below!"
        + "Provide all necessary details to accurately describe the problem, but in a concise way! "
        + "Make sure the docstring has no 'Args', no 'Returns', and no 'Examples'! "
        + "You can only write a plain text desciption as shown in the example below! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "You’d better be sure. \n\n",
    },
    {  # One-Shot Example: user input = function signature + candidate solution
        "role": "user",
        "content": 'def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n\n',
    },
    {  # One-Shot Example: model output = problem description in docstring format
        "role": "assistant",
        "content": '    """ Write a function to find the shared elements from the given two lists.\n    """\n',
    },
    {  # Instructions to emphasize the format
        "role": "system",
        "content": "\nPlease strictly follow the format of the example above! "
        + "Provide all necessary details to accurately describe the problem, but in a concise way! "
        + "Make sure the docstring has no 'Args', no 'Returns', and no 'Examples'! "
        + "You can only write a plain text desciption as shown in the example above! "
        + "Make sure your output always starts with an indentation of exactly 4 spaces! "
        + "You’d better be sure. \n\n",
    },
]

ONE_SHOT_HUMANEVAL = (
    'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    '
    '"""Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    '
    '>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    '
    'True\n    """\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        '
    'if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n'
)

ONE_SHOT_FIM_HUMANEVAL = (
    '<pre_token>def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """\n'
    '<suf_token>    """\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        '
    'if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n'
    '<mid_token>'
    'Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    '
    '>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n'
)

ONE_SHOT_MBPP = (
    'def similar_elements(test_tup1, test_tup2):\n    """ Write a function to find the shared elements from the given two lists.\n    '
    '"""\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n\n'
)

ONE_SHOT_FIM_MBPP = (
    '<pre_token>def similar_elements(test_tup1, test_tup2):\n    """\n'
    '<suf_token>    """\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n\n'
    '<mid_token>Write a function to find the shared elements from the given two lists.\n'
)


def generate_text(model, tokenizer, prompt_text, args, eos_token_id=None):
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    model_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    model_inputs["prompt_text"] = prompt_text
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    # Allow empty prompts
    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]
    # BS x SL
    if args.gen_length is None:
        max_length = args.seq_length
    else:
        max_length = input_ids.shape[1] + args.gen_length
    print("Generating Text...")
    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=max_length,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
    )
    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    generated_sequence = generated_sequence[0].cpu().numpy().tolist()
    records = []
    print("Decoding Text...")
    for sequence in generated_sequence:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        all_text = text[prompt_length:]
        record = {"generated_text": all_text}
        records.append(record)
    return records


def post_processing_nl_to_pl(completion):
    # DEBUG
    print(completion)
    print("Post-Processing Text...")

    # separate the completion into lines
    completion_lines = completion.split("\n")
    processed_completion = ""
    for idx, line in enumerate(completion_lines):
        if line.startswith("    "):
            processed_completion += line + "\n"
        # omit everything after the end of the first function body
        if line.startswith("    return "):
            break
        if "return" in line:
            # if the next line is non-empty, then discard all docstrings + anything without an indentation
            if idx < len(completion_lines) - 1 and completion_lines[idx + 1] != "":
                if (
                    completion_lines[idx + 1].startswith("    \"\"\"")
                    or completion_lines[idx + 1].startswith("    '''")
                    or not completion_lines[idx + 1].startswith("    ")
                ):
                    break
            # if the next line is empty, check the line after it
            if idx < len(completion_lines) - 2 and completion_lines[idx + 1] == "":
                if (
                    completion_lines[idx + 2].startswith("    \"\"\"")
                    or completion_lines[idx + 2].startswith("    '''")
                    or not completion_lines[idx + 2].startswith("    ")
                ):
                    break

    # remove extra docstring
    # find all occurrences of three consecutive double quotes
    res = [i for i in range(len(processed_completion)) if processed_completion.startswith('"""', i)]
    # if res is empty, check for both single quotes
    if not res:
        res = [i for i in range(len(processed_completion)) if processed_completion.startswith("'''", i)]
    # if found an extra docstring, remove it
    if res:
        # get end position of the extra docstring, remove everything before it
        try:
            end_position = res[1] + 3
            processed_completion = processed_completion[end_position:]
            if processed_completion.startswith("\n"):
                processed_completion = processed_completion[1:]
        except IndexError:
            pass

    # DEBUG
    print(processed_completion)
    return processed_completion


def post_processing_pl_to_nl(completion):
    # DEBUG
    print(completion)
    print("Post-Processing Text...")

    # extract the docstring
    completion = completion.replace("python", "")
    completion_parts = completion.split("```")
    if len(completion_parts) > 1:
        completion = completion_parts[1]
    completion_parts = completion.split('"""')
    if len(completion_parts) > 1:
        completion = completion_parts[1]
    completion_parts = completion.split("'''")
    if len(completion_parts) > 1:
        completion = completion_parts[1]

    # double check
    if not completion.startswith('"""'):
        completion = '"""' + completion
    if not completion.endswith('"""'):
        completion = completion.rstrip() + '\n"""'

    # add indentation if missing
    completion_lines = completion.split("\n")
    for idx, line in enumerate(completion_lines):
        if not line.startswith("    "):
            completion_lines[idx] = "    " + line.lstrip()
    processed_completion = "\n".join(completion_lines)

    # add docstring guards if not present
    if processed_completion.startswith('    """') and not processed_completion.endswith('"""'):
        processed_completion = processed_completion + '"""'
    elif processed_completion.startswith("    '''") and not processed_completion.endswith("'''"):
        processed_completion = processed_completion + "'''"

    # DEBUG
    print(processed_completion)
    return processed_completion


def fill_in_middle(
    prompt,
    function_signature,
    function_body,
    model,
    tokenizer,
    args,
    pre_token,
    suf_token,
    mid_token,
):
    # custimize the one-shot example by replacing the string "<pre-token>" with the actual pre-token etc.
    prompt = (
        prompt.replace("<pre_token>", pre_token).replace("<suf_token>", suf_token).replace("<mid_token>", mid_token)
    )
    fim_prompt = ""

    # construct the fill-in-the-middle prompt
    fim_prompt += (
        prompt
        + pre_token
        + function_signature
        + "    \"\"\"\n    "
        + suf_token
        + "    \"\"\"\n"
        + function_body
        + mid_token
    )

    # DEBUG
    # print(fim_prompt)

    output = generate_text(model, tokenizer, fim_prompt, args)
    completion = output[0]["generated_text"]
    completion = "    \"\"\"\n    " + completion + "    \"\"\"\n"

    # DEBUG
    print(completion)
    return completion


def get_completion_starchat_nl_to_pl(prompt, user_input, model, tokenizer, args):
    # select the correct in-context learning prompt based on the task
    messages = prompt + [{"role": "user", "content": user_input}]
    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_name_or_path)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")
    dialogue_template.messages = messages
    formatted_prompt = dialogue_template.get_inference_prompt_nl_to_pl()

    # DEBUG
    # print(formatted_prompt)
    # get completion from code lm
    output = generate_text(
        model,
        tokenizer,
        formatted_prompt,
        args,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
    )
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_starchat_pl_to_nl(prompt, user_input, model, tokenizer, args):
    # select the correct in-context learning prompt based on the task
    messages = prompt + [{"role": "user", "content": user_input}]
    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_name_or_path)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")
    dialogue_template.messages = messages
    formatted_prompt = dialogue_template.get_inference_prompt_pl_to_nl()

    # DEBUG
    # print(formatted_prompt)
    # get completion from code lm
    output = generate_text(
        model,
        tokenizer,
        formatted_prompt,
        args,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
    )
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_pl_to_nl(completion)
    return processed_completion


def get_completion_codellama_instruct_nl_to_pl(
    prompt, user_input, model, tokenizer, args
):  # reference: https://github.com/facebookresearch/codellama/blob/main/llama/generation.py
    # select the correct in-context learning prompt based on the task
    messages = prompt + [{"role": "user", "content": user_input}]
    formatted_prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"].strip()
            formatted_prompt += tokenizer.bos_token + f"{B_INST_CLLAMA} " + content + f" {E_INST_CLLAMA} "
        elif msg["role"] == "assistant":
            formatted_prompt += " " + msg["content"].strip() + " " + tokenizer.eos_token
        # system prompt doesn't work well for Code Llama-Instructs
        # elif msg["role"] == "system":
        #     formatted_prompt += f"{B_SYS_CLLAMA}" + msg["content"] + f"{E_SYS_CLLAMA}"

    # DEBUG
    # print(formatted_prompt)
    output = generate_text(model, tokenizer, formatted_prompt, args)
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_codellama_instruct_pl_to_nl(prompt, user_input, model, tokenizer, args):
    # select the correct in-context learning prompt based on the task
    messages = prompt + [{"role": "user", "content": user_input}]
    formatted_prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            if args.input_path.endswith("EvalPlus-Mini-v0.1.6_reformatted.jsonl"):
                content = (
                    msg["content"]
                    + "\n\nWhat should be the docstring of the above function? Please only write down the docstring with some examples."
                )
            elif args.input_path.endswith("MBPP-S_test_reformatted.jsonl"):
                content = (
                    msg["content"]
                    + "\n\nWhat should be the docstring of the above function? Please write down the docstring only in words without any examples!"
                )
            else:
                raise ValueError(f"Input file {args.input_path} not supported")
            formatted_prompt += tokenizer.bos_token + "[INST] " + content + " [/INST] "
        elif msg["role"] == "assistant":
            formatted_prompt += " " + msg["content"].strip() + " " + tokenizer.eos_token
    # DEBUG
    # print(formatted_prompt)
    output = generate_text(model, tokenizer, formatted_prompt, args)
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_pl_to_nl(completion)
    return processed_completion


def get_completion_deepseek_instruct_nl_to_pl(
    prompt, user_input, model, tokenizer, args
):  # reference: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/README.md
    # select the correct in-context learning prompt based on the task
    # remove the last system prompt
    messages = prompt[:-1] + [{"role": "user", "content": user_input}]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    print("Generating Text...")
    generated_sequence = model.generate(
        input_ids=input_ids,
        max_length=args.seq_length,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("Decoding Text...")
    completion = tokenizer.decode(generated_sequence[0][len(input_ids[0]) :], skip_special_tokens=True)

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_deepseek_instruct_pl_to_nl(
    prompt, user_input, model, tokenizer, args
):  # reference: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/README.md
    # select the correct in-context learning prompt based on the task
    # remove the last system prompt
    messages = prompt[:-1] + [{"role": "user", "content": user_input}]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    print("Generating Text...")
    generated_sequence = model.generate(
        input_ids=input_ids,
        max_length=args.seq_length,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("Decoding Text...")
    completion = tokenizer.decode(generated_sequence[0][len(input_ids[0]) :], skip_special_tokens=True)

    # post-processing
    processed_completion = post_processing_pl_to_nl(completion)
    return processed_completion


def get_completion_codellama(
    prompt, user_input, model, tokenizer, args
):  # prompt is ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP
    user_input = prompt + user_input
    output = generate_text(model, tokenizer, user_input, args)
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_codellama_fim(
    prompt, function_signature, function_body, model, tokenizer, args
):  # prompt is ONE_SHOT_FIM_HUMANEVAL or ONE_SHOT_FIM_MBPP
    return fill_in_middle(
        prompt=prompt,
        function_signature=function_signature,
        function_body=function_body,
        model=model,
        tokenizer=tokenizer,
        args=args,
        pre_token=" <PRE>",
        suf_token=" <SUF>",
        mid_token=" <MID>",
    )


def get_completion_starcoder(
    prompt, user_input, model, tokenizer, args
):  # prompt is ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP
    user_input = prompt + user_input
    output = generate_text(model, tokenizer, user_input.strip(), args)
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_starcoder_fim(
    prompt, function_signature, function_body, model, tokenizer, args
):  # prompt is ONE_SHOT_FIM_HUMANEVAL or ONE_SHOT_FIM_MBPP
    return fill_in_middle(
        prompt=prompt,
        function_signature=function_signature,
        function_body=function_body,
        model=model,
        tokenizer=tokenizer,
        args=args,
        pre_token="<fim_prefix>",
        suf_token="<fim_suffix>",
        mid_token="<fim_middle>",
    )


def get_completion_deepseek(
    prompt, user_input, model, tokenizer, args
):  # prompt is ONE_SHOT_HUMANEVAL or ONE_SHOT_MBPP
    user_input = prompt + user_input
    output = generate_text(model, tokenizer, user_input.strip(), args)
    completion = output[0]["generated_text"]

    # post-processing
    processed_completion = post_processing_nl_to_pl(completion)
    return processed_completion


def get_completion_deepseek_fim(
    prompt, function_signature, function_body, model, tokenizer, args
):  # prompt is ONE_SHOT_FIM_HUMANEVAL or ONE_SHOT_FIM_MBPP
    return fill_in_middle(
        prompt=prompt,
        function_signature=function_signature,
        function_body=function_body,
        model=model,
        tokenizer=tokenizer,
        args=args,
        pre_token="<｜fim▁begin｜>",
        suf_token="<｜fim▁hole｜>",
        mid_token="<｜fim▁end｜>",
    )


# EXAMPLE USAGE:
# python run_identity_chain_huggingface.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, help='Path to the model')
    parser.add_argument('--hf_dir', type=str, help='Path to the huggingface cache directory')
    parser.add_argument('--input_path', type=str, help='Path to the input file')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--chain_length', type=int, default=5, help='Number of steps in the Identity Chain')
    parser.add_argument('--seq_length', type=int, default=2048, help='max length of the sequence')
    parser.add_argument('--gen_length', type=int, default=None, help='max length of the generated sequence')
    parser.add_argument('--do_sample', action='store_true', help='whether to do sampling')
    parser.add_argument('--greedy_early_stop', action='store_true', help='whether to stop inference when fixed point')
    parser.add_argument('--temperature', type=float, default=0, help='temperature for sampling')
    parser.add_argument('--top_k', type=int, default=0, help='top k for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='top p for sampling')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='number of return sequences')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams for beam search')
    parser.add_argument('--use_int8', action='store_true', help='whether to use int8 quantization')
    parser.add_argument('--use_fp16', action='store_true', help='whether to use fp16 precision')
    parser.add_argument('--pass_only', action='store_true', help='whether to only pass the input to the next step')
    parser.add_argument('--mask_func_name', action='store_true', help='whether to mask the function name')
    parser.add_argument('--bootstrap_method', type=str, default='problem', help='method to bootstrap the chain')
    parser.add_argument('--resume_task_bs', type=int, default=0, help='task to resume at when bootstrapping')
    parser.add_argument('--resume_task_run', type=int, default=0, help='task to resume at')
    parser.add_argument('--skip_bootstrap', action='store_true', help='whether to skip the bootstrap stage')
    parser.add_argument('--version', type=str, default='v1', help='version of the identity chain')
    args = parser.parse_args()

    # set huggingface cache directory
    HF_HOME = args.hf_dir
    print("Loading model...")
    # if specified, use int8 quantization
    if args.use_int8:
        print("**********************************")
        print("**** Using 8-bit quantization ****")
        print("**********************************")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=HF_HOME,
        )
    # if specified, use fp16 precision
    elif args.use_fp16:
        print("**********************************")
        print("****** Using fp16 precision ******")
        print("**********************************")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=HF_HOME,
        )
    # otherwise, use default precision
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            cache_dir=HF_HOME,
        )

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.seq_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )

    # create output directory if not exists
    if not os.path.exists("../tmp"):
        os.makedirs("../tmp", exist_ok=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # unzip input file
    input_path = args.input_path
    input_file = input_path.split("/")[-1]
    g_unzip(f"{input_path}.gz", input_path)

    # for output path naming
    model_name = args.model_name_or_path.split("/")[-1]
    tmp = args.temperature
    len = args.chain_length
    bootstrap = "pb" if args.bootstrap_method == "problem" else "cb"
    pass_only = "po" if args.pass_only else "all"
    mask_name = "m" if args.mask_func_name else "um"
    greedy = "g" if args.greedy_early_stop else ""
    version = args.version

    # define the output path
    output_path = f"{args.output_dir}/IDChain_{model_name}_tmp{tmp}{greedy}_len{len}_{bootstrap}_{pass_only}_{mask_name}_{version}_{input_file}"

    # configure nl_2_pl/pl_2_nl functions based on the model
    # configure functions for starchat-beta
    if args.model_name_or_path == "HuggingFaceH4/starchat-beta":
        nl_2_pl_function = get_completion_starchat_nl_to_pl
        pl_2_nl_function = get_completion_starchat_pl_to_nl
    # configure functions for starcoder completion models
    elif args.model_name_or_path.startswith("bigcode/starcoder"):
        nl_2_pl_function = get_completion_starcoder
        pl_2_nl_function = get_completion_starcoder_fim
    # configure functions for codellama models
    elif "CodeLlama" in args.model_name_or_path:
        # configure functions for codellama-instruct chat models
        if "Instruct" in args.model_name_or_path:
            nl_2_pl_function = get_completion_codellama_instruct_nl_to_pl
            pl_2_nl_function = get_completion_codellama_instruct_pl_to_nl
        # configure functions for codellama completion models
        else:
            nl_2_pl_function = get_completion_codellama
            pl_2_nl_function = get_completion_codellama_fim
    # configure function for deepseekcoder models
    elif "deepseek-ai" in args.model_name_or_path:
        # configure functions for deepseekcoder chat models
        if "instruct" in args.model_name_or_path:
            nl_2_pl_function = get_completion_deepseek_instruct_nl_to_pl
            pl_2_nl_function = get_completion_deepseek_instruct_pl_to_nl
        # configure functions for deepseekcoder completion models
        else:
            nl_2_pl_function = get_completion_deepseek
            pl_2_nl_function = get_completion_deepseek_fim
    else:
        raise ValueError(f"Model {args.model_name_or_path} not supported")

    # configure nl_2_pl/pl_2_nl prompts based on the model and input dataset
    # configure prompts for HumanEvalPlus-Mini-v0.1.6 or v0.1.9
    if "EvalPlus-Mini" in args.input_path:
        # configure prompt for supported chat models
        if args.model_name_or_path in INSTRUCTION_MODELS:
            nl_2_pl_prompt = NL_2_PL_HUMANEVAL
            pl_2_nl_prompt = PL_2_NL_HUMANEVAL
        # configure prompt for supported completion models
        elif args.model_name_or_path in FOUNDATION_MODELS:
            # both nl_2_pl and pl_2_nl prompts are the same for completion models
            nl_2_pl_prompt = ONE_SHOT_HUMANEVAL
            pl_2_nl_prompt = ONE_SHOT_FIM_HUMANEVAL
        else:
            raise ValueError(f"Model {args.model_name_or_path} not supported")
    # configure prompts for MBPP-S_test
    elif args.input_path.endswith("MBPP-S_test_reformatted.jsonl"):
        # configure prompt for supported chat models
        if args.model_name_or_path in INSTRUCTION_MODELS:
            nl_2_pl_prompt = NL_2_PL_MBPP
            pl_2_nl_prompt = PL_2_NL_MBPP
        # configure prompt for supported completion models
        elif args.model_name_or_path in FOUNDATION_MODELS:
            # both nl_2_pl and pl_2_nl prompts are the same for completion models
            nl_2_pl_prompt = ONE_SHOT_MBPP
            pl_2_nl_prompt = ONE_SHOT_FIM_MBPP
        else:
            raise ValueError(f"Model {args.model_name_or_path} not supported")
    else:
        raise ValueError(f"Input file {args.input_path} not supported")

    # for DEBUGging
    print("--------- Prompt Configuration -----------")
    print(nl_2_pl_prompt)
    print(pl_2_nl_prompt)
    print("-----------------------------------------")

    # create the identity chain
    my_chain = IdentityChain(
        model=model,
        tokenizer=tokenizer,
        args=args,
        input_path=input_path,
        output_path=output_path,
        get_model_response_NL_to_PL=nl_2_pl_function,
        get_model_response_PL_to_NL=pl_2_nl_function,
        prompt_NL_to_PL=nl_2_pl_prompt,
        prompt_PL_to_NL=pl_2_nl_prompt,
        bootstrap_method="problem",
        length=args.chain_length,
    )
    print("-----------------------------------------")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print("-----------------------------------------")
    input("Please Confirm the Identity Chain Setup. Press 'Enter' to Continue...")

    # if resume_task_run != 0 or skip_bootstrap == True, then we don't need to bootstrap
    if (args.resume_task_run == 0) and (not args.skip_bootstrap):
        my_chain.bootstrap(resume_task=args.resume_task_bs)

    my_chain.run(
        resume_task=args.resume_task_run,
        resume_step=1,
        pass_only=args.pass_only,
        mask_func_name=args.mask_func_name,
        greedy_early_stop=args.greedy_early_stop,
    )


if __name__ == "__main__":
    main()
