# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import os
import time

# External Modules
import google.generativeai as genai

# Internal Modules
from identitychain import IdentityChain
from identitychain.utils import g_unzip


# add your OpenAI API key here
genai.configure(api_key="YOUR_API_KEY")


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


# convert the structured prompts to strings
NL_2_PL_HUMANEVAL_STR = (
    NL_2_PL_HUMANEVAL[0]["content"]
    + "User Input:\n"
    + NL_2_PL_HUMANEVAL[1]["content"]
    + "Your Output:\n"
    + NL_2_PL_HUMANEVAL[2]["content"]
    + NL_2_PL_HUMANEVAL[3]["content"]
)

PL_2_NL_HUMANEVAL_STR = (
    PL_2_NL_HUMANEVAL[0]["content"]
    + "User Input:\n"
    + PL_2_NL_HUMANEVAL[1]["content"]
    + "Your Output:\n"
    + PL_2_NL_HUMANEVAL[2]["content"]
    + PL_2_NL_HUMANEVAL[3]["content"]
)

NL_2_PL_MBPP_STR = (
    NL_2_PL_MBPP[0]["content"]
    + "User Input:\n"
    + NL_2_PL_MBPP[1]["content"]
    + "Your Output:\n"
    + NL_2_PL_MBPP[2]["content"]
    + NL_2_PL_MBPP[3]["content"]
)

PL_2_NL_MBPP_STR = (
    PL_2_NL_MBPP[0]["content"]
    + "User Input:\n"
    + PL_2_NL_MBPP[1]["content"]
    + "Your Output:\n"
    + PL_2_NL_MBPP[2]["content"]
    + PL_2_NL_MBPP[3]["content"]
)


# get completion from a Google model
def get_google_model_chat(
    prompt,
    user_input,
    model,
    tokenizer,
    args,
):
    # select the correct in-context learning prompt based on the task
    messages = prompt + user_input

    # get response from OpenAI
    try:
        # get the response
        response = model.generate_content(messages)
        # print(response.candidates)
        # extract the response text content
        response_content = response.candidates[0].content.parts[0].text
        # if the API is unstable, consider sleeping for a short period of time after each request
        time.sleep(0.5)
        return response_content

    # when encounter APIError, sleep for 5 or specified seconds and try again
    except Exception as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"{error}. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_google_model_chat(prompt, user_input, model, tokenizer, args)


# EXAMPLE USAGE:
# python run_identity_chain_google.py
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

    # configure prompts for HumanEvalPlus-Mini-v0.1.6
    if args.input_path.endswith("EvalPlus-Mini-v0.1.6_reformatted.jsonl"):
        nl_2_pl_prompt = NL_2_PL_HUMANEVAL_STR
        pl_2_nl_prompt = PL_2_NL_HUMANEVAL_STR
    # configure prompts for MBPP-S_test
    elif args.input_path.endswith("MBPP-S_test_reformatted.jsonl"):
        nl_2_pl_prompt = NL_2_PL_MBPP_STR
        pl_2_nl_prompt = PL_2_NL_MBPP_STR
    else:
        raise ValueError(f"Input file {args.input_path} not supported")

    # for debugging
    print("--------- Prompt Configuration -----------")
    print(nl_2_pl_prompt)
    print(pl_2_nl_prompt)
    print("-----------------------------------------")

    # create generation config
    generation_config = genai.GenerationConfig(
        candidate_count=1,
        max_output_tokens=args.gen_length,
        temperature=args.temperature,
    )
    # creat the model object
    model_obj = genai.GenerativeModel(args.model_name_or_path, generation_config=generation_config)

    # create an Identity Chain
    my_chain = IdentityChain(
        model=model_obj,
        tokenizer=None,
        args=args,
        input_path=input_path,
        output_path=output_path,
        get_model_response_NL_to_PL=get_google_model_chat,
        get_model_response_PL_to_NL=get_google_model_chat,
        prompt_NL_to_PL=nl_2_pl_prompt,
        prompt_PL_to_NL=pl_2_nl_prompt,
        bootstrap_method=args.bootstrap_method,
        length=args.chain_length,
    )
    print("-----------------------------------------")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    print("-----------------------------------------")
    input("Please Confirm the Identity Chain Setup. Press 'Enter' to Continue...")

    # uncomment the code below to bootstrap the chain
    # if resume_task_run != 0 or skip_bootstrap == True, then we don't need to bootstrap
    if (args.resume_task_run == 0) and (not args.skip_bootstrap):
        my_chain.bootstrap(resume_task=args.resume_task_bs)

    # if you already have a bootstraped chain, ignore the line above
    # the following line will resume the chain from your specified task and step
    my_chain.run(
        resume_task=args.resume_task_run,
        resume_step=1,
        pass_only=args.pass_only,
        mask_func_name=args.mask_func_name,
        greedy_early_stop=args.greedy_early_stop,
    )


if __name__ == "__main__":
    main()
