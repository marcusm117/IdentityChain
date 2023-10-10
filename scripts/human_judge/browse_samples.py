# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json
import os

# Internal Modules
from identitychain.utils import g_unzip


# EXAMPLE USAGE:
# python browse_samples.py --start 0
def main():
    # unzip the raw experiment results to a temporary directory
    tmp_dir = "../../tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    input_file = "IDChain_gpt-3.5-turbo-0613_tmp0_len1_pb_all_m_human-judge_EvalPlus-Mini-v0.1.6_reformatted.jsonl"
    input_path = f"{tmp_dir}/{input_file}"
    g_unzip(f"../../data/experiments/{input_file}.gz", input_path)

    # you can used this script to browse results of other Human Judgment experiments results
    # for that purpose, you can specify the command line argument --input_path
    # by default, we will browse the samples of the above experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=input_path)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    # read the Identity Chain results
    with open(input_path, "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    for task in tasks[args.start :]:
        input("Press Enter to continue...")
        task = json.loads(task)
        function_name = task["function_name"]
        function_signature = task["function_signature"]
        masked_function_signature = function_signature.replace(function_name, "func")
        print("*****************************************")
        print(f"Task: {task['task_id']}")
        print("-----------------------------------------")
        print("++ PL, Implementation of a Function ++:")
        print("-----------------------------------------")
        print(f"\n{masked_function_signature}\n{task['PL_0']}\n")
        print("-----------------------------------------")
        print("++ NL, Docstring of a Function ++:")
        print("-----------------------------------------")
        print(f"\n{masked_function_signature}\n{task['NL_1']}\n")
        print("*****************************************")


if __name__ == "__main__":
    main()
