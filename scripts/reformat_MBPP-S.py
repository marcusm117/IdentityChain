# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json
import os
import re

# Internal Modules
from identitychain.utils import g_unzip


# extract function signature from a canonical solution
# note that this only works for variations of the MBPP dataset
def extract_func_signature(canonical_sol):
    # find all occurrences of "def "
    res = [i for i in range(len(canonical_sol)) if canonical_sol.startswith("def ", i)]
    # pick the last one if it follows a newline, otherwise pick the first one
    if canonical_sol[res[-1] - 1] == "\n":
        start_position = res[-1]
    else:
        start_position = res[0]

    # find the end position of the function signature
    canonical_sol = canonical_sol[start_position:]
    end_position_candidate1 = canonical_sol.find("):")
    end_position_candidate2 = canonical_sol.find(") :")

    # choose candidate2 if candidate1 is not found
    if end_position_candidate1 == -1:
        end_position = end_position_candidate2 + 3
    # choose candidate1 if candidate2 is not found
    elif end_position_candidate2 == -1:
        end_position = end_position_candidate1 + 2
    else:
        # choose the smaller one if both are found
        if end_position_candidate1 < end_position_candidate2:
            end_position = end_position_candidate1 + 2
        else:
            end_position = end_position_candidate2 + 3

    return canonical_sol[:end_position]


# extract function name from a function signature
# note that this only works for variations of the MBPP dataset
def extract_func_name(func_signature):
    start_position = func_signature.find("def ") + 4
    end_position = func_signature.find("(")
    return func_signature[start_position:end_position].strip()


# rename the field "prompt" into "problem" to avoid confusion with the instrcutrion/few-shot prompt for the model
# reformat the field "problem" into HumanEval style: function signature + problem description in the docstring
# extract the function signature from the field "code", store in the field "function_signature"
# extract function name from function signature, store in the field "function_name"
# remove the function signature from the field "code", rename it to "canonical_solution"
def reformat_fields(output_path, MBPP_S_tasks):
    with open(output_path, "w", encoding="utf-8") as writer:
        for MBBP_S_task in MBPP_S_tasks:
            MBBP_S_task = json.loads(MBBP_S_task)

            # extract function signature from the field "code"
            func_signature = extract_func_signature(MBBP_S_task["code"])

            # rename the field "prompt" into "problem", and reformat it
            reformatted_task = {
                "task_id": MBBP_S_task["task_id"],
                "problem": func_signature + '\n    \"\"\" ' + MBBP_S_task["prompt"] + '\n    \"\"\"\n',
                "contract": "",
            }

            # add the fields "function_name" and "function_signature"
            reformatted_task["function_name"] = extract_func_name(func_signature)
            reformatted_task["function_signature"] = func_signature

            # write the reformatted task to the output file
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# rename the field "code" into "canonical_solution" for clarity
# reformat "canonical_solution" so that all indentation is exactly 4 spaces
def reformat_canonical_solution(output_path, MBPP_S_tasks):
    # read previously reformatted tasks
    with open(output_path, "r", encoding="utf-8") as reader:
        reformatted_tasks = reader.readlines()

    # write new changes to the reformatted tasks
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in range(len(MBPP_S_tasks)):
            MBBP_S_task = json.loads(MBPP_S_tasks[i])
            reformatted_task = json.loads(reformatted_tasks[i])
            task_id = reformatted_task["task_id"]

            # reformat the canonical solution
            func_signature = reformatted_task["function_signature"]
            canonical_sol = MBBP_S_task["code"]
            # remove the import statements, add back inside the function body later
            import_statements = []
            position = canonical_sol.find(func_signature)
            if position != 0:
                import_str = canonical_sol[:position]
                canonical_sol = canonical_sol.replace(import_str, "")
                assert canonical_sol[: len(func_signature)] == func_signature  # for debugging
                import_statements = import_str.splitlines()
                import_statements = [statement.rstrip() for statement in import_statements]
            # remove the function signature
            canonical_sol = canonical_sol.replace(func_signature, "")
            # remove unnecessary leading spaces
            canonical_sol = canonical_sol.lstrip(" ")
            # remove all comments
            canonical_sol = re.sub(r"#.*\n", "", canonical_sol)
            # replace tabs with 4 spaces
            canonical_sol = canonical_sol.replace("\t", "    ")
            # replace 3 or less spaces with 4 spaces
            pattern = r"^\n {1,3}"
            match = re.match(pattern, canonical_sol)
            if match:
                assert match.start() == 0  # for debugging
                canonical_sol = canonical_sol.replace(canonical_sol[1 : match.end()], "    ")
            # replace 5 or more spaces with 4 spaces
            pattern = r"^\n {5,}"
            match = re.match(pattern, canonical_sol)
            if match:
                assert match.start() == 0  # for debugging
                canonical_sol = canonical_sol.replace(canonical_sol[1 : match.end()], "    ")
            # add back the import statements, if any
            if import_statements:
                canonical_sol = "    " + "\n    ".join(import_statements) + "\n" + canonical_sol
            # remove unnecessary ";", spaces, and newlines at the end, if any
            while canonical_sol[-1] in [";", " ", "\n"]:
                canonical_sol = canonical_sol[:-1]
            # add one trailing newline back
            canonical_sol += "\n"

            # hanlde special cases
            if task_id == 223:
                reformatted_task["function_name"] = "is_majority"
                reformatted_task["function_signature"] = "def is_majority(arr, n, x):\n"
                reformatted_task["problem"] = reformatted_task["function_signature"] + reformatted_task["problem"][38:]
                canonical_sol = """    def binary_search(arr, low, high, x):
        if high >= low:
            mid = (low + high)//2
            if (mid == 0 or x > arr[mid-1]) and (arr[mid] == x):
                return mid
            elif x > arr[mid]:
                return binary_search(arr, (mid + 1), high, x)
            else:
                return binary_search(arr, low, (mid -1), x)
        return -1

    i = binary_search(arr, 0, n-1, x)
    if i == -1:
            return False
    if ((i + n//2) <= (n -1)) and arr[i + n//2] == x:
            return True
    else:
            return False\n"""
            if task_id == 230:
                canonical_sol = "    str2 = str1.replace(' ', char)\n    return str2"
            if task_id == 640:
                canonical_sol = (
                    "    import re\n\n    for item in items:\n        return (re.sub(r\" ?\\([^)]+\\)\", \"\", item))\n"
                )

            # rename the field "code" into "canonical_solution"
            reformatted_task["canonical_solution"] = canonical_sol

            # add trailnig newline to function signature
            reformatted_task["function_signature"] += "\n"

            # write the reformatted task to the output file
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# reformat "tests" by reporting the output of each test case when assertion fails
# add the field "atol"
def reformat_tests(output_path, MBPP_S_tasks):
    # read previously reformatted tasks
    with open(output_path, "r", encoding="utf-8") as reader:
        reformatted_tasks = reader.readlines()

    # write new changes to the reformatted tasks
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in range(len(MBPP_S_tasks)):
            MBBP_S_task = json.loads(MBPP_S_tasks[i])
            reformatted_task = json.loads(reformatted_tasks[i])
            task_id = reformatted_task["task_id"]

            # reformat tests
            tests = MBBP_S_task["test_list"]
            for j in range(len(tests)):
                # add import statements if necessary
                import_stmt = ""
                if "sys." in tests[j]:
                    import_stmt += "import sys\n"
                if "math." in tests[j]:
                    import_stmt += "import math\n"

                # capture the function call stmt, which will be evlauted as the output
                func_name = reformatted_task["function_name"]
                assert tests[j].count(func_name) == 1  # for debugging
                start_position = tests[j].find(f"{func_name}")
                end_position = tests[j].find("==")
                if end_position == -1:
                    end_position = tests[j].find(")") + 1
                func_call_stmt = tests[j][start_position:end_position].strip()
                # remove necessary spaces
                while func_call_stmt[len(func_name)] == " ":
                    func_call_stmt = func_call_stmt[: len(func_name)] + func_call_stmt[len(func_name) + 1 :]
                while func_call_stmt[len(func_name) + 1] == " ":
                    func_call_stmt = func_call_stmt[: len(func_name) + 1] + func_call_stmt[len(func_name) + 2 :]

                # hanlde special cases
                if task_id in {2, 7, 111, 140, 232, 769}:
                    func_call_stmt = func_call_stmt[:-1]
                if task_id == 98:
                    func_call_stmt += ")"

                # capture output and report it when assertion fails
                capture_output = f"test_out = {func_call_stmt}\n"
                assertion_stmt = tests[j].replace(func_call_stmt, "test_out") + r", f'outputs {test_out}'"

                # reformat test
                tests[j] = import_stmt + capture_output + assertion_stmt

            # add the fields "tests" and "atol"
            reformatted_task["tests"] = tests
            reformatted_task["atol"] = ""
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# EXAMPLE USAGE:
# python reformat_MBPP-S.py --output_path ../data/MBPP-S_test_reformatted.jsonl
# python reformat_MBPP-S.py --input_path ../data/MBPP-S/MBPP-S.jsonl.gz --output_path ../data/MBPP-S_reformatted.jsonl
# note that this script only works for the MBPP-S (sanitized) dataset, NOT the full MBPP dataset
# for more information, see the following HuggingFace Dataset:
# https://huggingface.co/datasets/mbpp/viewer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/MBPP-S/MBPP-S_test.jsonl.gz")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # uncompressed the data
    # note that, for model  evluation, we only need the test split of the MBPP-S dataset
    # however, this script works for the entire MBPP-S dataset as well
    # if you want to reformat the entrie dataset or other splits, specify the input_path and output_path accordingly
    uncompressed_path = args.input_path[:-3].replace("MBPP-S/", "")
    g_unzip(args.input_path, uncompressed_path)

    # read from the test split of the MBPP-S (sanitized) dataset
    # for more information, see the following GitHub Repo:
    # https://github.com/google-research/google-research/tree/master/mbpp
    with open(uncompressed_path, "r", encoding="utf-8") as reader:
        MBPP_S_tasks = reader.readlines()

    # reformat the data
    reformat_fields(args.output_path, MBPP_S_tasks)
    reformat_canonical_solution(args.output_path, MBPP_S_tasks)
    reformat_tests(args.output_path, MBPP_S_tasks)

    # remove the uncompressed input data
    os.remove(uncompressed_path)


if __name__ == "__main__":
    main()
