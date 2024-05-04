# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json
import os

# Internal Modules
from identitychain.utils import g_unzip


# extract function signature from a problem descriptions
# note that this only works for variations of the HumanEval dataset
def extract_func_signature(prompt):
    # find all occurrences of three consecutive double quotes
    res = [i for i in range(len(prompt)) if prompt.startswith('"""', i)]

    # if res is empty, check for both single quotes
    if not res:
        res = [i for i in range(len(prompt)) if prompt.startswith("'''", i)]

    # get the end position of the function signature
    end_position = res[-2] - 4

    return prompt[:end_position]


# fix typos, fix wrong/no examples, and remove unnecessary leading newlines
# rename the field "prompt" into "problem" to avoid confusion with the instrcutrion/few-shot prompt for the model
def reformat_problem_descriptions(output_path, EvalPlus_tasks, HEE_tasks):
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in range(len(HEE_tasks)):
            EvalPlus_task = json.loads(EvalPlus_tasks[i])
            HEE_task = json.loads(HEE_tasks[i])

            # for HumanEval/75, 95, 116, 163, we choose the problem descriptions from EvalPlus
            # otherwise, we choose the problem descriptions from Human-Eval-Enhanced (HEE)
            # for more information, see the following issue and PR:
            # https://github.com/evalplus/evalplus/issues/12
            # https://github.com/openai/human-eval/pull/23
            # however, for HumanEval/116, one of the example still conflicts the problem description, we keep it
            # only to stay consistent with the original HumanEval dataset
            if i in {75, 95, 116, 163}:
                problem = EvalPlus_task["prompt"]
                problem = problem[1:]  # remove unnecessary leading newlines
            else:
                problem = HEE_task["prompt"]

            # rename the field "prompt" into "problem"
            reformatted_task = {
                "task_id": EvalPlus_task["task_id"],
                "problem": problem,
                "contract": EvalPlus_task["contract"],
            }

            # write the reformatted task to the output file
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# rename the field "entry_point" into "function_name" for clarity
# extract function signature from problem descriptions, store in the field "function_signature"
def reformat_fields(output_path, EvalPlus_tasks):
    # read previously reformatted tasks
    with open(output_path, "r", encoding="utf-8") as reader:
        reformatted_tasks = reader.readlines()

    # write new changes to the reformatted tasks
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in range(len(EvalPlus_tasks)):
            EvalPlus_task = json.loads(EvalPlus_tasks[i])
            reformatted_task = json.loads(reformatted_tasks[i])

            # rename the field "entry_point" into "function_name"
            # add the field "function_signature"
            reformatted_task["function_name"] = EvalPlus_task["entry_point"]
            reformatted_task["function_signature"] = extract_func_signature(reformatted_task["problem"])
            reformatted_task["canonical_solution"] = EvalPlus_task["canonical_solution"]

            # correct invalid contract for HumanEvalPlus-Mini/99, for HumanEvalPlus-Mini-v0.1.9 only
            if (
                reformatted_task["task_id"] == "HumanEval/99"
                and output_path == "../data/EvalPlus-Mini-v0.1.9_reformatted.jsonl"
            ):
                reformatted_task["contract"] = (
                    "\n    try: # $_CONTRACT_$\n        assert isinstance(value, str) "
                    "# $_CONTRACT_$\n        value = float(value) # $_CONTRACT_$\n    except: # $_CONTRACT_$\n        "
                    "raise Exception(\"invalid inputs\") # $_CONTRACT_$\n    import math # $_CONTRACT_$\n    "
                    "assert not (math.isinf(value) or math.isnan(value)), \"invalid inputs\" # $_CONTRACT_$\n"
                )

            # write the reformatted task to the output file
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# use the canonical solution provided by EvalPlus
# to generate outputs for each test case input in EvalPlus-Mini
# reformat them in into a List of Dicts, for example,
#    "tests": [
#        {
#            "input": "'(()()) ((())) () ((())()())'",
#            "output": "['(()())', '((()))', '()', '((())()())']",
#        },
#        {
#            "input": "'() (()) ((())) (((())))'",
#            "output": "['()', '(())', '((()))', '(((())))']",
#        },
#        {
#           "input": "'(()(())((())))'",
#           "output": "['(()(())((())))']"
#        },
#    ]
def reformat_tests(output_path, EvalPlus_tasks):
    # read previously reformatted tasks
    with open(output_path, "r", encoding="utf-8") as reader:
        reformatted_tasks = reader.readlines()

    # write new changes to the reformatted tasks
    with open(output_path, "w", encoding="utf-8") as writer:
        for i in range(len(EvalPlus_tasks)):
            EvalPlus_task = json.loads(EvalPlus_tasks[i])
            reformatted_task = json.loads(reformatted_tasks[i])

            # reformat the inputs
            task_id = EvalPlus_task["task_id"]
            # remove redundant tests in HumanEvalPlus-Mini/15
            if task_id == "HumanEval/15":
                inputs = EvalPlus_task["base_input"] + [[1], [100]]
            # remove Time Complexity tests in HumanEvalPlus-Mini/83, 100, 130, 139
            elif task_id in {"HumanEval/83", "HumanEval/100", "HumanEval/130", "HumanEval/139"}:
                inputs = EvalPlus_task["base_input"] + EvalPlus_task["plus_input"][:-1]
            # remove an invalid input in HumanEvalPlus-Mini/116
            elif task_id == "HumanEval/116":
                EvalPlus_task["base_input"].pop(1)
                inputs = EvalPlus_task["base_input"] + EvalPlus_task["plus_input"]
            # remove an invalid input in HumanEvalPlus-Mini/1, for HumanEvalPlus-Mini-v0.1.9 only
            elif task_id == "HumanEval/1" and output_path == "../data/EvalPlus-Mini-v0.1.9_reformatted.jsonl":
                EvalPlus_task["plus_input"].pop(1)
                EvalPlus_task["plus_input"].pop(-2)
                inputs = EvalPlus_task["base_input"] + EvalPlus_task["plus_input"]
            # remove an invalid input in HumanEvalPlus-Mini/28, for HumanEvalPlus-Mini-v0.1.9 only
            elif task_id == "HumanEval/28" and output_path == "../data/EvalPlus-Mini-v0.1.9_reformatted.jsonl":
                EvalPlus_task["plus_input"].pop(1)
                inputs = EvalPlus_task["base_input"] + EvalPlus_task["plus_input"]
            # otherwise, concatenate the base_inputs and plus_inputs
            else:
                inputs = EvalPlus_task["base_input"] + EvalPlus_task["plus_input"]

            # construct the tests List, where each element is a Dict
            tests = []
            for input in inputs:
                # reformat the input as a string
                input_str = f"{str(input)[1:-1]}"
                tests.append({"input": input_str, "output": ""})

            # concatenate the problem description, contract, and canonical solution
            complete_code = (
                reformatted_task["problem"] + reformatted_task["contract"] + reformatted_task["canonical_solution"]
            )

            # for debugging, uncomment the following line
            print(reformatted_task["task_id"])
            print(complete_code)

            # get corresponding outputs for each test case input
            for test in tests:
                # EvalPlus canonical solutions can be trusted, okay to use exec() here
                exec_globals = {}
                exec(complete_code, exec_globals)
                # fetch the function from exec_globals
                func = exec_globals[f"{reformatted_task['function_name']}"]
                # get the input from the string
                input = eval(test["input"])
                # get corresponding output
                if type(input) is tuple:
                    output = func(*input)
                else:
                    print(input)
                    output = func(input)
                # store the output to the test Dict
                if type(output) is str:
                    # handle escape characters
                    output = output.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                    test["output"] = f"'{output}'"
                else:
                    test["output"] = str(output)

                # for debugging, uncomment the following line
                # print(test)

            # add the tests List to the reformatted task
            reformatted_task["tests"] = tests
            reformatted_task["atol"] = EvalPlus_task["atol"]
            writer.write(json.dumps(reformatted_task) + "\n")
            writer.flush()


# EXAMPLE USAGE:
# python reformat_EvalPlus.py --evalplus_path ../data/HumanEval/HumanEvalPlus-Mini-v0.1.10.jsonl.gz
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalplus_path", type=str, required=True)
    args = parser.parse_args()

    # uncompressed the input data
    uncompressed_file_name = args.evalplus_path[18:-3]
    uncompressed_path = f"../data/{uncompressed_file_name}"
    g_unzip(args.evalplus_path, uncompressed_path)
    g_unzip("../data/HumanEval/human-eval-enhanced-202307.jsonl.gz", "../data/human-eval-enhanced-202307.jsonl")

    # read from the HumanEvalPlus-Mini-v0.1.6 (EvalPlus) and human-eval-enhanced-202307 (HEE) datasets
    # for more information, see the following GitHub Repos:
    # https://github.com/evalplus/evalplus
    # https://github.com/marcusm117/human-eval-enhanced
    with open(uncompressed_path, "r", encoding="utf-8") as reader:
        EvalPlus_tasks = reader.readlines()
    with open("../data/human-eval-enhanced-202307.jsonl", "r", encoding="utf-8") as reader:
        HEE_tasks = reader.readlines()

    # reformat the data
    output_path = "../data/" + uncompressed_file_name[5:-6] + "_reformatted.jsonl"
    reformat_problem_descriptions(output_path, EvalPlus_tasks, HEE_tasks)
    reformat_fields(output_path, EvalPlus_tasks)
    reformat_tests(output_path, EvalPlus_tasks)

    # remove the uncompressed input data
    os.remove(uncompressed_path)
    os.remove("../data/human-eval-enhanced-202307.jsonl")


if __name__ == "__main__":
    main()
