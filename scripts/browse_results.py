# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json


# EXAMPLE USAGE:
# python browse_results.py --input_path
# ../tmp/starcoderbase-1b/IDChain_starcoderbase-1b_tmp0.0g_len5_pb_all_m_v1_EvalPlus-Mini-v0.1.6_reformatted.jsonl
# --chain_length 5 --start 0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--chain_length", type=int, default=1)
    args = parser.parse_args()

    # read the Identity Chain results
    with open(args.input_path, "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    for task in tasks[args.start :]:
        input("Press Enter to continue...")
        task = json.loads(task)
        print("*****************************************")
        print(f"Task: {task['task_id']}")
        print("-----------------------------------------")
        print(f"++ Description NL_0 & Solution PL_0 ++:\n\n{task['NL_0']}{task['contract']}\n{task['PL_0']}\n")
        print("-----------------------------------------")
        print(f"++ Tests ++:\n\n{task['tests']}")
        print("-----------------------------------------")
        print(f"++ Solution PL_0 Test Result ++:\n\n{task['PL_0_results']}\n")
        for i in range(1, args.chain_length + 1):
            if task[f"NL_{i}"] == "NA":
                break
            print("-----------------------------------------")
            print(
                f"++ Description NL_{i} & Solution PL_{i} ++:\n\n{task['function_signature']}"
                + task[f"NL_{i}"]
                + "\n"
                + task[f"PL_{i}"]
                + "\n"
            )
            print("-----------------------------------------")
            print(f"++ Solution PL_{i} Test Result ++:\n\n" + str(task[f"PL_{i}_results"]) + "\n")
        print("*****************************************")


if __name__ == "__main__":
    main()
