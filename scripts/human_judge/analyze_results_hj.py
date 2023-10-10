# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json
import os

# External Modules
from codebleu import calc_codebleu
from evaluate import load
import pandas as pd
from tqdm import tqdm

# Internal Modules
from identitychain.utils import g_unzip


# summarize the test results of a single task
def summarize_test_results(PL_i_results):
    # if not defined, return "NA"
    if PL_i_results == "NA":
        return PL_i_results
    # if some test cases didn't pass, return the first Error
    for result in PL_i_results:
        if "Error" in result:
            return result
    return "Passed"


# calculate Exact Match
def calculate_EM(reference, candidate):
    if candidate == reference:
        return True
    else:
        return False


# calculate BLEU score
def calculate_BLEU(reference, candidate):
    metric = load("bleu")
    result = metric.compute(references=[reference], predictions=[candidate])
    return result


# calculate chrF score
def calculate_chrF(reference, candidate):
    metric = load("chrf")
    result = metric.compute(references=[reference], predictions=[candidate])
    return result


# calculate METEOR score
# note that currently (09/2023) there's a bug with the 'meteor' metric in the evaluate package
# for more information, see:
# https://github.com/huggingface/evaluate/issues/480
@DeprecationWarning
def calculate_METEOR(reference, candidate):
    metric = load("meteor")
    result = metric.compute(references=[reference], predictions=[candidate])
    return result


# calculate ROUGE score
def calculate_ROUGE(reference, candidate):
    metric = load('rouge')
    result = metric.compute(references=[reference], predictions=[candidate])
    return result


# calculate BERTScore
# for our experiments, the natural language descriptions are in English
# but it can be customized to other natural languages
def calculate_BERTScore(reference, candidate):
    metric = load("bertscore")
    result = metric.compute(references=[reference], predictions=[candidate], lang="en")
    return result


# calculate CodeBLEU score
# for our experiments, the programming language is Python
# but it can be customized to other programming languages
def calculate_CodeBLEU(reference, candidate):
    result = calc_codebleu(
        references=[reference], predictions=[candidate], lang="python", weights=(0.25, 0.25, 0.25, 0.25)
    )
    return result


# calculate Test Ouput Match (TOM) score
# TOM = number of matched test outputs / total number of test cases
# for example, if 2 test outputs are
#   [Passed, Passed, AssertionError: outputs 10, AssertionError: outputs 4, ValueError: message]
#   [Passed, Passed, AssertionError: outputs 10, AssertionError: outputs 7, TypeError: message]
# there are 3 matched test outputs, so TOM = 3 / 5 = 0.6
# note that we are also matching the Error Messages, if Error Messages are different, then still not matched
def calculate_TOM(reference_test_outputs, candidate_test_outputs, ignore_Timeout=True):
    # initialize match_res to store the matching result for each test case,
    # and match_count to store the number of matched test outputs
    match_res = []

    for i in range(len(reference_test_outputs)):
        # if ignore_Timeout is True, ignore all test outputs that are "Time Out"
        if ignore_Timeout and (reference_test_outputs[i] == "Time Out" or candidate_test_outputs[i] == "Time Out"):
            continue
        # if not matched, record 0; otherwise record 1
        if reference_test_outputs[i] != candidate_test_outputs[i]:
            match_res.append(0)
        else:
            match_res.append(1)

    # if all test outputs are "Time Out", then the TOM score is 0.0
    if len(match_res) == 0:
        return {"tom": 0.0, "match_res": match_res}

    # compute TOM score
    TOM_score = sum(match_res) / len(match_res)
    return {"tom": TOM_score, "match_res": match_res}


# EXAMPLE USAGE:
# python analyze_results_hj.py --chain_length 1
def main():
    # unzip the raw experiment results to a temporary directory
    tmp_dir = "../../tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    input_file = "IDChain_gpt-3.5-turbo-0613_tmp0_len1_pb_all_m_human-judge_EvalPlus-Mini-v0.1.6_reformatted.jsonl"
    input_path = f"{tmp_dir}/{input_file}"
    g_unzip(f"../../data/experiments/{input_file}.gz", input_path)

    # you can used this script to analyze results of other Human Judgment experiments results
    # for that purpose, you can specify the command line argument --input_path
    # by default, we will browse the samples of the above experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=input_path)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--chain_length", type=int, default=1)
    parser.add_argument("--no_mask", action="store_true")
    args = parser.parse_args()

    # check if chain_length is non-negative
    if args.chain_length < 0:
        raise ValueError("--chain_length must be non-negative.")

    # read the Identity Chain results
    with open(input_path, "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    # load raw test results to a DataFrame raw, and
    # calculate trajectory statistics for each task
    raw = pd.DataFrame()
    traj = pd.DataFrame()

    # iterate over each task
    for idx, task in enumerate(tqdm(tasks)):
        task = json.loads(task)
        # fetch results from task
        task_id = task["task_id"]
        # create row to add to raw and traj
        row_raw = {"task_id": task_id}
        row_traj = {"task_id": task_id}

        # summarize results for each tasks
        for i in range(args.chain_length + 1):
            PL_i_results = task[f"PL_{i}_results"]
            # get summary of results
            PL_i_results_summary = summarize_test_results(PL_i_results)
            # update row for DataFrame raw
            row_raw[f"PL_{i} Results"] = PL_i_results
            row_raw[f"PL_{i} Results Summary"] = PL_i_results_summary

        # add row_raw to the DataFrame raw
        raw = pd.concat([raw, pd.DataFrame([row_raw])], ignore_index=True)

        # calculate trajectory statistics for each task
        for i in range(args.chain_length):
            PL_i_results_summary = raw.loc[idx][f"PL_{i} Results Summary"]
            PL_i_plus_1_results_summary = raw.loc[idx][f"PL_{i+1} Results Summary"]

            # get summary of trajectory
            summary = ""
            if PL_i_results_summary == "Passed":
                if PL_i_plus_1_results_summary == "Passed":
                    summary = "Pass-Pass"
                else:
                    summary = "Pass-Fail"
            else:
                if PL_i_plus_1_results_summary == "Passed":
                    summary = "Fail-Pass"
                else:
                    summary = "Fail-Fail"

            # process NL_0 before calculating metrics
            # remove the function signature from NL_0
            if i == 0:
                ref_nl = task[f"NL_{i}"].replace(task["function_signature"], "")
                # replace function name with "func" the parameter if we masked the function name running IdentityChain
                # by default, we assume it's masked, but if --no_mask is specified, then we do nothing
                if not args.no_mask:
                    ref_nl = ref_nl.replace(task["function_name"], "func")
            else:
                ref_nl = task[f"NL_{i}"]

            # calculate stats of trajectory
            em_NL = calculate_EM(ref_nl, task[f"NL_{i+1}"])
            em_PL = calculate_EM(task[f"PL_{i}"], task[f"PL_{i+1}"])
            # no need to calculate NL metrics if EM-NL is True
            if em_NL:
                bleu_NL = 1.0
                chrF_NL = 1.0
                rouge_1_NL = 1.0
                rouge_2_NL = 1.0
                rouge_L_NL = 1.0
                bertscore_NL = 1.0
            else:
                # calculate BLEU-NL
                bleu_NL = calculate_BLEU(ref_nl, task[f"NL_{i+1}"])["bleu"]
                # calculate chrF-NL
                chrF_NL = calculate_chrF(ref_nl, task[f"NL_{i+1}"])["score"]
                # calculate ROUGE-1-NL, ROUGE-2-NL, ROUGE-L-NL
                rouge_res = calculate_ROUGE(ref_nl, task[f"NL_{i+1}"])
                rouge_1_NL = rouge_res["rouge1"]
                rouge_2_NL = rouge_res["rouge2"]
                rouge_L_NL = rouge_res["rougeL"]
                # calculate BERTScore-NL
                bertscore_NL = calculate_BERTScore(ref_nl, task[f"NL_{i+1}"])["f1"][0]

            # no need to calculate PL metrics if EM-PL is True
            if em_PL:
                codebleu = 1.0
                pfm = 1.0
                tom = 1.0
            else:
                # calculate CodeBLEU
                codebleu = calculate_CodeBLEU(task[f"PL_{i}"], task[f"PL_{i+1}"])["codebleu"]
                # calculate Pass/Fail Match (PFM)
                pfm = 1 if (summary == "Pass-Pass") or (summary == "Fail-Fail") else 0
                # calculate TOM
                tom = calculate_TOM(task[f"PL_{i}_results"], task[f"PL_{i+1}_results"], ignore_Timeout=True)["tom"]

            # update row for DataFrame traj
            row_traj["Summary"] = summary
            row_traj["EM-NL"] = str(int(em_NL))
            row_traj["BLEU-NL"] = str(bleu_NL)
            row_traj["chrF-NL"] = str(chrF_NL)
            row_traj["ROUGE-1-NL"] = str(rouge_1_NL)
            row_traj["ROUGE-2-NL"] = str(rouge_2_NL)
            row_traj["ROUGE-L-NL"] = str(rouge_L_NL)
            row_traj["BERTScore-NL"] = str(bertscore_NL)
            row_traj["EM-PL"] = str(int(em_PL))
            row_traj["CodeBLEU"] = str(codebleu)
            row_traj["P/FM"] = str(pfm)
            row_traj["TOM"] = str(tom)

        # add row_traj to the DataFrame traj
        traj = pd.concat([traj, pd.DataFrame([row_traj])], ignore_index=True)

    # write trajectory statistics and raw results to excel
    if args.output_path is None:
        output_path = args.input_path.replace(".jsonl", ".xlsx")
    else:
        output_path = args.output_path
    with pd.ExcelWriter(output_path) as writer:
        traj.to_excel(writer, sheet_name="traj", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        raw.to_excel(writer, sheet_name="raw", index=False, startrow=1, startcol=1, engine="xlsxwriter")


if __name__ == "__main__":
    main()
