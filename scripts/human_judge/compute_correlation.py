# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse

# External Modules
import pandas as pd
import scipy.stats


# calculate Pearson, Spearman, and Kendall-Tau correlations
def calculate_correlation(x, y):
    pearson_cor = scipy.stats.pearsonr(x, y)  # Pearson's r
    spearman_cor = scipy.stats.spearmanr(x, y)  # Spearman's rho
    kendall_tau_cor = scipy.stats.kendalltau(x, y)  # Kendall's tau
    return pearson_cor, spearman_cor, kendall_tau_cor


# EXAMPLE USAGE:
# python compute_correlation.py --gt_path ../../tmp/Human_Judge_GT.xlsx
# --metric_path ../../tmp/IDChain_gpt-3.5-turbo-0613_tmp0_len1_pb_all_m_human-judge_EvalPlus-Mini-v0.1.6_reformatted.jsonl.xlsx
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--metric_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # load the GT and analyzed results for various metrics
    gt_df = pd.read_excel(args.gt_path, sheet_name="gt", header=1)
    metric_df = pd.read_excel(args.metric_path, sheet_name="traj", header=1)

    # if PL0 pass, NL0 can seen as a Ground Truth reference for NL1, we can evaluate NL1 in both NL and PL space
    # otherwise, NL1 doesn't have a Ground Truth reference, we can only evaluate NL1 in PL space
    # therefore, we handle the above 2 case separately
    # create filter conditions
    filter_condition1 = metric_df["Summary"] == "Pass-Pass"
    filter_condition2 = metric_df["Summary"] == "Pass-Fail"
    filter_condiiton3 = metric_df["Summary"] == "Fail-Pass"
    filter_condition4 = metric_df["Summary"] == "Fail-Fail"
    # we will calculate correlation separately for "PL0 Pass" and "PL0 Fail" samples
    metric_df_pass = metric_df[filter_condition1 | filter_condition2]
    gt_df_pass = gt_df[filter_condition1 | filter_condition2]
    metric_df_fail = metric_df[filter_condiiton3 | filter_condition4]
    gt_df_fail = gt_df[filter_condiiton3 | filter_condition4]

    # sanity check
    print(f"Number of PL0 Pass Samples: {len(metric_df_pass)}")
    print(f"Number of PL0 Fail Samples: {len(metric_df_fail)}")

    # creat DataFrames to store the correlation results
    # we have Ground Truth (GT) and Relaxed Ground Truth (Relaxed GT)
    # Relaxed GT means in the docstring, nl description is correct, but some of the examples wrong
    cor_gt_pass = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])
    cor_relaxed_gt_pass = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])
    cor_gt_fail = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])
    cor_relaxed_gt_fail = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])
    cor_gt_all = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])
    cor_relaxed_gt_all = pd.DataFrame(columns=["Metric", "Pearson's r", "Spearman's rho", "Kendall's tau"])

    # correlation between GT and metrics
    for case in ["Pass", "Fail", "All"]:
        # if PL0 pass, we can see NL0 as the Ground Truth for PL1, we can evaluate in both NL and PL space
        if case == "Pass":
            gt = gt_df_pass["Ground Truth"].astype(float)
            relaxed_gt = gt_df_pass["Relaxed GT"].astype(float)
            # we have 6 NL space metrics
            bleu_NL = metric_df_pass["BLEU-NL"].astype(float)
            chrF_NL = metric_df_pass["chrF-NL"].astype(float)
            rouge_1_NL = metric_df_pass["ROUGE-1-NL"].astype(float)
            rouge_2_NL = metric_df_pass["ROUGE-2-NL"].astype(float)
            rouge_L_NL = metric_df_pass["ROUGE-L-NL"].astype(float)
            bertscore_NL = metric_df_pass["BERTScore-NL"].astype(float)
            # we have 4 PL space metrics
            em_PL = metric_df_pass["EM-PL"].astype(float)
            codebleu = metric_df_pass["CodeBLEU"].astype(float)
            pfm = metric_df_pass["P/FM"].astype(float)
            tom = metric_df_pass["TOM"].astype(float)
        # if PL0 fail, we have no Ground Truth for NL1, we can only evaluate in PL space
        else:
            # if case = "All", we will calculate correlation for samples that PL0 fails
            if case == "Fail":
                gt = gt_df_fail["Ground Truth"].astype(float)
                relaxed_gt = gt_df_fail["Relaxed GT"].astype(float)
                # we have 4 PL space metrics
                em_PL = metric_df_fail["EM-PL"].astype(float)
                codebleu = metric_df_fail["CodeBLEU"].astype(float)
                pfm = metric_df_fail["P/FM"].astype(float)
                tom = metric_df_fail["TOM"].astype(float)
            # if case = "All", we will calculate correlation for all samples
            else:
                gt = gt_df["Ground Truth"].astype(float)
                relaxed_gt = gt_df["Relaxed GT"].astype(float)
                # we have 4 PL space metrics
                em_PL = metric_df["EM-PL"].astype(float)
                codebleu = metric_df["CodeBLEU"].astype(float)
                pfm = metric_df["P/FM"].astype(float)
                tom = metric_df["TOM"].astype(float)
            # we don't have NL space metrics
            bleu_NL = None
            chrF_NL = None
            rouge_1_NL = None
            rouge_2_NL = None
            rouge_L_NL = None
            bertscore_NL = None

        # we have Ground Truth (GT) and Relaxed Ground Truth (Relaxed GT)
        # Relaxed GT means in the docstring, nl description is correct, but some of the examples wrong
        for gt_type in ["GT", "Relaxed GT"]:
            # list lout all the metrics we want to compute correlation with
            if case == "Pass":
                metrics = {
                    "BLEU-NL": bleu_NL,
                    "chrF-NL": chrF_NL,
                    "ROUGE-1-NL": rouge_1_NL,
                    "ROUGE-2-NL": rouge_2_NL,
                    "ROUGE-L-NL": rouge_L_NL,
                    "BERTScore-NL": bertscore_NL,
                    "EM-PL": em_PL,
                    "CodeBLEU": codebleu,
                    "P/FM": pfm,
                    "TOM": tom,
                }
            # if PL0 fail, we don't have NL space metrics
            else:
                metrics = {
                    "EM-PL": em_PL,
                    "CodeBLEU": codebleu,
                    "P/FM": pfm,
                    "TOM": tom,
                }

            # for each metric, calculate correlation with GT or Relaxed GT
            for metric_name, metric_result in metrics.items():
                print("==========================================")
                print(f"Correlation between {gt_type} and {metric_name}:")
                if gt_type == "GT":
                    pearson_cor, spearman_cor, kendall_tau_cor = calculate_correlation(gt, metric_result)
                else:
                    pearson_cor, spearman_cor, kendall_tau_cor = calculate_correlation(relaxed_gt, metric_result)

                print("-----------------------------------------")
                print(f"Pearson's r: {pearson_cor}")
                print(f"Spearman's rho: {spearman_cor}")
                print(f"Kendall's tau: {kendall_tau_cor}")
                print("==========================================")
                # add row to the corresponding DataFrame
                row = {
                    "Metric": metric_name,
                    "Pearson's r": pearson_cor[0],
                    "Spearman's rho": spearman_cor[0],
                    "Kendall's tau": kendall_tau_cor[0],
                }
                if gt_type == "GT":
                    if case == "Pass":
                        cor_gt_pass = pd.concat([cor_gt_pass, pd.DataFrame([row])], ignore_index=True)
                    elif case == "Fail":
                        cor_gt_fail = pd.concat([cor_gt_fail, pd.DataFrame([row])], ignore_index=True)
                    else:
                        cor_gt_all = pd.concat([cor_gt_all, pd.DataFrame([row])], ignore_index=True)
                else:
                    if case == "Pass":
                        cor_relaxed_gt_pass = pd.concat([cor_relaxed_gt_pass, pd.DataFrame([row])], ignore_index=True)
                    elif case == "Fail":
                        cor_relaxed_gt_fail = pd.concat([cor_relaxed_gt_fail, pd.DataFrame([row])], ignore_index=True)
                    else:
                        cor_relaxed_gt_all = pd.concat([cor_relaxed_gt_all, pd.DataFrame([row])], ignore_index=True)

            print("\n**********************************************************************************\n")

    # write correlation statistics to excel
    if args.output_path is None:
        output_path = "Human_Judge_Correlation.xlsx"
    else:
        output_path = args.output_path
    with pd.ExcelWriter(output_path) as writer:
        gt_df.to_excel(writer, sheet_name="gt", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        metric_df.to_excel(writer, sheet_name="traj", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        cor_gt_pass.to_excel(writer, sheet_name="cor_gt_pass", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        cor_relaxed_gt_pass.to_excel(
            writer, sheet_name="cor_relaxed_gt_pass", index=False, startrow=1, startcol=1, engine="xlsxwriter"
        )
        cor_gt_fail.to_excel(writer, sheet_name="cor_gt_fail", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        cor_relaxed_gt_fail.to_excel(
            writer, sheet_name="cor_relaxed_gt_fail", index=False, startrow=1, startcol=1, engine="xlsxwriter"
        )
        cor_gt_all.to_excel(writer, sheet_name="cor_gt_all", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        cor_relaxed_gt_all.to_excel(
            writer, sheet_name="cor_relaxed_gt_all", index=False, startrow=1, startcol=1, engine="xlsxwriter"
        )


if __name__ == "__main__":
    main()
