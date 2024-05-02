# Authors: marcusm117, Robin-Y-Ding
# License: Apache 2.0


# Standard Library Modules
import json
from typing import Any, Callable, Dict, List

# External Modules
from tqdm import tqdm

# Internal Modules
from identitychain import unsafe_execute


# a list of supported Foundation Models
# PL-2-NL generation for Foundation Model is Fill-In-the-Middle (FIM)
FOUNDATION_MODELS = [
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-70b-hf",
    "bigcode/starcoderbase-1b",
    "bigcode/starcoderbase-3b",
    "bigcode/starcoderbase-7b",
    "bigcode/starcoderbase",
    "bigcode/starcoderplus",
    "bigcode/starcoder",
    "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-ai/deepseek-coder-33b-base",
    "deepseek-ai/deepseek-coder-7b-base-v1.5",
]


# a list of supported Instruction-Tuned Models
INSTRUCTION_MODELS = [
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "HuggingFaceH4/starchat-beta",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
]


class IdentityChainError(Exception):
    """Exceptions raised by methods related to class IdentityChain."""


class IdentityChain:
    """A class that performs a chain of Identity Transformations, to wholistically evluate a Code LM.

    Given a problem description, NL_0, prompt the model to generate a solution, PL_0, and run test cases on PL_0.
    Prompt the model to backtranslate/summarize PL_0 into a new problem description, NL_1,
    prompt the model to generate a solution, PL_1, and run test cases on PL_1.

    Denote the semantics of PL and NL as sem(PL) and sem(NL),
    I() is the Identity Transformation in the Semantics Space.
    then we have:

    sem ∘ NL-2-PL ∘ PL-2-NL(PL_0) = sem ∘ NL-2-PL(NL_1) = sem(PL_1) = I^2(sem(PL_0))

    We expect semantics, thus the test outputs, of PL_1 to be exactly the same as PL_0.
    In general, we want to see:

    sem ∘ (NL-2-PL ∘ PL-2-NL)^n (PL_0) = I^(2n)(sem(PL_0))

    where n is the number of Identity Transformations performed, which is also the length of the Chain.


    Attributes:
        input_path (str): a string representing the path to the input file
        output_path (str): a string representing the path to the output file
        get_model_response_NL_to_PL (callable): a function that takes in a prompt and a problem description,
            and returns the Model's solution to the problem description
        get_model_response_PL_to_NL (callable): a function that takes in a prompt and a code solution,
            and returns the Model's summarization of the code solution
        prompt_NL_to_PL: a prompt to get the Model's solution to a Natural Langauge problem description,
            will be used by self.get_model_response_NL_to_PL
        prompt_PL_to_NL: a prompt to get the Model's summarization of a Programming Language code solution,
            will be used by self.get_model_response_PL_to_NL
        length (int): the length of the Identity Chain
        bootstrap_method (str): either "problem" or "canonical_solution"

    """

    def __init__(
        self,
        model,
        tokenizer,
        args,
        input_path,
        output_path,
        get_model_response_NL_to_PL,
        get_model_response_PL_to_NL,
        prompt_NL_to_PL,
        prompt_PL_to_NL,
        length=5,
        bootstrap_method="problem",
    ):
        self.model = model  # model is either a string (for APIs) or a HuggingFace AutoModelForCausalLM object
        self.tokenizer = tokenizer  # tokenizer is either None (for APIs) or a HuggingFace AutoTokenizer object
        self.args = args  # args is a argparse.Namespace object to store hyperparameters for model inference
        self.input_path = input_path
        self.output_path = output_path
        self.get_model_response_NL_to_PL = get_model_response_NL_to_PL
        self.get_model_response_PL_to_NL = get_model_response_PL_to_NL
        self.prompt_NL_to_PL = prompt_NL_to_PL
        self.prompt_PL_to_NL = prompt_PL_to_NL
        self.set_length(length)
        self.set_bootstrap_method(bootstrap_method)

    def set_model(self, model: Any):
        """Set the model.

        Args:
            model: either a string (for APIs) or a HuggingFace AutoModelForCausalLM object

        """
        self.model = model

    def set_tokenizer(self, tokenizer: Any):
        """Set the tokenizer.

        Args:
            tokenizer: either None (for APIs) or a HuggingFace AutoTokenizer object

        """
        self.tokenizer = tokenizer

    def set_args(self, args: Any):
        """Set the args.

        Args:
            args: a argparse.Namespace object to store hyperparameters for model inference

        """
        self.args = args

    def set_input_path(self, input_path: str):
        """Set the path to the input file.

        Args:
            input_path: a string representing the path to the input file

        """
        self.input_path = input_path

    def set_output_path(self, output_path: str):
        """Set the path to the output file.

        Args:
            output_path: a string representing the path to the output file

        """
        self.output_path = output_path

    def set_get_model_response_NL_to_PL(self, get_model_response_NL_to_PL: Callable):
        """Set the function to get the Model's solution to a Natural Langauge problem description.

        Args:
            get_model_response_NL_to_PL: a function that takes in a prompt and a problem description,
                and returns the Model's solution to the problem description

        """
        self.get_model_response_NL_to_PL = get_model_response_NL_to_PL

    def set_get_model_response_PL_to_NL(self, get_model_response_PL_to_NL: Callable):
        """Set the function to get the Model's summarization of a Programming Language code solution.

        Args:
            get_model_response_PL_to_NL: a function that takes in a prompt and a code solution,
                and returns the Model's summarization of the code solution

        Note:
            If the Model have PL-2-NL generation ability i.e.
            the ability to summarize a code solution into a problem description,
            then this function should be the same as self.get_model_response_NL_to_PL.
            If not, this function should be different.

        """
        self.get_model_response_PL_to_NL = get_model_response_PL_to_NL

    def set_prompt_NL_to_PL(self, prompt_NL_to_PL: str | List[Dict[str, str]]):
        """Set the prompt to get the Model's solution to a Natural Langauge problem description.

        Args:
            prompt_NL_to_PL: a prompt to get the Model's solution to a Natural Langauge problem description,
                will be used by self.get_model_response_NL_to_PL

        """
        self.prompt_NL_to_PL = prompt_NL_to_PL

    def set_prompt_PL_to_NL(self, prompt_PL_to_NL: str | List[Dict[str, str]]):
        """Set the prompt to get the Model's summarization of a Programming Language code solution.

        Args:
            prompt_PL_to_NL: a prompt to get the Model's summarization of a Programming Language code solution,
                will be used by self.get_model_response_PL_to_NL

        """
        self.prompt_PL_to_NL = prompt_PL_to_NL

    def set_length(self, length: int):
        """Set the length of the Identity Chain.

        Args:
            length: the length of the Identity Chain

        Raises:
            IdentityChainError: if length is not a positive integer
            IdentityChainError: if user does not confirm to run an Identity Chain of length > 5

        Note:
            lenght = 5 is usually enough to get a good estimate of the model's performance.
            If you are using a very large evaluation set, you may want to set length to 4 or even 3.

        """
        if length < 1:
            raise IdentityChainError("length must be a positive integer")
        if length > 5:
            answer = input("WARNING: length > 5 may take a long time to run. Continue? (y/n) ")
            if answer != "y" and answer != "Y":
                raise IdentityChainError("enter 'y' or 'Y' to confirm an Identity Chain of length > 5")
        self.length = length

    def set_bootstrap_method(self, bootstrap_method: str):
        """Set the method to bootstrap the Identity Chain.

        We can either use the problem description as NL_0, and use the model's generated solution PL_0,
        or if the dataset includes the canonical solution, we can use it as PL_0 directly.

        Args:
            bootstrap_method: either "problem" or "canonical_solution"

        Raises:
            IdentityChainError: if bootstrap_method is not "problem" or "canonical_solution"

        """
        if bootstrap_method not in ["problem", "canonical_solution"]:
            raise IdentityChainError("bootstrap_method must be either 'problem' or 'canonical_solution'")
        self.bootstrap_method = bootstrap_method

    def bootstrap(self, resume_task: int = 1):
        """Bootstrap the Identity Chain to get PL_0.

        Args:
            resume_task: the task to resume from. Defaults to 1, since tasks[0] can used for in-context prompting.

        Raises:
            IdentityChainError: if resume_task is a negative integer

        """
        # check for invalid input
        if resume_task < 0:
            raise IdentityChainError("resume_task must be a non-negative integer")

        # read in all the tasks
        with open(self.input_path, "r", encoding="utf-8") as reader:
            tasks = reader.readlines()

        # bootstrap the Identity Chain for each task
        with open(self.output_path, "a", encoding="utf-8") as writer:
            for input_task in tqdm(tasks[resume_task:]):
                input_task = json.loads(input_task)

                # get the Original Problem Description NL_0, and other invariant fields
                NL_0 = input_task["problem"]
                task_id = input_task["task_id"]
                contract = input_task["contract"]
                function_name = input_task["function_name"]
                function_signature = input_task["function_signature"]
                tests = input_task["tests"]
                atol = input_task["atol"]

                # bootstrap PL_0, it can be either the Model Solution or the Canonical Solution
                if self.bootstrap_method == "problem":
                    # prompt the model to generate a solution PL_0 given the problem description NL_0
                    # the prompt we are using is "self.prompt_NL_to_PL"
                    PL_0 = self.get_model_response_NL_to_PL(
                        self.prompt_NL_to_PL, NL_0, self.model, self.tokenizer, self.args
                    )
                else:
                    PL_0 = input_task["canonical_solution"]

                # run test cases on PL_0
                test_task = {
                    "problem": NL_0,
                    "contract": contract,
                    "function_name": function_name,
                    "model_solution": PL_0,
                    "tests": tests,
                    "atol": atol,
                }
                # no need to mask the function name when bootstrapping
                results = unsafe_execute(test_task, mask_func_name=False, timeout=0.5)

                # write to output file
                output_task = {
                    "task_id": task_id,
                    "NL_0": NL_0,
                    "contract": contract,
                    "function_name": function_name,
                    "function_signature": function_signature,
                    "tests": tests,
                    "atol": atol,
                    "PL_0": PL_0,
                    "PL_0_results": results,
                }
                writer.write(json.dumps(output_task) + "\n")
                writer.flush()

                # for debugging
                # print(output_task)

    def run(
        self,
        resume_task: int = 0,
        resume_step: int = 1,
        pass_only: bool = False,
        mask_func_name: bool = True,
        greedy_early_stop: bool = False,
    ):
        """Run the Identity Chain to get PL_1 to PL_n.

        Args:
            resume_task: the task to resume from
            resume_step: the step to resume from. Defaults to 1, since Step 0 is the bootstrap step.
            pass_only: whether to only run on tasks that passed all tests in the previous step
            mask_func_name: whether to replace the function name with a generic "func",
                to avoid conflict when a wrong solution/problem description that doesn't match the function name
            greedy_early_stop: when using greedy decoding, whether to stop the model generation when
                PL_i is exactly the same as PL_{i-1} or NL_i is exactly the same as NL_{i-1}.
                The Identity Chain keeps running by just copying the previous step's solution/problem description.
                !!!WARNING!!! This should only be set to True when the model is set to do greedy decoding!!!

        Raises:
            IdentityChainError: if resume_task is a negative integer
            IdentityChainError: if resume_step is not in the closed interval [1, self.length]
            IdentityChainError: if greedy_early_stop is True and the user does not confirm

        """
        # check for invalid input
        if resume_task < 0:
            raise IdentityChainError("resume_task must be a non-negative integer")
        if resume_step > self.length or resume_step < 1:
            raise IdentityChainError("resume_step must be in [1, self.length]")

        # double check if using greedy decoding
        if greedy_early_stop:
            answer = input(
                "WARNING: greedy_early_stop is True. This should only be set to True when doing greedy decoding. Continue? (y/n) "
            )
            if answer != "y" and answer != "Y":
                raise IdentityChainError("enter 'y' or 'Y' to confirm setting greedy_early_stop to True")

        # read in all the tasks, from the bootstrapped output file
        with open(self.output_path, "r", encoding="utf-8") as reader:
            tasks = reader.readlines()

        # run the Identity Chain for each task
        for idx, task in enumerate(tqdm(tasks[resume_task:])):
            task = json.loads(task)
            # if greedy_early_stop is set to True, we use this flag to indicate if we have reached a fixed point
            # so that we don't waste time generating the exact same solutions/problem descriptions again
            fixed_point = False

            # run Identity Transformation for each step in the Chain
            for i in range(resume_step, self.length + 1):
                # if pass_only option is False,
                # check if PL_{i-1} is defined and passed all test cases,
                # where PL_{i-1} is the Model Solution from the previous step,
                # if undefined or passed, stop running the Identity Chain for this task
                if pass_only and (f"PL_{i-1}_results" not in task or "Error" in "".join(task[f"PL_{i-1}_results"])):
                    task[f"NL_{i}"] = "NA"
                    task[f"PL_{i}"] = "NA"
                    task[f"PL_{i}_results"] = "NA"

                # if pass_only option is False, run the Identity Chain on every task regardless
                else:
                    # get invariant fields
                    contract = task["contract"]
                    function_name = task["function_name"]
                    function_signature = task["function_signature"]
                    tests = task["tests"]
                    atol = task["atol"]

                    # if mask_func_name option is True, repalce the function name with a generic "func" in the function signature
                    masked_function_signature = function_signature
                    if mask_func_name:
                        masked_function_signature = function_signature.replace(function_name, "func")

                    # get PL_{i-1}, the Model Solution from the previous step
                    PL_i_minus_1 = task[f"PL_{i-1}"]
                    # if mask_func_name option is True, repalce the function name with a generic "func" in PL_0, as it might be recursive
                    # if PL_0 is recursive and we don't replace, then the original function name in PL_0 will be an undefined name
                    if mask_func_name and i == 1:
                        PL_i_minus_1 = PL_i_minus_1.replace(function_name, "func")

                    # prompt the Code Model to summarize a new problem description NL_i given PL_{i-1}
                    # the prompt we are using is self.PL_to_NL
                    # if we have reached a fixed point, copy the problem description from the previous step
                    if fixed_point:
                        NL_i = task[f"NL_{i-1}"]
                    else:
                        if self.args.model_name_or_path in FOUNDATION_MODELS:
                            NL_i = self.get_model_response_PL_to_NL(
                                self.prompt_PL_to_NL,
                                masked_function_signature,
                                PL_i_minus_1,
                                self.model,
                                self.tokenizer,
                                self.args,
                            )
                        else:
                            NL_i = self.get_model_response_PL_to_NL(
                                self.prompt_PL_to_NL,
                                masked_function_signature.rstrip('\n') + '\n' + PL_i_minus_1.lstrip('\n'),
                                self.model,
                                self.tokenizer,
                                self.args,
                            )

                    # if NL_i = NL_{i-1} and we're using greedy decoding, then we have reached a fixed point
                    if greedy_early_stop and NL_i == task[f"NL_{i-1}"]:
                        # for debugging
                        print("Reached a Fixed Point! Stop Model Inference.")
                        fixed_point = True

                    # prompt the Code Model to generate a solution PL_i given the problem description NL_i
                    # the prompt we are using is "self.NL_to_PL"
                    # if we have reached a fixed point, copy the solution from the previous step
                    if fixed_point:
                        PL_i = task[f"PL_{i-1}"]
                    else:
                        PL_i = self.get_model_response_NL_to_PL(
                            self.prompt_NL_to_PL,
                            masked_function_signature + NL_i,
                            self.model,
                            self.tokenizer,
                            self.args,
                        )

                    # if PL_i = PL_{i-1} and we're using greedy decoding, then we have reached a fixed point
                    if greedy_early_stop and PL_i == task[f"PL_{i-1}"]:
                        # for debugging
                        print("Reached a Fixed Point! Stop Model Inference.")
                        fixed_point = True

                    # run test cases on PL_i
                    # if we have reached a fixed point, copy the results from the previous step
                    if fixed_point:
                        results = task[f"PL_{i-1}_results"]
                    else:
                        test_task = {
                            "problem": masked_function_signature + NL_i,
                            "contract": contract,
                            "function_name": function_name,
                            "model_solution": PL_i,
                            "tests": tests,
                            "atol": atol,
                        }
                        results = unsafe_execute(test_task, mask_func_name, timeout=0.5)

                    # update new fields
                    task[f"NL_{i}"] = NL_i
                    task[f"PL_{i}"] = PL_i
                    task[f"PL_{i}_results"] = results

                # write to output file
                tasks[idx + resume_task] = json.dumps(task) + "\n"
                with open(self.output_path, "w", encoding="utf-8") as writer:
                    writer.writelines(tasks)
                    writer.flush()

                # for debugging
                print(tasks[idx + resume_task])
