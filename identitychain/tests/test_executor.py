# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import json
import os

# Internal Modules
from identitychain.utils import g_unzip

# Internal Modules to be tested
from identitychain import unsafe_execute


def test_unsafe_execute():
    task = {
        "problem": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "contract": "\n    assert type(paren_string) == str, \"invalid inputs\" # $_CONTRACT_$\n    cnt = 0 # $_CONTRACT_$\n    for ch in paren_string: # $_CONTRACT_$\n        assert ch in [\"(\", \")\", \" \"], \"invalid inputs\"  # $_CONTRACT_$\n        if ch == \"(\": cnt += 1 # $_CONTRACT_$\n        if ch == \")\": cnt -= 1 # $_CONTRACT_$\n        assert cnt >= 0, \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "separate_paren_groups",
        "model_solution": "\n\n    cnt, group, results = 0, \"\", []\n    for ch in paren_string:\n        if ch == \"(\": cnt += 1\n        if ch == \")\": cnt -= 1\n        if ch != \" \": group += ch\n        if cnt == 0:\n            if group != \"\": results.append(group)\n            group = \"\"\n    return results\n\n",
        "tests": [
            {
                "input": "'(()()) ((())) () ((())()())'",
                "output": "['(()())', '((()))', '()', '((())()())']",
            },
            {
                "input": "'() (()) ((())) (((())))'",
                "output": "['()', '(())', '((()))', '(((())))']",
            },
        ],
        "atol": 0,
    }
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == ["Passed", "Passed"]

    task["model_solution"] = "    return []"
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "AssertionError: outputs []",
        "AssertionError: outputs []",
    ]

    task["model_solution"] = "    return ['(()())', '((()))', '()', '((())()())']"
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == ["Passed", "AssertionError: outputs ['(()())', '((()))', '()', '((())()())']"]

    # print results
    print("++ All tests passed for test_unsafe_execute() ++")


# test executing functions with float ouputs
def test_unsafe_execute_atol():
    task = {
        "problem": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "contract": "\n    assert number > 0, \"invalid inputs\" # $_CONTRACT_$\n    assert isinstance(number, float), \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "truncate_number",
        "model_solution": "\n\n    return number - int(number)\n\n",
        "tests": [
            {"input": "3.5", "output": "0.5"},
            {"input": "1.33", "output": "0.33000000000000007"},
            {"input": "123.456", "output": "0.45600000000000307"},
            {"input": "999.99999", "output": "0.9999900000000252"},
            {"input": "0.04320870526393539", "output": "0.04320870526393539"},
            {"input": "1.0", "output": "0.0"},
            {"input": "1e-323", "output": "1e-323"},
            {"input": "1000000000.0", "output": "0.0"},
        ],
        "atol": 1e-06,
    }

    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == ["Passed", "Passed", "Passed", "Passed", "Passed", "Passed", "Passed", "Passed"]

    task["model_solution"] = "    return 0.0"
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "AssertionError: outputs 0.0",
        "AssertionError: outputs 0.0",
        "AssertionError: outputs 0.0",
        "AssertionError: outputs 0.0",
        "AssertionError: outputs 0.0",
        "Passed",
        "Passed",
        "Passed",
    ]

    task["model_solution"] = "    return 0.5"
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "Passed",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
        "AssertionError: outputs 0.5",
    ]

    # print results
    print("++ All tests passed for test_unsafe_execute_atol() ++")


# test executing functions with multiple def
def test_unsafe_execute_multiple_def():
    task = {
        "problem": "def is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n    return string == string[::-1]\n\n\ndef make_palindrome(string: str) -> str:\n    \"\"\" Find the shortest palindrome that begins with a supplied string.\n    Algorithm idea is simple:\n    - Find the longest postfix of supplied string that is a palindrome.\n    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.\n    >>> make_palindrome('')\n    ''\n    >>> make_palindrome('cat')\n    'catac'\n    >>> make_palindrome('cata')\n    'catac'\n    \"\"\"\n",
        "contract": "\n    assert type(string) == str, \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "make_palindrome",
        "model_solution": "\n    if is_palindrome(string):\n        return string\n    for i in range(len(string)):\n        if is_palindrome(string[i:]):\n            return string + string[i-1::-1]\n\n",
        "tests": [
            {"input": "''", "output": "''"},
            {"input": "'x'", "output": "'x'"},
            {"input": "'xyz'", "output": "'xyzyx'"},
            {"input": "'xyx'", "output": "'xyx'"},
            {"input": "'jerry'", "output": "'jerryrrej'"},
            {"input": "'race'", "output": "'racecar'"},
            {"input": "'level'", "output": "'level'"},
            {"input": "'raracece'", "output": "'raracececarar'"},
            {"input": "'raceredder'", "output": "'raceredderecar'"},
            {"input": "'abacabacaba'", "output": "'abacabacaba'"},
            {"input": "'rrefrerace'", "output": "'rrefreracecarerferr'"},
            {"input": "''", "output": "''"},
        ],
        "atol": 0,
    }
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
    ]

    # print results
    print("++ All tests passed for test_unsafe_execute_multiple_def() ++")


# test executing functions with escape characters in outputs
def test_unsafe_execute_escape():
    task = {
        "problem": "def remove_vowels(text):\n    \"\"\"\n    remove_vowels is a function that takes string and returns string without vowels.\n    >>> remove_vowels('')\n    ''\n    >>> remove_vowels(\"abcdef\\nghijklm\")\n    'bcdf\\nghjklm'\n    >>> remove_vowels('abcdef')\n    'bcdf'\n    >>> remove_vowels('aaaaa')\n    ''\n    >>> remove_vowels('aaBAA')\n    'B'\n    >>> remove_vowels('zbcd')\n    'zbcd'\n    \"\"\"\n",
        "contract": "\n    assert type(text) == str, \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "remove_vowels",
        "model_solution": "\n    return \"\".join(list(filter(lambda ch: ch not in \"aeiouAEIOU\", text)))\n\n",
        "tests": [
            {"input": "''", "output": "''"},
            {"input": "'abcdef\\nghijklm'", "output": "'bcdf\\nghjklm'"},
            {"input": "'fedcba'", "output": "'fdcb'"},
            {"input": "'eeeee'", "output": "''"},
            {"input": "'acBAA'", "output": "'cB'"},
            {"input": "'EcBOO'", "output": "'cB'"},
            {"input": "'ybcd'", "output": "'ybcd'"},
            {"input": "'hello'", "output": "'hll'"},
            {"input": "'CX'", "output": "'CX'"},
            {"input": "'AEEIayoubcd\\n\\n\\r\\t'", "output": "'ybcd\\n\\n\\r\\t'"},
            {"input": "'\\n'", "output": "'\\n'"},
            {"input": "'a'", "output": "''"},
        ],
        "atol": 0,
    }
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
    ]

    # print results
    print("++ All tests passed for test_unsafe_execute_escape() ++")


# test executing functions that will reulst in a timeout
def test_unsafe_execute_timeout():
    task = {
        "problem": "def fib(n: int):\n    \"\"\"Return n-th Fibonacci number.\n    >>> fib(10)\n    55\n    >>> fib(1)\n    1\n    >>> fib(8)\n    21\n    \"\"\"\n",
        "contract": "\n    assert n >= 0, \"invalid inputs\" # $_CONTRACT_$\n    assert isinstance(n, int), \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "fib",
        "model_solution": "    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fib(n-1) + fib(n-2)",
        "tests": [
            {"input": "10", "output": "55"},
            {"input": "1", "output": "1"},
            {"input": "8", "output": "21"},
            {"input": "11", "output": "89"},
            {"input": "12", "output": "144"},
            {"input": "16", "output": "987"},
            {"input": "0", "output": "0"},
            {"input": "1", "output": "1"},
            {"input": "3", "output": "2"},
            {"input": "63", "output": "6557470319842"},
            {"input": "2", "output": "1"},
        ],
        "atol": 0,
    }

    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Passed",
        "Time Out",
        "Passed",
    ]

    # print results
    print("++ All tests passed for test_unsafe_execute_timeout() ++")


# test executing functions with MBPP-reformatted Style test case format
def test_unsafe_execute_MBPP():
    task = {
        "problem": "def first_repeated_char(str1):\n    \"\"\" Write a python function to find the first repeated character in a given string.\n    \"\"\"\n",
        "contract": "",
        "function_name": "first_repeated_char",
        "model_solution": "    return 'a'",
        "tests": [
            "test_out = first_repeated_char(\"abcabc\")\nassert test_out == \"a\", f'outputs {test_out}'",
            "test_out = first_repeated_char(\"abc\")\nassert test_out == None, f'outputs {test_out}'",
            "test_out = first_repeated_char(\"123123\")\nassert test_out == \"1\", f'outputs {test_out}'",
        ],
        "atol": 0,
    }
    results = unsafe_execute(task, mask_func_name=False, timeout=0.5)
    assert results == [
        "Passed",
        "AssertionError: outputs a",
        "AssertionError: outputs a",
    ]

    os.makedirs("tmp", exist_ok=True)
    g_unzip("data/MBPP-S_test_reformatted.jsonl.gz", "tmp/MBPP-S_test_reformatted.jsonl")
    with open("tmp/MBPP-S_test_reformatted.jsonl", "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    for task in tasks:
        task = json.loads(task)
        test_task = {
            "problem": task["problem"],
            "contract": task["contract"],
            "function_name": task["function_name"],
            "model_solution": task["canonical_solution"],
            "tests": task["tests"],
            "atol": task["atol"],
        }
        results = unsafe_execute(test_task, mask_func_name=False, timeout=10)
        for result in results:
            assert result == "Passed"

    # print results
    print("++ All tests passed for test_unsafe_execute_MBPP() ++")


# test executing functions with name masked
def test_unsafe_execute_name_masked():
    # Test Case Format 1: a dict of input and output (HumanEvalPlus Style)
    task = {
        "problem": "from typing import List\n\n\ndef func(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "contract": "\n    assert type(paren_string) == str, \"invalid inputs\" # $_CONTRACT_$\n    cnt = 0 # $_CONTRACT_$\n    for ch in paren_string: # $_CONTRACT_$\n        assert ch in [\"(\", \")\", \" \"], \"invalid inputs\"  # $_CONTRACT_$\n        if ch == \"(\": cnt += 1 # $_CONTRACT_$\n        if ch == \")\": cnt -= 1 # $_CONTRACT_$\n        assert cnt >= 0, \"invalid inputs\" # $_CONTRACT_$\n",
        "function_name": "separate_paren_groups",
        "model_solution": "\n\n    cnt, group, results = 0, \"\", []\n    for ch in paren_string:\n        if ch == \"(\": cnt += 1\n        if ch == \")\": cnt -= 1\n        if ch != \" \": group += ch\n        if cnt == 0:\n            if group != \"\": results.append(group)\n            group = \"\"\n    return results\n\n",
        "tests": [
            {
                "input": "'(()()) ((())) () ((())()())'",
                "output": "['(()())', '((()))', '()', '((())()())']",
            },
            {
                "input": "'() (()) ((())) (((())))'",
                "output": "['()', '(())', '((()))', '(((())))']",
            },
        ],
        "atol": 0,
    }

    results = unsafe_execute(task, mask_func_name=True, timeout=0.5)
    assert results == ["Passed", "Passed"]

    # Test Case Format 2: a string of assertion statement (MBBP Style)
    os.makedirs("tmp", exist_ok=True)
    g_unzip("data/MBPP-S_test_reformatted.jsonl.gz", "tmp/MBPP-S_test_reformatted.jsonl")
    with open("tmp/MBPP-S_test_reformatted.jsonl", "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    task = json.loads(tasks[0])
    test_task = {
        "problem": task["problem"].replace(task["function_name"], "func"),
        "contract": task["contract"],
        "function_name": task["function_name"],
        "model_solution": task["canonical_solution"].replace(task["function_name"], "func"),
        "tests": task["tests"],
        "atol": task["atol"],
    }
    results = unsafe_execute(test_task, mask_func_name=True, timeout=5)
    for result in results:
        assert result == "Passed"

    # print results
    print("++ All tests passed for test_unsafe_execute_name_masked() ++")


def main():
    print("***********************************************")
    print("** Running tests for module <executor.py> **")
    print("-----------------------------------------------")
    test_unsafe_execute()
    test_unsafe_execute_atol()
    test_unsafe_execute_multiple_def()
    test_unsafe_execute_escape()
    test_unsafe_execute_timeout()
    test_unsafe_execute_MBPP()
    test_unsafe_execute_name_masked()
    print("-----------------------------------------------")
    print("** All tests passed for module <executor.py> **")
    print("***********************************************")


if __name__ == "__main__":
    main()
