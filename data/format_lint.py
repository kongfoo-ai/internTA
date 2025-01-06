import json
import os
import sys

# Define the expected structure of the JSON
expected_structure = {
    "labeler": str,
    "timestamp": str,
    "generation": int,
    "is_quality_control_question": bool,
    "is_initial_screening_question": bool,
    "question": {
        "problem": str,
        "ground_truth_solution": str,
        "ground_truth_answer": str,
        "pre_generated_steps": list,
        "pre_generated_answer": str,
        "pre_generated_verifier_score": float
    },
    "label": {
        "steps": list,
        "total_time": int,
        "finish_reason": str
    }
}

def validate_json_format(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return check_structure(data, expected_structure)

def check_structure(data, expected):
    if isinstance(expected, dict):
        if not isinstance(data, dict):
            return False
        for key, value_type in expected.items():
            if key not in data or not check_structure(data[key], value_type):
                return False
    elif isinstance(expected, list):
        if not isinstance(data, list):
            return False
        for item in data:
            if not check_structure(item, expected[0]):  # Assuming all items in the list are of the same structure
                return False
    else:
        return isinstance(data, expected)
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python format_lint.py <filename.json>")
        return

    file_path = sys.argv[1]
    
    if not os.path.isfile(file_path):
        print(f"{file_path}: File does not exist.")
        return

    if validate_json_format(file_path):
        print(f"{file_path}: Valid format")
    else:
        print(f"{file_path}: Invalid format")

# Example usage
if __name__ == "__main__":
    main()