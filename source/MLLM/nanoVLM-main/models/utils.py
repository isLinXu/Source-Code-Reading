import re

# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        correct_answer = correct_answer.upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results