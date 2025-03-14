"""
Direct GPT-3.5-turbo evaluation on either CommonsenseQA or arithmetic problems
"""

import os
import json
import argparse
from tqdm import tqdm
from utils.agent import Agent

class QASolver(Agent):
    def __init__(self, model_name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """Create a QA solver"""
        super(QASolver, self).__init__(model_name, "QA Solver", temperature, sleep_time)
        self.openai_api_key = openai_api_key

def solve_question(question_data, model_name, temperature, openai_api_key, sleep_time=0):
    """Solve a question using direct query to the specified model"""
    
    # Create a solver agent
    solver = QASolver(model_name, temperature, openai_api_key, sleep_time)
    
    # Determine the question format
    is_multiple_choice = "choices" in question_data
    
    # Set up prompt based on question format
    if is_multiple_choice:
        # Format the choices as a readable string
        choices_text = ""
        for choice in question_data["choices"]:
            choices_text += f"{choice['label']}: {choice['text']}\n"
        
        prompt = f"""Please answer this multiple-choice question:
Question: {question_data['question']}

Choices:
{choices_text}

Analyze each option carefully and select the best answer. Provide your reasoning and clearly indicate your final answer by stating "The answer is [LETTER]" at the end.
"""
    else:
        # Math/arithmetic problem
        prompt = f"""Please solve this math problem carefully and step by step. 
Problem: {question_data['question']}

Think through the problem logically and provide your answer in a clear format.
Make sure to carefully consider any counterintuitive aspects of the problem.
At the end, clearly state your final answer.
"""
    
    # Get solution
    solver.add_event(prompt)
    solution = solver.ask()
    
    # Process result based on question format
    if is_multiple_choice:
        # Extract the answer from the solution for multiple choice
        final_answer = None
        
        # Try to find a clear statement of the answer
        answer_patterns = [
            "the answer is ([a-eA-E])",
            "answer: ([a-eA-E])",
            "choose ([a-eA-E])",
            "option ([a-eA-E])",
            "select ([a-eA-E])"
        ]
        
        import re
        solution_lower = solution.lower()
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution_lower)
            if match:
                final_answer = match.group(1).upper()
                break
        
        # If no clear answer found, try to infer from context
        if final_answer is None:
            for choice in question_data["choices"]:
                label = choice["label"].upper()
                if f"option {label}" in solution_lower or f"choice {label}" in solution_lower:
                    # Check if this option is being endorsed
                    context = re.search(f"option {label}.*", solution_lower)
                    if context and ("correct" in context.group(0) or "best" in context.group(0)):
                        final_answer = label
                        break
        
        # If still no clear answer, default to None
        if final_answer is None:
            final_answer = "Could not determine"
        
        # Check if the answer is correct
        is_correct = (final_answer.upper() == question_data["answerKey"].upper())
        
    else:
        # Extract answer for math problems
        import re
        solution_lower = solution.lower()
        
        # Try to find the final answer
        final_answer = "Could not determine"
        
        # Look for patterns like "final answer: X" or "the answer is X"
        answer_patterns = [
            "final answer:?\s*(.+?)[\.|\n]",
            "the answer is:?\s*(.+?)[\.|\n]",
            "therefore,? the answer is:?\s*(.+?)[\.|\n]",
            "therefore:?\s*(.+?)[\.|\n]"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution_lower)
            if match:
                final_answer = match.group(1).strip()
                break
                
        # If no clear answer found, use the last line
        if final_answer == "Could not determine":
            last_line = solution.strip().split('\n')[-1]
            if "answer" in last_line.lower():
                final_answer = last_line
        
        # Check if the answer is correct
        correct_answers = question_data.get("answer", [])
        is_correct = False
        
        for answer in correct_answers:
            # Normalize both answers for comparison
            norm_extracted = final_answer.lower().replace(" ", "")
            norm_correct = answer.lower().replace(" ", "")
            
            if norm_correct in norm_extracted:
                is_correct = True
                break
                
            # Try numeric comparison if possible
            try:
                # Handle fractions
                if '/' in answer:
                    num, denom = answer.split('/')
                    correct_val = float(num) / float(denom)
                else:
                    correct_val = float(answer)
                    
                # Extract numeric values from final answer
                import re
                numbers = re.findall(r'\d+\.\d+|\d+/\d+|\d+', final_answer)
                for num in numbers:
                    if '/' in num:
                        n, d = num.split('/')
                        extracted_val = float(n) / float(d)
                    else:
                        extracted_val = float(num)
                        
                    # Compare with small tolerance
                    if abs(extracted_val - correct_val) < 0.001:
                        is_correct = True
                        break
            except:
                pass
    
    # Create result object
    result = {
        "id": question_data.get("id", "N/A"),
        "question": question_data["question"],
        "correct_answer": question_data.get("answerKey", question_data.get("answer", [])),
        "model_solution": solution,
        "extracted_answer": final_answer,
        "is_correct": is_correct
    }
    
    # Add choices if applicable
    if is_multiple_choice:
        result["choices"] = question_data["choices"]
    
    return result

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit number of questions to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key
    
    # Load the data
    with open(args.input_file, 'r') as f:
        questions = json.load(f)
    
    # Limit the number of questions if specified
    if args.limit is not None:
        questions = questions[:args.limit]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine the data format
    is_multiple_choice = "choices" in questions[0] if questions else False
    
    # Track correct answers
    correct_count = 0
    total_count = len(questions)
    results = []
    
    # Process each question
    for i, question in enumerate(tqdm(questions)):
        print(f"\n\n===== Question {i+1}/{total_count} =====")
        print(f"Question: {question['question']}")
        
        # Print choices if applicable
        if "choices" in question:
            print("Choices:")
            for choice in question["choices"]:
                print(f"{choice['label']}: {choice['text']}")
        
        # Solve the question
        result = solve_question(
            question_data=question,
            model_name=args.model_name,
            temperature=args.temperature,
            openai_api_key=openai_api_key
        )
        
        # Print results
        print(f"\nModel solution:\n{result['model_solution']}")
        print(f"\nExtracted answer: {result['extracted_answer']}")
        print(f"Correct answer: {result['correct_answer']}")
        print(f"Is correct: {result['is_correct']}")
        
        # Update accuracy count
        if result["is_correct"]:
            correct_count += 1
        
        # Save result
        results.append(result)
        
        # Save all results so far
        with open(os.path.join(args.output_dir, "direct_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save current accuracy
        accuracy = correct_count / (i + 1)
        print(f"\nCurrent accuracy: {correct_count}/{i+1} = {accuracy:.2%}")
        
    # Final accuracy
    final_accuracy = correct_count / total_count
    print(f"\n===== Final Results =====")
    print(f"Accuracy: {correct_count}/{total_count} = {final_accuracy:.2%}")
    
    with open(os.path.join(args.output_dir, "direct_accuracy.txt"), 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Accuracy: {correct_count}/{total_count} = {(correct_count/total_count):.2%}\n")