"""
Direct GPT-3.5-turbo evaluation on Counterintuitive Arithmetic Reasoning problems
"""

import os
import json
import argparse
import openai
from tqdm import tqdm
import time
from utils.agent import Agent

class MathSolver(Agent):
    def __init__(self, model_name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """Create a math problem solver"""
        super(MathSolver, self).__init__(model_name, "Math Solver", temperature, sleep_time)
        self.openai_api_key = openai_api_key

def solve_problem(problem, model_name, temperature, openai_api_key, sleep_time=0):
    """Solve a single problem using direct GPT-3.5-turbo query"""
    
    # Create a solver agent
    solver = MathSolver(model_name, temperature, openai_api_key, sleep_time)
    
    # Set up prompt
    prompt = f"""Please solve this math problem carefully and step by step. 
Problem: {problem['question']}

Think through the problem logically and provide your answer in a clear format.
Make sure to carefully consider any counterintuitive aspects of the problem.
"""
    
    # Get solution
    solver.add_event(prompt)
    solution = solver.ask()
    
    # Extract the answer from the solution
    # This simple extraction might need refinement for complex responses
    answer_line = None
    for line in solution.split('\n'):
        if "answer" in line.lower() or "result" in line.lower() or "therefore" in line.lower():
            answer_line = line
            break
    
    # If no clear answer line found, just use the last line
    if answer_line is None:
        answer_line = solution.split('\n')[-1]
    
    # Check if the answer is correct
    is_correct = check_answer(solution, problem["answer"])
    
    result = {
        "problem": problem["question"],
        "correct_answer": problem["answer"],
        "model_solution": solution,
        "extracted_answer": answer_line,
        "is_correct": is_correct
    }
    
    return result

def check_answer(solution, correct_answers):
    """Check if the solution contains any of the correct answers"""
    solution_lower = solution.lower()
    
    for answer in correct_answers:
        # Check for exact match with the answer
        if answer.lower() in solution_lower:
            return True
        
        # Check for numerical equivalence (e.g., 1.5 vs 3/2)
        try:
            if '/' in answer:
                num, denom = answer.split('/')
                numeric_value = float(num) / float(denom)
            else:
                numeric_value = float(answer)
                
            # Look for this numeric value in the text
            if str(numeric_value) in solution_lower:
                return True
                
            # For percentages
            percentage = numeric_value * 100
            if f"{percentage}%" in solution_lower or f"{int(percentage)}%" in solution_lower:
                return True
        except:
            pass
            
    return False

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input CIAR.json file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key
    
    # Load the CIAR data
    with open(args.input_file, 'r') as f:
        problems = json.load(f)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Track correct answers
    correct_count = 0
    total_count = len(problems)
    results = []
    
    # Process each problem
    for i, problem in enumerate(tqdm(problems)):
        print(f"\n\n===== Problem {i+1}/{total_count} =====")
        print(f"Question: {problem['question']}")
        
        # Solve the problem
        result = solve_problem(
            problem=problem,
            model_name=args.model_name,
            temperature=args.temperature,
            openai_api_key=openai_api_key
        )
        
        # Print results
        print(f"\nModel solution:\n{result['model_solution']}")
        print(f"\nCorrect answer: {', '.join(problem['answer'])}")
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
        f.write(f"Accuracy: {correct_count}/{total_count} = {(correct_count/total_count):.2%}\n")