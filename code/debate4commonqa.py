"""
MAD: Multi-Agent Debate with Large Language Models for CommonsenseQA
"""

import os
import json
import argparse
from utils.agent import Agent
from tqdm import tqdm

NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """Create a player in the debate"""
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key

class Debate:
    def __init__(self,
            model_name: str='gpt-3.5-turbo', 
            temperature: float=0, 
            num_players: int=3, 
            openai_api_key: str=None,
            config: dict=None,
            max_round: int=3,
            sleep_time: float=0
        ) -> None:
        """Create a debate"""

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.openai_api_key = openai_api_key
        self.config = config
        self.max_round = max_round
        self.sleep_time = sleep_time

        self.init_prompt()

        # creat&init agents
        self.creat_agents()
        self.init_agents()
        
    def init_prompt(self):
        # Format the choices as a readable string
        choices_text = ""
        for choice in self.config["choices"]:
            choices_text += f"{choice['label']}: {choice['text']}\n"
        
        # Replace placeholders in prompts
        def prompt_replace(key):
            self.config[key] = self.config[key].replace("##question##", self.config["question"]).replace("##choices##", choices_text)
        
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, 
                        openai_api_key=self.openai_api_key, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.config['player_meta_prompt'])
        self.negative.set_meta_prompt(self.config['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.config['moderator_meta_prompt'])
        
        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.config['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)
        self.config['base_answer'] = self.aff_ans

        self.negative.add_event(self.config['negative_prompt'].replace('##aff_ans##', self.aff_ans))
        self.neg_ans = self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', 'first'))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = eval(self.mod_ans)

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def print_answer(self):
        print("\n\n===== Debate Done! =====")
        print("\n----- Question -----")
        print(self.config["question"])
        print("\n----- Choices -----")
        for choice in self.config["choices"]:
            print(f"{choice['label']}: {choice['text']}")
        print("\n----- Base Answer -----")
        print(self.config["base_answer"])
        print("\n----- Debate Answer -----")
        print(self.config["debate_answer"])
        print("\n----- Correct Answer -----")
        print(self.config["answerKey"])
        print("\n----- Is Correct -----")
        print(self.config["is_correct"])
        print("\n----- Debate Reason -----")
        print(self.config["Reason"])

    def run(self):
        for round in range(self.max_round - 1):
            if self.mod_ans["debate_answer"] != '':
                break
            else:
                print(f"===== Debate Round-{round+2} =====\n")
                self.affirmative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.neg_ans))
                self.aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(self.aff_ans)

                self.negative.add_event(self.config['debate_prompt'].replace('##oppo_ans##', self.aff_ans))
                self.neg_ans = self.negative.ask()
                self.negative.add_memory(self.neg_ans)

                self.moderator.add_event(self.config['moderator_prompt'].replace('##aff_ans##', self.aff_ans).replace('##neg_ans##', self.neg_ans).replace('##round##', self.round_dct(round+2)))
                self.mod_ans = self.moderator.ask()
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_answer"] != '':
            self.config.update(self.mod_ans)
            self.config['success'] = True

        # ultimate deadly technique if no conclusive answer yet
        else:
            judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time)
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            judge_player.set_meta_prompt(self.config['moderator_meta_prompt'])

            # extract answer candidates
            judge_player.add_event(self.config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            ans = judge_player.ask()
            judge_player.add_memory(ans)

            # select one from the candidates
            judge_player.add_event(self.config['judge_prompt_last2'])
            ans = judge_player.ask()
            judge_player.add_memory(ans)
            
            ans = eval(ans)
            if ans["debate_answer"] != '':
                self.config['success'] = True
            self.config.update(ans)
            self.players.append(judge_player)

        # Check if the debate answer is correct
        self.check_correctness()
        self.print_answer()
        return self.config

    def check_correctness(self):
        """Check if the debate answer matches the correct answer"""
        debate_answer = self.config["debate_answer"].strip().upper()
        correct_answer = self.config["answerKey"].strip().upper()
        
        # Look for the answer choice (A, B, C, D, E)
        # First check if it's already just the letter
        if debate_answer in ["A", "B", "C", "D", "E"]:
            self.config["is_correct"] = (debate_answer == correct_answer)
            return
            
        # Check if the letter is mentioned with the answer
        for choice in self.config["choices"]:
            choice_label = choice["label"].upper()
            choice_text = choice["text"].lower()
            
            # Check patterns like "Answer is A" or "I choose A" or "Option A is correct"
            if (f"answer is {choice_label}" in debate_answer.lower() or 
                f"choose {choice_label}" in debate_answer.lower() or
                f"option {choice_label}" in debate_answer.lower() or
                f"select {choice_label}" in debate_answer.lower()):
                self.config["is_correct"] = (choice_label == correct_answer)
                return
                
            # Check if it contains the text of the correct answer
            if choice_text in debate_answer.lower():
                self.config["is_correct"] = (choice_label == correct_answer)
                return
                
        # If no clear match, default to false
        self.config["is_correct"] = False


def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input CommonsenseQA JSON file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit number of questions to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    # Create config template for CommonsenseQA
    config_template = {
        "player_meta_prompt": "You are participating in a debate about a multiple-choice question. The question is: ##question##\nThe choices are:\n##choices##\nYour goal is to determine the correct answer by analyzing each option carefully.",
        
        "moderator_meta_prompt": "You are moderating a debate about a multiple-choice question. The question is: ##question##\nThe choices are:\n##choices##\nYour job is to evaluate arguments from both sides and determine which answer choice is correct based on commonsense reasoning.",
        
        "affirmative_prompt": "Please answer this multiple-choice question: ##question##\nThe choices are:\n##choices##\nAnalyze each option and select the best answer. Explain your reasoning.",
        
        "negative_prompt": "Review this solution to the multiple-choice question: ##aff_ans##\nIf you disagree with the answer or reasoning, explain why and provide your own analysis. If you agree, provide additional reasoning to support the answer.",
        
        "debate_prompt": "Consider the opposing argument: ##oppo_ans##\nRespond to their points with your own reasoning. You can defend your position or revise your answer if necessary.",
        
        "moderator_prompt": "The affirmative side argues: ##aff_ans##\n\nThe negative side argues: ##neg_ans##\n\nThis is the ##round## round. Evaluate both solutions carefully. If you can determine which answer is correct with confidence, return a JSON with this format: {\"debate_answer\": \"final answer (should be a letter A-E)\", \"Reason\": \"reason for your decision\", \"Supported Side\": \"side you support\"}\n\nIf you cannot determine with confidence yet, return {\"debate_answer\": \"\", \"Reason\": \"\", \"Supported Side\": \"\"}",
        
        "judge_prompt_last1": "You are judging a debate on this multiple-choice question: ##question##\nThe choices are:\n##choices##\n\nThe affirmative side argues: ##aff_ans##\n\nThe negative side argues: ##neg_ans##\n\nAnalyze both arguments and identify each of their proposed answers and reasoning.",
        
        "judge_prompt_last2": "Based on your analysis, which answer choice (A, B, C, D, or E) is correct? Return a JSON with this format: {\"debate_answer\": \"final answer (a letter A-E)\", \"Reason\": \"reason for your decision\", \"Supported Side\": \"side you support\"}"
    }

    # Load the CommonsenseQA data
    with open(args.input_file, 'r') as f:
        questions = json.load(f)
    
    # Limit the number of questions if specified
    if args.limit is not None:
        questions = questions[:args.limit]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Track correct answers
    correct_count = 0
    total_count = len(questions)
    results = []
    
    # Process each question
    for i, question in enumerate(tqdm(questions)):
        print(f"\n\n===== Question {i+1}/{total_count} =====")
        
        # Prepare config for this question
        config = config_template.copy()
        config.update({
            "id": question["id"],
            "question": question["question"],
            "choices": question["choices"],
            "answerKey": question["answerKey"]
        })
        
        # Run debate
        debate = Debate(
            model_name=args.model_name,
            temperature=args.temperature,
            num_players=3,
            openai_api_key=openai_api_key,
            config=config,
            max_round=3,
            sleep_time=0
        )
        
        result = debate.run()
        
        # Update accuracy count
        if result.get("is_correct", False):
            correct_count += 1
        
        # Save result
        results.append(result)
        
        # Save all results so far
        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save current accuracy
        accuracy = correct_count / (i + 1)
        print(f"\nCurrent accuracy: {correct_count}/{i+1} = {accuracy:.2%}")
        
        with open(os.path.join(args.output_dir, "accuracy.txt"), 'w') as f:
            f.write(f"Accuracy: {correct_count}/{total_count} = {(correct_count/total_count):.2%}\n")
    
    # Final accuracy
    final_accuracy = correct_count / total_count
    print(f"\n===== Final Results =====")
    print(f"Accuracy: {correct_count}/{total_count} = {final_accuracy:.2%}")