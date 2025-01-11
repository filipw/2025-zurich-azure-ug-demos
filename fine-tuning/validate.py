import json
import time
import subprocess
from typing import List, Dict, Tuple

def load_validation_data(file_path: str, limit: int = 10) -> List[Dict[str, str]]:
    examples = []
    print(f"Loading {limit} validation examples...")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line.replace('*', '"'))
            text = data["text"]
            user_part = text.split('<|user|>')[1].split('<|end|>')[0]
            assistant_part = text.split('<|assistant|>')[1].split('<|end|>')[0]
            examples.append({
                "input": user_part,
                "expected": assistant_part
            })
    return examples

def run_inference(model_path: str, adapter_path: str, prompt: str, base_model: bool = False) -> Tuple[str, float]:
    start_time = time.time()
    
    if base_model:
        system_prompt = """You control a music player. You can use these functions:
- play_song(title): Play a specific song
- play_list(title): Play a specific playlist
- pause: Pause playback
- stop: Stop playback
- next: Skip to next track
- prev: Go to previous track
- vol_up: Increase volume
- vol_down: Decrease volume
- mute: Mute audio
- unmute: Unmute audio

You should respond with a function call in the format: fn:function_name "parameter" (if needed)
For example: fn:play_song "bohemian rhapsody" or fn:play_list "workout mix" or fn:next, otherwise say "Sorry I cannot help with that"""
        
        formatted_prompt = f"<|system|>{system_prompt}<|end|>\n<|user|>{prompt}<|end|>"
    else:
        formatted_prompt = f"<|user|>{prompt}<|end|>"
    
    cmd = [
        "python", "-m", "mlx_lm.generate",
        "--model", model_path,
        "--prompt", formatted_prompt,
        "--max-tokens", "50",
        "--temp", "0.0"
    ]
    
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        response = result.stdout.strip()
        if "==========" in response:
            response = response.split("==========")[1].strip()
        if "<|end|>" in response:
            response = response.split("<|end|>")[0].strip()
        return response, time.time() - start_time
    except subprocess.CalledProcessError as e:
        return "", time.time() - start_time

def evaluate_model(model_path: str, adapter_path: str, validation_data: List[Dict[str, str]], model_name: str):
    results = {"perfect": 0, "command": 0, "total": 0, "time": 0}
    
    print(f"\n=== Testing {model_name} ===")
    
    for i, example in enumerate(validation_data, 1):
        input_text = example["input"]
        expected = example["expected"]
        
        actual, duration = run_inference(
            model_path, 
            adapter_path, 
            input_text, 
            base_model=(adapter_path is None)
        )
        
        results["total"] += 1
        perfect_match = expected.strip() == actual.strip()
        results["perfect"] += int(perfect_match)
        results["time"] += duration
        
        print(f"\nExample {i}/10:")
        print(f"Input: '{input_text}'")
        print(f"Expected: {expected}")
        print(f"Got: {actual}")
        print(f"Match: {'✓' if perfect_match else '✗'}")
    
    print(f"\n{model_name} Summary:")
    print(f"Perfect Match: {results['perfect']}/{results['total']} ({results['perfect'] / results['total']:.1%})")
    print(f"Avg Response: {results['time'] / results['total']:.1f}s")
    return results

def main():
    MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"
    ADAPTER_PATH = "adapters"
    VALIDATION_FILE = "data/valid.jsonl"
    
    print("=== Starting Validation ===")
    # we can set this to a larger value to test more examples
    validation_data = load_validation_data(VALIDATION_FILE, limit=10)
    
    finetuned_results = evaluate_model(MODEL_PATH, ADAPTER_PATH, validation_data, "Fine-tuned Model")
    print("\n" + "="*50 + "\n")
    base_results = evaluate_model(MODEL_PATH, None, validation_data, "Base Model")

if __name__ == "__main__":
    main()