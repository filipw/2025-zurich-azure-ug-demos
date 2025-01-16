# Phi fine tuning with MLX

## Prerequisites 

Install MLX (included in the `requirements.txt` file of this repo).

```
pip install mlx-lm
```

or 

```
pip install -r requirements.txt
```

## 1. Sample task

Fine tuning Phi to become a music library controller, capable of invoking the following functions

* play_song(title)
* play_playlist(title)
* pause
* stop
* next track
* previous track
* volume up
* volume down
* mute
* unmute

## 2. Prepare test data

Sample `./data/train.jsonl`:

```json
{"text": "<|user|>Play Bohemian Rhapsody<|end|>\n<|assistant|>fn:play_song \"bohemian rhapsody\"<|end|>"}
{"text": "<|user|>Start my workout playlist<|end|>\n<|assistant|>fn:play_list \"workout mix\"<|end|>"}
{"text": "<|user|>Next song<|end|>\n<|assistant|>fn:next<|end|>"}
{"text": "<|user|>Skip track<|end|>\n<|assistant|>fn:next<|end|>"}
{"text": "<|user|>Make it louder<|end|>\n<|assistant|>fn:vol_up<|end|>"}
{"text": "<|user|>Turn down volume<|end|>\n<|assistant|>fn:vol_down<|end|>"}
```

## 3. Run LoRa fine tuning

```
python -m mlx_lm.lora --model microsoft/Phi-3-mini-4k-instruct --train --data ./data --iters 500
```

With the included data set this runs about 10 min on M1 Mac. It will be longer if each line is extended for more tokens.

Expected output should be similar too:

```
python -m mlx_lm.lora --model microsoft/Phi-3-mini-4k-instruct --train --data ./data --iters 500
Loading pretrained model
Fetching 13 files: 100%|███████████████████████████████████████████████| 13/13 [00:00<00:00, 120366.34it/s]
Loading datasets
Training
Trainable parameters: 0.082% (3.146M/3821.080M)
Starting training..., iters: 500
Iter 1: Val loss 5.617, Val took 10.292s
Iter 10: Train loss 5.889, Learning Rate 1.000e-05, It/sec 0.718, Tokens/sec 39.334, Trained Tokens 548, Peak mem 8.138 GB
Iter 20: Train loss 3.726, Learning Rate 1.000e-05, It/sec 0.963, Tokens/sec 62.197, Trained Tokens 1194, Peak mem 8.138 GB
Iter 30: Train loss 2.615, Learning Rate 1.000e-05, It/sec 0.907, Tokens/sec 61.238, Trained Tokens 1869, Peak mem 8.299 GB
Iter 40: Train loss 1.926, Learning Rate 1.000e-05, It/sec 0.215, Tokens/sec 12.482, Trained Tokens 2449, Peak mem 8.299 GB
Iter 50: Train loss 1.370, Learning Rate 1.000e-05, It/sec 1.062, Tokens/sec 63.596, Trained Tokens 3048, Peak mem 8.299 GB
Iter 60: Train loss 1.242, Learning Rate 1.000e-05, It/sec 0.765, Tokens/sec 52.368, Trained Tokens 3733, Peak mem 8.299 GB
Iter 70: Train loss 1.066, Learning Rate 1.000e-05, It/sec 0.912, Tokens/sec 64.395, Trained Tokens 4439, Peak mem 8.299 GB
Iter 80: Train loss 1.246, Learning Rate 1.000e-05, It/sec 1.130, Tokens/sec 63.478, Trained Tokens 5001, Peak mem 8.299 GB
Iter 90: Train loss 1.013, Learning Rate 1.000e-05, It/sec 1.129, Tokens/sec 70.541, Trained Tokens 5626, Peak mem 8.299 GB
Iter 100: Train loss 0.961, Learning Rate 1.000e-05, It/sec 1.069, Tokens/sec 64.164, Trained Tokens 6226, Peak mem 8.299 GB
Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.
Iter 110: Train loss 1.031, Learning Rate 1.000e-05, It/sec 1.043, Tokens/sec 60.923, Trained Tokens 6810, Peak mem 8.299 GB
Iter 120: Train loss 1.020, Learning Rate 1.000e-05, It/sec 1.133, Tokens/sec 62.794, Trained Tokens 7364, Peak mem 8.299 GB
Iter 130: Train loss 0.957, Learning Rate 1.000e-05, It/sec 1.015, Tokens/sec 64.164, Trained Tokens 7996, Peak mem 8.299 GB
Iter 140: Train loss 0.980, Learning Rate 1.000e-05, It/sec 1.124, Tokens/sec 68.016, Trained Tokens 8601, Peak mem 8.299 GB
Iter 150: Train loss 1.077, Learning Rate 1.000e-05, It/sec 1.133, Tokens/sec 72.973, Trained Tokens 9245, Peak mem 8.299 GB
Iter 160: Train loss 1.060, Learning Rate 1.000e-05, It/sec 1.134, Tokens/sec 63.957, Trained Tokens 9809, Peak mem 8.299 GB
Iter 170: Train loss 0.875, Learning Rate 1.000e-05, It/sec 1.200, Tokens/sec 67.674, Trained Tokens 10373, Peak mem 8.299 GB
Iter 180: Train loss 0.881, Learning Rate 1.000e-05, It/sec 1.061, Tokens/sec 64.908, Trained Tokens 10985, Peak mem 8.311 GB
Iter 190: Train loss 0.921, Learning Rate 1.000e-05, It/sec 1.013, Tokens/sec 62.836, Trained Tokens 11605, Peak mem 8.311 GB
Iter 200: Val loss 0.866, Val took 9.943s
Iter 200: Train loss 0.853, Learning Rate 1.000e-05, It/sec 12.562, Tokens/sec 788.899, Trained Tokens 12233, Peak mem 8.311 GB
Iter 200: Saved adapter weights to adapters/adapters.safetensors and adapters/0000200_adapters.safetensors.
Iter 210: Train loss 0.892, Learning Rate 1.000e-05, It/sec 0.963, Tokens/sec 65.009, Trained Tokens 12908, Peak mem 8.311 GB
Iter 220: Train loss 0.962, Learning Rate 1.000e-05, It/sec 1.132, Tokens/sec 63.863, Trained Tokens 13472, Peak mem 8.311 GB
Iter 230: Train loss 0.856, Learning Rate 1.000e-05, It/sec 0.962, Tokens/sec 61.475, Trained Tokens 14111, Peak mem 8.311 GB
Iter 240: Train loss 0.946, Learning Rate 1.000e-05, It/sec 1.066, Tokens/sec 66.943, Trained Tokens 14739, Peak mem 8.311 GB
Iter 250: Train loss 0.915, Learning Rate 1.000e-05, It/sec 1.134, Tokens/sec 63.838, Trained Tokens 15302, Peak mem 8.311 GB
Iter 260: Train loss 0.854, Learning Rate 1.000e-05, It/sec 1.073, Tokens/sec 64.836, Trained Tokens 15906, Peak mem 8.311 GB
Iter 270: Train loss 0.794, Learning Rate 1.000e-05, It/sec 0.955, Tokens/sec 68.840, Trained Tokens 16627, Peak mem 8.311 GB
Iter 280: Train loss 0.924, Learning Rate 1.000e-05, It/sec 1.013, Tokens/sec 65.612, Trained Tokens 17275, Peak mem 8.311 GB
Iter 290: Train loss 0.822, Learning Rate 1.000e-05, It/sec 1.071, Tokens/sec 63.807, Trained Tokens 17871, Peak mem 8.311 GB
Iter 300: Train loss 0.822, Learning Rate 1.000e-05, It/sec 1.140, Tokens/sec 64.963, Trained Tokens 18441, Peak mem 8.311 GB
Iter 300: Saved adapter weights to adapters/adapters.safetensors and adapters/0000300_adapters.safetensors.
Iter 310: Train loss 0.853, Learning Rate 1.000e-05, It/sec 1.055, Tokens/sec 64.690, Trained Tokens 19054, Peak mem 8.311 GB
Iter 320: Train loss 0.801, Learning Rate 1.000e-05, It/sec 1.067, Tokens/sec 61.757, Trained Tokens 19633, Peak mem 8.311 GB
Iter 330: Train loss 0.832, Learning Rate 1.000e-05, It/sec 1.201, Tokens/sec 69.149, Trained Tokens 20209, Peak mem 8.311 GB
Iter 340: Train loss 0.789, Learning Rate 1.000e-05, It/sec 1.071, Tokens/sec 65.561, Trained Tokens 20821, Peak mem 8.311 GB
Iter 350: Train loss 0.820, Learning Rate 1.000e-05, It/sec 1.012, Tokens/sec 68.030, Trained Tokens 21493, Peak mem 8.311 GB
Iter 360: Train loss 0.823, Learning Rate 1.000e-05, It/sec 1.075, Tokens/sec 68.154, Trained Tokens 22127, Peak mem 8.311 GB
Iter 370: Train loss 0.756, Learning Rate 1.000e-05, It/sec 0.920, Tokens/sec 61.796, Trained Tokens 22799, Peak mem 8.311 GB
Iter 380: Train loss 0.823, Learning Rate 1.000e-05, It/sec 1.066, Tokens/sec 66.544, Trained Tokens 23423, Peak mem 8.311 GB
Iter 390: Train loss 0.722, Learning Rate 1.000e-05, It/sec 1.016, Tokens/sec 63.221, Trained Tokens 24045, Peak mem 8.311 GB
Iter 400: Val loss 0.813, Val took 9.947s
Iter 400: Train loss 0.770, Learning Rate 1.000e-05, It/sec 11.773, Tokens/sec 804.125, Trained Tokens 24728, Peak mem 8.311 GB
Iter 400: Saved adapter weights to adapters/adapters.safetensors and adapters/0000400_adapters.safetensors.
Iter 410: Train loss 0.787, Learning Rate 1.000e-05, It/sec 1.123, Tokens/sec 63.569, Trained Tokens 25294, Peak mem 8.311 GB
Iter 420: Train loss 0.795, Learning Rate 1.000e-05, It/sec 1.202, Tokens/sec 65.155, Trained Tokens 25836, Peak mem 8.311 GB
Iter 430: Train loss 0.841, Learning Rate 1.000e-05, It/sec 1.069, Tokens/sec 62.086, Trained Tokens 26417, Peak mem 8.311 GB
Iter 440: Train loss 0.776, Learning Rate 1.000e-05, It/sec 1.065, Tokens/sec 62.702, Trained Tokens 27006, Peak mem 8.311 GB
Iter 450: Train loss 0.843, Learning Rate 1.000e-05, It/sec 1.068, Tokens/sec 64.512, Trained Tokens 27610, Peak mem 8.311 GB
Iter 460: Train loss 0.799, Learning Rate 1.000e-05, It/sec 1.006, Tokens/sec 68.085, Trained Tokens 28287, Peak mem 8.311 GB
Iter 470: Train loss 0.825, Learning Rate 1.000e-05, It/sec 1.202, Tokens/sec 65.887, Trained Tokens 28835, Peak mem 8.311 GB
Iter 480: Train loss 0.749, Learning Rate 1.000e-05, It/sec 1.067, Tokens/sec 62.291, Trained Tokens 29419, Peak mem 8.311 GB
Iter 490: Train loss 0.707, Learning Rate 1.000e-05, It/sec 0.427, Tokens/sec 28.804, Trained Tokens 30093, Peak mem 8.311 GB
Iter 500: Val loss 0.845, Val took 10.376s
Iter 500: Train loss 0.705, Learning Rate 1.000e-05, It/sec 7.810, Tokens/sec 488.131, Trained Tokens 30718, Peak mem 8.311 GB
Iter 500: Saved adapter weights to adapters/adapters.safetensors and adapters/0000500_adapters.safetensors.
Saved final weights to adapters/adapters.safetensors.
```

### Loss progression

Starting val loss: 5.617
Final val loss: 0.845
Training loss stabilized around 0.7-0.8 in the later iterations
Val loss has been fairly consistent since iter 200 (0.866 → 0.813 → 0.845)

### Key observations:

 - Very rapid initial improvement (loss dropped from 5.6 to ~1.0 in first 100 iterations)
 - Stabilized well by iter 200
 - Small fluctuations but no concerning overfitting (val loss tracks training loss well)
 - Final training loss around 0.7 is quite good for this type of task

### Summary

Given these results:

* The model has trained successfully
* 500 iterations was a good choice - we could have even have stopped at 300-400 iterations
* The stable validation loss suggests the model should generalize well

The output is created into `./adapters` folder and is an adapter which can be layered on top of the model.

It should be about 140 MB in size.

## 4. Test the adapter

The base model can be invoked with the adapter to test the fine tuning.

```
python -m mlx_lm.generate --model microsoft/Phi-3-mini-4k-instruct --adapter-path ./adapters --max-token 2048 --prompt "i don't like this song" --extra-eos-token "<|end|>" --temp 0.0
```

```
Expected output:
==========
fn:next
==========
Prompt: 10 tokens, 44.057 tokens-per-sec
Generation: 4 tokens, 9.403 tokens-per-sec
Peak memory: 7.821 GB
```

## 5. Merge the adapter into the model

```
python -m mlx_lm.fuse --model microsoft/Phi-3-mini-4k-instruct
```

This creates a fused safe tensors model inside `./fused_model` folder. From there it can be used directly with any ML framework that supports safe tensors, or it can be subject to quantization or other optimizations. 

## 6. Test the final model

There is a simple validation script in the repo which runs the inference using data from `./data/valid.jsonl`. This is a data set that the model has not seen before.

```
python validate.py
```

The script execute the inference against the fine tuned model (without the system instruction) and the base model with a system instruction and a few shot learning approach. The result should show that the fine tuned model is not only faster than the base model (as it uses less tokens!) but also dramatically more accurate than the base model.

The output should be similar to:

```
=== Starting Validation ===
Loading 25 validation examples...

=== Loading Fine-tuned Model ===
Fetching 13 files: 100%|██████████████████████████████████████████████████| 13/13 [00:00<00:00, 74387.38it/s]

=== Testing Fine-tuned Model ===
[ 1/25] ✓ 'Boost the volume' → fn:vol_up
[ 2/25] ✓ 'What's your name again?' → Sorry I cannot help with that
[ 3/25] ✓ 'Turn up' → fn:vol_up
[ 4/25] ✓ 'Lower it' → fn:vol_down
[ 5/25] ✓ 'Last song please' → fn:prev
[ 6/25] ✓ 'Skip' → fn:next
[ 7/25] ✓ 'Stop all' → fn:stop
[ 8/25] ✓ 'Audio enable' → fn:unmute
[ 9/25] ✓ 'start power workout' → fn:play_list "power workout"
[10/25] ✓ 'Play We Will Rock You' → fn:play_song "we will rock you"
[11/25] ✓ 'Play Any Way You Want It' → fn:play_song "any way you want it"
[12/25] ✓ 'Forward' → fn:next
[13/25] ✓ 'Play Faithfully' → fn:play_song "faithfully"
[14/25] ✓ 'Play Open Arms' → fn:play_song "open arms"
[15/25] ✓ 'Play Separate Ways' → fn:play_song "separate ways"
[16/25] ✓ 'Switch' → fn:next
[17/25] ✓ 'Back' → fn:prev
[18/25] ✓ 'Reverse' → fn:prev
[19/25] ✓ 'End' → fn:stop
[20/25] ✓ 'Finish it' → fn:stop
[21/25] ✓ 'Pause now' → fn:pause
[22/25] ✓ 'Stop briefly' → fn:pause
[23/25] ✓ 'Silent' → fn:mute
[24/25] ✓ 'No noise' → fn:mute
[25/25] ✓ 'Kill audio' → fn:stop

Summary: 25/25 correct (100.0%), avg 1.3s per request

==================================================


=== Loading Base Model ===
Fetching 13 files: 100%|██████████████████████████████████████████████████| 13/13 [00:00<00:00, 41401.63it/s]

=== Testing Base Model ===
[ 1/25] ✓ 'Boost the volume' → fn:vol_up
[ 2/25] ✗ 'What's your name again?' → I'm Phi, an AI developed by Microsoft.
[ 3/25] ✓ 'Turn up' → fn:vol_up
[ 4/25] ✓ 'Lower it' → fn:vol_down
[ 5/25] ✗ 'Last song please' → fn:next
[ 6/25] ✓ 'Skip' → fn:next
[ 7/25] ✓ 'Stop all' → fn:stop
[ 8/25] ✓ 'Audio enable' → fn:unmute
[ 9/25] ✓ 'start power workout' → fn:play_list "power workout"
[10/25] ✗ 'Play We Will Rock You' → fn:play_song "We Will Rock You"
[11/25] ✗ 'Play Any Way You Want It' → fn:play_song "Any Way You Want It"
[12/25] ✓ 'Forward' → fn:next
[13/25] ✗ 'Play Faithfully' → fn:play_song "Faithfully"
[14/25] ✗ 'Play Open Arms' → fn:play_song "Open Arms"
[15/25] ✗ 'Play Separate Ways' → fn:play_song "Separate Ways"
[16/25] ✓ 'Switch' → fn:next
[17/25] ✗ 'Back' → fn:play_song "Back in Black"
[18/25] ✓ 'Reverse' → fn:prev
[19/25] ✗ 'End' → fn:stop
fn:mute
fn:vol_down
[20/25] ✗ 'Finish it' → I'm sorry, but it seems like there was no specific request or command provided to respond to. If you have a command or request related to the music player functions, please provide it, and I'll respond accordingly.
[21/25] ✓ 'Pause now' → fn:pause
[22/25] ✗ 'Stop briefly' → fn:stop
[23/25] ✓ 'Silent' → fn:mute
[24/25] ✗ 'No noise' → fn:mute

This command will mute the audio output of the music player, effectively stopping any noise from being emitted.
[25/25] ✓ 'Kill audio' → fn:stop

Summary: 13/25 correct (52.0%), avg 2.5s per request
```