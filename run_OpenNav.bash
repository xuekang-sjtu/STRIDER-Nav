flag="--exp_name opennav-robust-my-pertub
      --exp-config run_OpenNav.yaml
      --llm gpt-4o-2024-08-06
      --llm_api_key YOUR_API
      --vlm qwen-vl-max-latest
      --vlm_api_key YOUR_API
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=2 python run.py $flag