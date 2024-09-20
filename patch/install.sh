transformers_path=$(python -c "import os; import transformers; transformers_dir = os.path.dirname(transformers.__file__); print(transformers_dir)")
echo $transformers_path
python run_patch.py --package_path $transformers_path/models/llama --patch_files modeling_llama.py
python run_patch.py --package_path $transformers_path/generation --patch_files utils.py