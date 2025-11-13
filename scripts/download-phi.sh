#!/bin/bash

set -e

model_dir="$1"

if [ -z "$model_dir" ]; then
    echo "Usage: $0 <model_directory>"
    exit 1
fi

model_url_base="https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
model_files=(
	"config.json"
	"genai_config.json"
	"model.onnx"
	"phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx.data"
	"phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx"
	"special_tokens_map.json"
	"tokenizer.json"
	"tokenizer_config.json"
)

mkdir -p "$model_dir"

echo "Downloading Phi-3.5-mini-instruct-onnx model if not present..."
for f in "${model_files[@]}"; do
	if [ ! -f "$model_dir/$f" ]; then
		echo "Downloading $f..."
		curl -L -o "$model_dir/$f" "$model_url_base/$f"
	fi
done