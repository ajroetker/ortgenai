#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

onnxruntime_version="$1"

if [[ -z $onnxruntime_version ]]; then
    echo version is required
    exit 1
fi

name="onnxruntime-genai-${onnxruntime_version}"-linux-x64
url="https://github.com/microsoft/onnxruntime-genai/releases/download/v${onnxruntime_version}/onnxruntime-genai-${onnxruntime_version}-linux-x64.tar.gz"

echo Downloading version "$onnxruntime_version" \(cpu\) from "${url} into $(pwd)"

function cleanup() {
    rm -r "$name.tar.gz" "$name" || true
}

trap cleanup EXIT

curl -LO "$url" && tar -xzf "./$name.tar.gz" && mv "./$name/lib/libonnxruntime-genai.so" /usr/lib64/libonnxruntime-genai.so