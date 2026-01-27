import argparse
import json
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import random
import multiprocessing

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor
from vllm import LLM, SamplingParams

ds_collections = {
    'MathVista_testmini': {
        'root': 'AI4Math/MathVista',  # 保持为 HF 数据集 ID
        'max_new_tokens': 4096,
        'min_new_tokens': 1,
        'split': 'testmini'  # 修改为 'test'
    },
}

SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
    "The final answer MUST BE put in \\boxed{} in <answer> </answer> tags."
)


def evaluate_chat_model(args, llm, processor):
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(ds_collections[ds_name]['root'])[ds_collections[ds_name]['split']]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            image_path = 'file://' + os.path.join('/data/baidu.zijing.he/data/mathvista_images', data_item['image'])
            data_item['query'] = data_item['query']
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {
                            "type": "text",
                            "text": data_item['query'] + " " + SYSTEM_PROMPT,
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            })

        # sampling_params = SamplingParams(temperature=0.01, top_p=0.001, top_k=1, max_tokens=4096,
        sampling_params = SamplingParams(
            temperature=0.0,
            top_k=1,
            n=1,
            max_tokens=4096,
            skip_special_tokens=False,
        )
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item['decoded_image']
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            pid = data_item['pid']
            temp[pid] = data_item

        print(f'Evaluating {ds_name} ...')
        results_file = args.filename
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/data/baidu.zijing.he/code/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3b__grpo__ep2_20260120_184400/global_step_16/actor/huggingface')
    parser.add_argument('--datasets', type=str, default='MathVista_testmini')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--filename', type=str, default='mathvista.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_num_seqs=512,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model(args, llm, processor)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    main()