import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
except ImportError:
    pass

try:
    # Try importing Qwen2.5-VL classes (usually in newer transformers)
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
except ImportError:
    try:
        # Fallback to internal path if not exposed at top level
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
        Qwen2_5_VLProcessor = None


def load_items(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # ensure list
    if isinstance(data, dict):
        data = [data]
    return data


def flatten_keywords(steps: List[List[str]]) -> List[str]:
    out = []
    for group in steps:
        for token in group:
            if isinstance(token, str):
                out.append(token.strip())
    # keep unique order
    seen = set()
    uniq = []
    for t in out:
        if t and t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def find_token_positions(tokenizer, input_ids: torch.Tensor, keywords: List[str]) -> List[Tuple[str, List[int]]]:
    """
    Find keyword positions with a case-insensitive subtoken match.
    This avoids misses like "X-axis" -> ["x", "-", "axis"].
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    tokens_lc = [t.lower() for t in tokens]
    positions = []
    for kw in keywords:
        kw_ids = tokenizer.encode(kw, add_special_tokens=False)
        kw_tokens = tokenizer.convert_ids_to_tokens(kw_ids)
        kw_tokens_lc = [t.lower() for t in kw_tokens]
        match_pos = []
        if not kw_tokens_lc:
            positions.append((kw, match_pos))
            continue
        L = len(kw_tokens_lc)
        for i in range(len(tokens_lc) - L + 1):
            if tokens_lc[i : i + L] == kw_tokens_lc:
                match_pos.append(i + L - 1)  # use last subtoken as anchor
        positions.append((kw, match_pos))
    return positions


def get_vision_span(tokenizer, input_ids: torch.Tensor) -> Tuple[int, int]:
    vid_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vid_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    start_idx = -1
    end_idx = -1
    for i, tid in enumerate(input_ids.tolist()):
        if tid == vid_start and start_idx < 0:
            start_idx = i + 1  # tokens after start
        if tid == vid_end and start_idx >= 0:
            end_idx = i  # vision tokens end before end token
            break
    return start_idx, end_idx


def extract_attention_to_image(attentions: List[torch.Tensor], token_idx: int, img_start: int, img_end: int):
    """
    鲁棒版提取逻辑：
    1. 聚合倒数中间层 (Qwen2-VL-2B 推荐 layer 10-20)
    2. 解决 NaN 问题 (nan_to_num)
    3. 剔除 Sink Token (首尾 image token)
    4. 使用 Max 聚合而不是 Mean (让微弱的信号显现出来)
    """
    # Qwen2-VL-2B 有 28 层。选取中间层效果最好，避开顶层的语义抽象和底层的噪音
    # 比如取 [12, 24) 层
    selected_layers = attentions[12:24]
    
    aggregated_att = []
    for layer in selected_layers:
        if layer.dim() == 4:
            layer = layer[0]  # remove batch
        
        # 取出 (heads, img_tokens)
        # layer shape: [heads, seq_len, seq_len]
        # 我们要看: query=token_idx, key=img_start:img_end
        att_heads = layer[:, token_idx, img_start:img_end]
        
        # --- 防 NaN 关键步 ---
        # 如果模型之前算出了 NaN，这里把它变成 0
        att_heads = torch.nan_to_num(att_heads, nan=0.0, posinf=0.0, neginf=0.0)
        
        aggregated_att.append(att_heads)
    
    # stack shape: (num_layers, heads, img_tokens)
    stack = torch.stack(aggregated_att, dim=0)
    
    # --- 聚合策略 ---
    # 1. 先在层间取平均 (平滑层间差异)
    layer_mean = stack.mean(dim=0)  # (heads, img_tokens)
    
    # 2. 暴力去除 Sink Token
    # Qwen2-VL 的视觉特征序列中，第一个和最后一个通常是 attention sink
    # 注意：一定要先 clone，防止 inplace 操作报错
    layer_mean = layer_mean.clone()
    if layer_mean.shape[1] > 4: # 确保长度足够
        layer_mean[:, 0] = 0.0   # 杀左上角
        layer_mean[:, 1] = 0.0   # 杀左上角第二个
        layer_mean[:, -1] = 0.0  # 杀右下角
        layer_mean[:, -2] = 0.0  # 杀右下角第二个

    # 3. 在 Head 维度取 MAX 而不是 Mean
    # Mean 会被几十个“摸鱼”的 Head 拉低数值，导致一片黑
    # Max 能捕捉到那个“真正看对了”的 Head
    final_att, _ = layer_mean.max(dim=0) # (img_tokens,)
    
    # --- 后处理平滑 ---
    # 归一化前先开根号，提升暗部细节 (Gamma Correction)
    final_att = torch.pow(final_att, 0.5)
    
    return final_att, -1

def save_heatmap(image_path: str, att_flat: torch.Tensor, grid_thw: torch.Tensor, merge_size: int, out_path: str, title: str, head_idx: int):
    t, h, w = grid_thw.tolist()
    h_eff = max(1, h // merge_size)
    w_eff = max(1, w // merge_size)
    expect = t * h_eff * w_eff

    arr = att_flat.detach().float().cpu().numpy()
    if arr.shape[0] != expect:
        print(f"[warn] vision token count mismatch: got {arr.shape[0]}, expected {expect} (merge_size={merge_size})")
        size = min(arr.shape[0], expect)
        padded = np.zeros((expect,), dtype=arr.dtype)
        padded[:size] = arr[:size]
        arr = padded

    arr = arr.reshape(t, h_eff, w_eff).mean(axis=0)  # average over temporal dim
    arr = arr / (arr.max() + 1e-6)

    # 找到最高注意力的 patch，绘制矩形框
    max_idx = np.argmax(arr)
    max_h = max_idx // w_eff
    max_w = max_idx % w_eff

    img = Image.open(image_path).convert("RGB")
    # resize heatmap to image size for overlay
    heat = Image.fromarray((arr * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)
    heat_arr = np.array(heat) / 255.0

    print(f"Raw Max: {arr.max():.4f}, Raw Mean: {arr.mean():.4f}")

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.imshow(heat_arr, cmap="jet", alpha=0.4)
    # 在原图尺度标出最高注意力 patch 的边界
    box_w = img.width / w_eff
    box_h = img.height / h_eff
    x0 = max_w * box_w
    y0 = max_h * box_h
    rect = plt.Rectangle((x0, y0), box_w, box_h, linewidth=2, edgecolor="yellow", facecolor="none")
    plt.gca().add_patch(rect)
    plt.text(x0, y0 - 2, f"head {head_idx}", color="yellow", fontsize=8, bbox=dict(facecolor='black', alpha=0.4, pad=1))
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # 生成前20%注意力的mask图：保留高注意力区域，其他区域压暗
    mask_thresh = np.quantile(arr, 0.8)
    mask = (arr >= mask_thresh).astype(np.float32)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, resample=Image.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0  # (H, W)

    base = np.array(img, dtype=np.float32) / 255.0
    masked = base.copy()
    # 非mask区域压暗
    masked *= 0.2
    # mask区域高亮为红色叠加
    red_overlay = np.zeros_like(masked)
    red_overlay[..., 0] = 1.0
    masked = masked * (1 - mask_arr[..., None]) + red_overlay * mask_arr[..., None]
    masked = np.clip(masked, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(masked)
    plt.axis("off")
    plt.title(f"Top20% mask - {title}")
    plt.tight_layout()
    mask_out_path = os.path.splitext(out_path)[0] + "_mask.png"
    plt.savefig(mask_out_path)
    plt.close()
    return mask_out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/data/baidu.zijing.he/data/try.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--outdir", default="/data/baidu.zijing.he/code/EasyR1/attention_vis")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = args.device

    print(f"Loading model: {args.model}")
    if "2.5" in args.model and Qwen2_5_VLForConditionalGeneration is not None:
        print("Detected Qwen2.5-VL model, using Qwen2_5_VLForConditionalGeneration")
        model_cls = Qwen2_5_VLForConditionalGeneration
        processor_cls = Qwen2_5_VLProcessor
    else:
        print("Using Qwen2VLForConditionalGeneration")
        model_cls = Qwen2VLForConditionalGeneration
        processor_cls = Qwen2VLProcessor

    model = model_cls.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # required for output_attentions
    ).to(device)
    model.config.output_attentions = True
    model.config.use_cache = False

    processor = processor_cls.from_pretrained(args.model)

    items = load_items(args.json)

    for idx, item in enumerate(items):
        image_path = item["image_path"]
        steps = item.get("key_steps", {}).get("Steps", [])
        keywords = flatten_keywords(steps)
        prompt = item.get("problem", "")

        # 将关键词作为单独的 text block 追加，避免混入问句导致分词形态变化
        kw_block = "\n".join(keywords) if keywords else ""

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
            {"type": "text", "text": kw_block},
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        input_ids = inputs.input_ids[0].to(device)
        attention_mask = inputs.attention_mask[0].to(device)
        pixel_values = inputs.pixel_values.to(device)
        image_grid_thw = inputs.image_grid_thw.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_attentions=True,
                use_cache=False,
            )

        img_start, img_end = get_vision_span(processor.tokenizer, input_ids)
        if img_start < 0 or img_end < 0:
            print(f"[warn] cannot find vision tokens for item {idx}")
            continue

        pos_info = find_token_positions(processor.tokenizer, input_ids, keywords)
        
        all_atts = []
        
        for kw, positions in pos_info:
            if not positions:
                print(f"[warn] keyword '{kw}' not found in tokens for item {idx}")
                continue
            for p in positions:
                # att is (img_tokens,)
                att, _ = extract_attention_to_image(out.attentions, p, img_start, img_end)
                all_atts.append(att)
        
        if not all_atts:
            print(f"[warn] No valid tokens found for item {idx}")
            continue

        # 将所有 token 的注意力图堆叠 (N, img_tokens) 并取最大值
        # 这样能体现出“只要有一个 token 强烈关注某处，该处就被高亮”
        stacked_att = torch.stack(all_atts, dim=0)
        final_att, _ = stacked_att.max(dim=0)

        out_path = Path(args.outdir) / f"item{idx}_aggregated.png"
        save_heatmap(
            image_path,
            final_att,
            image_grid_thw[0],
            processor.image_processor.merge_size,
            str(out_path),
            title=f"Item {idx} Aggregated ({len(all_atts)} tokens)",
            head_idx=-1,
        )
        print(f"saved aggregated heatmap to {out_path}")


if __name__ == "__main__":
    main()
