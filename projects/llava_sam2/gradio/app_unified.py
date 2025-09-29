import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image  # for type hints

# Ensure repo root on sys.path for 'projects.' absolute imports when running directly
_CUR_FILE = os.path.abspath(__file__)
_ROOT = os.path.abspath(os.path.join(_CUR_FILE, "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Reuse utilities
from projects.llava_sam2.gradio.app_utils import (
    show_mask_pred,
    process_markdown,
    markdown_default,
)

DESCRIPTION = """
# Sa2VA Unified Demo

支持功能:
- 图像描述 / VQA (VQA/Caption)
- 基于文本的图像目标分割 (Segment by Text)
- 多轮对话 (Chat) 与可选的分割输出

提示:
1. 如果是第一次对话或任务需要视觉信息, 会自动在输入前添加 `<image>` 标记。
2. 分割任务会尝试解析答案中的 `prediction_masks` 并叠加显示。
3. 多轮对话模式会保持上下文 (Chat)。切换任务或点击 "清空会话" 可重置。
"""

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)

# ----------------------------- Argument Parsing ----------------------------- #


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sa2VA Unified Gradio App")
    parser.add_argument("hf_path", help="Path to HF exported Sa2VA model")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(TORCH_DTYPE_MAP.keys()),
        help="Torch dtype for loading",
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable public sharing (Gradio)"
    )
    return parser.parse_args(argv)


# ----------------------------- Model Loading ----------------------------- #


def load_model(model_path: str, dtype: str):
    torch_dtype = TORCH_DTYPE_MAP[dtype]
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


# ----------------------------- Core Inference ------------------------------ #


def build_input_dict(
    task: str,
    image: Optional[Image.Image],
    text: str,
    past_text: str,
    tokenizer,
) -> Dict[str, Any]:
    # auto prepend <image> if first turn
    if past_text == "" and "<image>" not in text:
        text = "<image>" + text
    input_dict = {
        "text": text,
        "past_text": past_text,
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    if image is not None:
        input_dict["image"] = image
    return input_dict


def overlay_masks(base_image, prediction_masks):
    if not prediction_masks:
        return base_image, []
    try:
        overlay_img, colors = show_mask_pred(base_image, prediction_masks)
        return overlay_img, colors
    except Exception as e:
        print("[WARN] Mask overlay failed:", e)
        return base_image, []


# ----------------------------- Gradio Handlers ----------------------------- #


def infer_handler(
    image,
    task,
    user_text,
    state: Dict[str, Any],
    chatbot: List[Tuple[str, str]],
):
    """Unified inference handler.

    state keys:
      past_text: internal model conversation serialization
      turns: int conversation turns
    """
    if state is None:
        state = {"past_text": "", "turns": 0}

    if image is None:
        return (
            None,
            chatbot,
            process_markdown("请先上传一张图片。", []),
            state,
        )

    text = user_text.strip()
    if not text:
        return image, chatbot, process_markdown("请输入指令。", []), state

    # Task specific phrasing (can be refined)
    if task == "Segment by Text" and ("segment" not in text.lower()):
        # encourage segmentation style phrase
        text = f"Please segment: {text}"  # simple decoration

    input_dict = build_input_dict(
        task=task,
        image=image,
        text=text,
        past_text=state["past_text"],
        tokenizer=tokenizer,
    )

    try:
        ret = sa2va_model.predict_forward(**input_dict)
    except Exception as e:
        err = f"推理失败: {e}"
        print(err)
        return image, chatbot, process_markdown(err, []), state

    # Update internal past_text for multi-turn
    state["past_text"] = ret.get("past_text", state["past_text"])  # safe
    state["turns"] += 1

    prediction = ret.get("prediction", "(No prediction text)").strip()
    masks = ret.get("prediction_masks", [])

    # Only overlay for segmentation-related tasks or if masks exist
    final_image = image
    color_list = []
    if masks and (task == "Segment by Text" or task == "Chat"):
        final_image, color_list = overlay_masks(image, masks)

    md_answer = process_markdown(prediction, color_list)
    chatbot.append((user_text, prediction))

    return final_image, chatbot, md_answer, state


def clear_history(state, chatbot):
    state = {"past_text": "", "turns": 0}
    return state, []


# ----------------------------- Launch UI ---------------------------------- #


def build_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="上传图像 (Image)")
                task = gr.Radio(
                    ["VQA/Caption", "Segment by Text", "Chat"],
                    value="VQA/Caption",
                    label="任务 (Task)",
                )
                user_text = gr.Textbox(
                    lines=2,
                    label="输入 / 指令 (Prompt)",
                    placeholder="Describe the image...",
                )
                run_btn = gr.Button("提交 / Submit", variant="primary")
                clear_btn = gr.Button("清空会话 / Clear")
            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="输出图像 (Result)")
                answer_md = gr.Markdown()
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="对话 (Chat)")

        state = gr.State({"past_text": "", "turns": 0})

        run_btn.click(
            infer_handler,
            inputs=[image, task, user_text, state, chatbot],
            outputs=[output_image, chatbot, answer_md, state],
        )
        clear_btn.click(
            clear_history, inputs=[state, chatbot], outputs=[state, chatbot]
        )

    return demo


# ----------------------------- Main --------------------------------------- #
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sa2va_model, tokenizer = load_model(args.hf_path, args.dtype)
    demo = build_interface()
    demo.queue()
    demo.launch(share=args.share)
