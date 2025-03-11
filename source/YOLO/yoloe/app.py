import torch
import numpy as np
import gradio as gr
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from gradio_image_prompter import ImagePrompter
from huggingface_hub import hf_hub_download

def init_model(model_id, is_pf=False):
    if not is_pf:
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=f"{model_id}-seg.pt")
        model = YOLOE(path)
    else:
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=f"{model_id}-seg-pf.pt")
        model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type):
    model = init_model(model_id)
    kwargs = {}
    if prompt_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    elif prompt_type == "Visual":
        kwargs = dict(
            prompts=prompts,
            predictor=YOLOEVPSegPredictor
        )
        if target_image:
            model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
            model.set_classes(["object0"], model.predictor.vpe)
            model.predictor = None  # unset VPPredictor
            image = target_image
            kwargs = {}
    elif prompt_type == "Prompt-free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True)
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000

    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    raw_image = gr.Image(type="pil", label="Image", visible=True, interactive=True)
                    box_image = ImagePrompter(type="pil", label="DrawBox", visible=False, interactive=True)
                    mask_image = gr.ImageEditor(type="pil", label="DrawMask", visible=False, interactive=True, layers=False, canvas_size=(640, 640))
                    target_image = gr.Image(type="pil", label="Target Image", visible=False, interactive=True)
                
                yoloe_infer = gr.Button(value="Detect & Segment Objects")
                prompt_type = gr.Textbox(value="Text", visible=False)

                with gr.Tab("Text") as text_tab:
                    texts = gr.Textbox(label="Input Texts", value='person,bus', placeholder='person,bus', visible=True, interactive=True)
                
                with gr.Tab("Visual") as visual_tab:
                    with gr.Row():
                        visual_prompt_type = gr.Dropdown(choices=["bboxes", "masks"], value="bboxes", label="Visual Type", interactive=True)
                        visual_usage_type = gr.Radio(choices=["Intra-Image", "Inter-Image"], value="Intra-Image", label="Intra/Inter Image", interactive=True)
                
                with gr.Tab("Prompt-Free") as prompt_free_tab:
                    gr.HTML(
                        """
                        <p style='text-align: center'>
                        Prompt-Free Mode is On
                        </p>
                    """, show_label=False)

                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yoloe-v8s",
                        "yoloe-v8m",
                        "yoloe-v8l",
                        "yoloe-11s",
                        "yoloe-11m",
                        "yoloe-11l",
                    ],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_thresh = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                iou_thresh = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.70,
                )

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
        
        def update_text_image_visibility():
            return gr.update(value="Text"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        def update_visual_image_visiblity(visual_prompt_type, visual_usage_type):
            use_target = gr.update(visible=True) if visual_usage_type == "Inter-Image" else gr.update(visible=False)
            if visual_prompt_type == "bboxes":
                return gr.update(value="Visual"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), use_target
            elif visual_prompt_type == "masks":
                return gr.update(value="Visual"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), use_target

        def update_pf_image_visibility():
            return gr.update(value="Prompt-free"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        text_tab.select(
            fn=update_text_image_visibility,
            inputs=None,
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
        )
        
        visual_tab.select(
            fn=update_visual_image_visiblity,
            inputs=[visual_prompt_type, visual_usage_type],
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
        )
        
        prompt_free_tab.select(
            fn=update_pf_image_visibility,
            inputs=None,
            outputs=[prompt_type, raw_image, box_image, mask_image, target_image]
        )

        def update_visual_prompt_type(visual_prompt_type):
            if visual_prompt_type == "bboxes":
                return gr.update(visible=True), gr.update(visible=False)
            if visual_prompt_type == "masks":
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        def update_visual_usage_type(visual_usage_type):
            if visual_usage_type == "Intra-Image":
                return gr.update(visible=False, value=None)
            if visual_usage_type == "Inter-Image":
                return gr.update(visible=True, value=None)
            return gr.update(visible=False, value=None)

        visual_prompt_type.change(
            fn=update_visual_prompt_type,
            inputs=[visual_prompt_type],
            outputs=[box_image, mask_image]
        )

        visual_usage_type.change(
            fn=update_visual_usage_type,
            inputs=[visual_usage_type],
            outputs=[target_image]
        )

        def run_inference(raw_image, box_image, mask_image, target_image, texts, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type):
            # add text/built-in prompts
            if prompt_type == "Text" or prompt_type == "Prompt-free":
                image = raw_image
                if prompt_type == "Prompt-free":
                    with open('tools/ram_tag_list.txt', 'r') as f:
                        texts = [x.strip() for x in f.readlines()]
                else:
                    texts = [text.strip() for text in texts.split(',')]
                prompts = {
                    "texts": texts
                }
            # add visual prompt
            elif prompt_type == "Visual":
                if visual_prompt_type == "bboxes":
                    image, points = box_image["image"], box_image["points"]
                    points = np.array(points)
                    prompts = {
                        "bboxes": np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2]),
                    }
                elif visual_prompt_type == "masks":
                    image, masks = mask_image["background"], mask_image["layers"][0]
                    # image = image.convert("RGB")
                    masks = np.array(masks.convert("L"))
                    masks = binary_fill_holes(masks).astype(np.uint8)
                    masks[masks > 0] = 1
                    prompts = {
                        "masks": masks[None]
                    }
            return yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type)

        yoloe_infer.click(
            fn=run_inference,
            inputs=[raw_image, box_image, mask_image, target_image, texts, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type],
            outputs=[output_image],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    <img src="/file=figures/logo.png" width="2.5%" style="display:inline;padding-bottom:4px">
    YOLOE: Real-Time Seeing Anything
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2503.07465' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yoloe' target='_blank'>github</a>
        </h3>
        """)
    gr.Markdown(
        """
        We introduce **YOLOE(ye)**, a highly **efficient**, **unified**, and **open** object detection and segmentation model, like human eye, under different prompt mechanisms, like *texts*, *visual inputs*, and *prompt-free paradigm*.
        """
    )
    gr.Markdown(
        """
        If desired objects are not identified, pleaset set a **smaller** confidence threshold, e.g., for visual prompts with handcrafted shape or cross-image prompts.
        """
    )
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch(allowed_paths=["figures"])