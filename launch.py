import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

labels = [
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
    "기타 혐오",
    "악플/욕설",
    "clean",
]

model = AutoModelForSequenceClassification.from_pretrained(
    "smilegate-ai/kor_unsmile", use_auth_token=False
)
tokenizer = AutoTokenizer.from_pretrained(
    "smilegate-ai/kor_unsmile", use_auth_token=False
)


def checkHateSpeech(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)

        probs_by_labels = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
        return probs_by_labels


gr.Interface(
    fn=checkHateSpeech,
    inputs=gr.Textbox(label="Text"),
    outputs=gr.Label(
        label="Output Box (multi-label classification)", num_top_classes=5
    ),
    examples=[
        "사랑해요~",
        "ㅅㅂ 이딴게 세상이냐?",
        "남자 존나 싫어",
        "여자 존나 싫어",
        "게이 존나 싫어",
        "영감탱 존나 싫어",
        "흑인이 존나 싫어",
        "기독교 존나 싫어",
        "시골 존나 싫어",
        "저 정도면 지능장애",
    ],
).launch()
