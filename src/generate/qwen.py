import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You answer questions about a scientific paper using only the provided evidence. "
    "Answer concisely with a short phrase or a single sentence, not a paragraph. "
    "For yes/no questions, answer 'Yes' or 'No'. "
    "If the evidence does not contain the answer, reply exactly with 'Unanswerable'."
)

# Qwen2.5-1.5B를 bf16 로드. adapter_path를 주면 base 위에 LoRA/QLoRA 어댑터를 얹어 base에 합친다

def load_generator(model_name: str = MODEL_NAME, adapter_path=None):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,torch_dtype=torch.bfloat16)
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    model.to("cuda").eval()
    return model, tokenizer

# 근거 문단과 질문을 system/user 메시지로 조립. 근거 없으면 '(근거 없음)'

def build_messages(question: str, evidence: list[str]) -> list[dict]:
    joined = "\n\n".join(evidence) if evidence else "(근거 없음)"
    user = f"Evidence:\n{joined}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

# 근거로 답을 생성해 문자열로 돌려준다(그리디 디코딩)

@torch.no_grad()
def generate_answer(
    question: str,
    evidence: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(question, evidence), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
