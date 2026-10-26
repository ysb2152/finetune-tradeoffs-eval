import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from src.generate.qwen import MODEL_NAME

LEARNING_RATE={"full":2e-5,"lora":2e-4,"qlora":2e-4}

def load_model_and_tokenizer(regime) -> tuple:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.bfloat16)
    lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    qlora=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)

    if regime=="lora":

        model.enable_input_require_grads()
        model=get_peft_model(model,lora)
        
    elif regime=="qlora":

        model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,quantization_config=qlora)
        model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, lora)


    return (model , tokenizer)

def run(regime,train_path,out_dir) -> dict:

    model,tokenizer = load_model_and_tokenizer(regime)

    dataset=load_dataset("json",data_files=str(train_path) ,split="train")

    collator=DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant\n", tokenizer=tokenizer)
    config=SFTConfig(num_train_epochs=3,per_device_train_batch_size=1,gradient_accumulation_steps=8,output_dir=str(out_dir), learning_rate=LEARNING_RATE[regime],  optim="adamw_bnb_8bit", bf16=True, gradient_checkpointing=True, max_seq_length=1280, save_strategy="no")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, data_collator=collator)
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    trainer.train()
    wall=time.perf_counter() - t0
    peak=torch.cuda.max_memory_allocated()/ (1024**3)

    trainer.save_model(str(out_dir))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {"regime":regime , "wall_clock_sec":wall , "peak_vram_gb":peak , "trainable_params":trainable  , "total_params":total }



def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=str, choices=["full","lora","qlora"])
    parser.add_argument("train_path", type=Path)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로",required=True)
    args = parser.parse_args()

    report = run(args.regime,args.train_path, args.out)
    text = json.dumps(report, ensure_ascii=False, indent=3)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json" ).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
