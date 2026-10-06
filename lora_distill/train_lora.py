#!/usr/bin/env python3
"""
LoRA SFT of a small model on the distillation set from build_dataset.py (plain PyTorch + peft).

Defaults: r=16, alpha=32, target all linear layers, lr 2e-4, 3 epochs, batch 32 (gradient accumulation over
--micro_batch), AdamW, cosine schedule with 10% warmup, grad-clip 1.0, bf16, loss only on the response tokens
(prompt = the chat-template prompt the model sees at test time). The adapter is saved after every epoch:
  python lora_distill/train_lora.py --train_file logs/lora_32b_to_3b/train.jsonl --out_dir logs/lora_32b_to_3b/lora
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F


def build_examples(tok, rows, max_length):
    """(input_ids, labels) with -100 on the prompt; the response ends with the eos token."""
    examples, dropped = [], 0
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(r["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(a) > max_length:
            dropped += 1
            continue
        examples.append((p + a, [-100] * len(p) + a))
    return examples, dropped


def collate(batch, pad_id, device):
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def train(model, examples, *, lr, epochs, batch_size, micro_batch, warmup_ratio, seed, pad_id, device,
          log_path=None, on_epoch_end=None):
    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    gen = torch.Generator().manual_seed(seed)
    use_amp = device.startswith("cuda")
    model.train()
    log_file = open(log_path, "w") if log_path else None
    t0, step = time.time(), 0
    for epoch in range(epochs):
        perm = torch.randperm(len(examples), generator=gen).tolist()
        for s in range(steps_per_epoch):
            batch = [examples[i] for i in perm[s * batch_size:(s + 1) * batch_size]]
            n_tok = sum(sum(1 for t in lab if t != -100) for _, lab in batch)   # normalise over the whole batch
            loss_sum = 0.0
            for m in range(0, len(batch), micro_batch):
                ids, labels, mask = collate(batch[m:m + micro_batch], pad_id, device)
                with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                       labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / n_tok
                loss.backward()
                loss_sum += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch + 1, "loss": loss_sum, "grad_norm": grad_norm,
                   "lr": sched.get_last_lr()[0], "tokens": n_tok, "elapsed_s": time.time() - t0}
            print(f"step {step}/{total} epoch {epoch + 1} loss {loss_sum:.4f} grad_norm {grad_norm:.2f} "
                  f"lr {rec['lr']:.2e} ({rec['elapsed_s']:.0f}s)", flush=True)
            if log_file:
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
        if on_epoch_end:
            on_epoch_end(epoch + 1)
    if log_file:
        log_file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--micro_batch", type=int, default=4, help="examples per forward pass; lower it if out of memory")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_checkpointing", action="store_true")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    rows = [json.loads(line) for line in open(args.train_file, encoding="utf-8") if line.strip()]
    tok = AutoTokenizer.from_pretrained(args.model_name)
    examples, dropped = build_examples(tok, rows, args.max_length)
    print(f"{len(examples)} examples ({dropped} dropped as longer than {args.max_length} tokens), "
          f"{math.ceil(len(examples) / args.batch_size)} steps per epoch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.to(device)

    def save(epoch):
        path = os.path.join(args.out_dir, f"epoch_{epoch}")
        model.save_pretrained(path)
        print(f"saved {path}", flush=True)

    train(model, examples, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, micro_batch=args.micro_batch,
          warmup_ratio=args.warmup_ratio, seed=args.seed, pad_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
          device=device, log_path=os.path.join(args.out_dir, "train_log.jsonl"), on_epoch_end=save)


if __name__ == "__main__":
    main()
