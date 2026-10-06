"""Helpers shared by the scaled-up distillation scripts (label_teacher / label_student / build_big_dataset)."""
import json
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# distillation train sets: name -> (data handler, train file; None = the handler's default train file)
SOURCES = {"math": ("math500", "data/math-train/train.jsonl"), "gsm8k": ("gsm8k", None)}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_train_problems(names):
    """{name: (handler, datas)}: every problem of the train file of each source."""
    from data_handlers import get_dataset_handler
    out = {}
    for n in names:
        handler_name, path = SOURCES[n]
        handler = get_dataset_handler(handler_name)
        path = path or handler.default_train_path
        if not os.path.exists(path):
            sys.exit(f"{path} not found (python scripts/prepare_math_train.py / prepare_gsm8k.py)")
        out[n] = (handler, handler.load_data(path, split="train"))
    return out


def build_prompts(tokenizer, problems, max_prompt_tokens):
    """({(name, idx): prompt}, number dropped for being longer than max_prompt_tokens)."""
    prompts, dropped = {}, 0
    for name, (_, datas) in problems.items():
        texts = [tokenizer.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
                 for d in datas]
        lengths = [len(x) for x in tokenizer(texts, add_special_tokens=False)["input_ids"]]
        for idx, (text, n) in enumerate(zip(texts, lengths)):
            if n <= max_prompt_tokens:
                prompts[(name, idx)] = text
            else:
                dropped += 1
    return prompts, dropped


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def add_engine_args(p, tp=1, base_on_cpu=False, gpu_memory_utilization=0.85, max_num_seqs=256):
    p.add_argument("--model_name", required=True)
    p.add_argument("--tp", type=int, default=tp)
    p.add_argument("--base_on_cpu", action="store_true", default=base_on_cpu)
    p.add_argument("--gpu_memory_utilization", type=float, default=gpu_memory_utilization)
    p.add_argument("--max_num_seqs", type=int, default=max_num_seqs)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_prompt_tokens", type=int, default=2500)
    p.add_argument("--no_cuda_graphs", action="store_true")
    p.add_argument("--cuda_devices", default=None)


def make_engine(args):
    """evaluate.DirectEngine (plain vllm.LLM; the worker extension gives perturb/reset)."""
    import evaluate as ev
    return ev.DirectEngine(types.SimpleNamespace(
        cuda_devices=args.cuda_devices, model_name=args.model_name, precision="bfloat16",
        max_num_seqs=args.max_num_seqs, cuda_graphs=not args.no_cuda_graphs, prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization, procs_per_gpu=1, tp=args.tp,
        base_on_cpu=args.base_on_cpu, max_model_len=args.max_model_len))
