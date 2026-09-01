import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dataset.sac_compat  # noqa: F401  # 须先于 torch 导入 datasets（issue #771 DLL 顺序），并绕开 Smart App Control 对 pyarrow.dataset 的拦截
import argparse
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                On-Policy Distillation 散度项  (参考 verl algo/opd)
#   L_OPD(θ) = E[x~p_data, y~π_θ(·|x)] [ 1/|y| · Σ_t D(π_θ(·|s_t), ν(·|s_t), y_t) ]
#   π_θ = 学生策略（可训练）， ν = 教师策略（冻结）， s_t = (x, y_<t)
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
def distill_divergence(student_logits, teacher_logits, sampled_ids, loss_mode="k3", topk=32):
    """逐 token 散度，返回 [B, R]。student_logits 保留梯度，teacher_logits 已 detach。

    k3              : 采样 token 上的低方差反向 KL 估计量 exp(r)-r-1，恒 >= 0
    forward_kl      : 全词表前向 KL  Σ_v ν(v)[log ν(v) - log π(v)]
    forward_kl_topk : 仅在教师 top-k 上计算前向 KL（verl 默认 topk=32）
    """
    if loss_mode == "k3":
        # 只需采样位置的 logprob，显存开销最小
        s_lp = F.log_softmax(student_logits, dim=-1).gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        t_lp = F.log_softmax(teacher_logits, dim=-1).gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        r = t_lp - s_lp
        return torch.exp(r) - r - 1.0

    t_logprobs = F.log_softmax(teacher_logits, dim=-1)
    s_logprobs = F.log_softmax(student_logits, dim=-1)

    if loss_mode == "forward_kl":
        t_probs = t_logprobs.exp()
        return (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)

    if loss_mode == "forward_kl_topk":
        k = min(topk, t_logprobs.size(-1))
        top_lp, top_idx = torch.topk(t_logprobs, k, dim=-1)          # [B, R, k]
        top_p = top_lp.exp()
        s_top = s_logprobs.gather(-1, top_idx)
        return (top_p * (top_lp - s_top)).sum(dim=-1)

    raise ValueError(f"未知的 loss_mode: {loss_mode}")


def opd_train_epoch(epoch, loader, iters, rollout_engine, teacher_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ---- 1) 学生 on-policy 采样；rollout 视为固定，不回传梯度 ----
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=args.rollout_temperature,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        # 采样出来的 token（即 y_t），用于 k3 估计量
        sampled_ids = outputs[:, 1:].gather(1, logp_pos)

        # ---- 2) 学生前向（带梯度）----
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            V = res.logits.size(-1)
            s_logits = res.logits[:, :-1, :].gather(
                1, logp_pos.unsqueeze(-1).expand(-1, -1, V))            # [B, R, V]

        # ---- 3) 教师前向（冻结，温度恒为 1.0）----
        with torch.no_grad():
            t_logits_full = teacher_model(outputs, attention_mask=full_mask).logits[:, :-1, :]
            t_logits = t_logits_full.gather(1, logp_pos.unsqueeze(-1).expand(-1, -1, t_logits_full.size(-1)))
            t_logits = t_logits[..., :V].float()                        # 教师词表可能更大时对齐学生
            del t_logits_full

        # ---- 4) 逐 token 散度 ----
        per_token_loss = distill_divergence(s_logits.float(), t_logits, sampled_ids,
                                            loss_mode=args.loss_mode, topk=args.topk)
        if args.loss_max_clamp is not None:
            per_token_loss = per_token_loss.clamp(-args.loss_max_clamp, args.loss_max_clamp)

        # ---- 5) 只统计到 EOS 为止的回答 token；按 1/|y| 归一后对 batch 取均值 ----
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)
                            <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()

        distill_loss = ((per_token_loss * completion_mask).sum(dim=1)
                        / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (args.distillation_loss_coef * distill_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            avg_len = completion_mask.sum(dim=1).float().mean().item()
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Distill Loss: {distill_loss.item():.4f}, Mode: {args.loss_mode}, '
                   f'Aux Loss: {aux_loss.item():.4f}, Avg Response Len: {avg_len:.2f}, '
                   f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
            if wandb and is_main_process():
                wandb.log({"distill_loss": distill_loss.item(), "aux_loss": aux_loss.item(),
                           "avg_response_len": avg_len, "learning_rate": optimizer.param_groups[0]['lr']})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, s_logits, t_logits, per_token_loss, completion_mask

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind OPD (On-Policy Distillation)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='opd', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    # ---- 学生 ----
    parser.add_argument('--hidden_size', default=768, type=int, help="学生隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="学生隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="学生是否使用MoE")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="学生基于哪个权重")
    # ---- 教师 ----
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师隐藏层数量")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师是否使用MoE")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师基于哪个权重")
    parser.add_argument('--teacher_dtype', default='bfloat16', type=str, choices=['bfloat16', 'float16', 'float32'],
                        help="教师权重存放精度，教师只前向不回传，半精度足够且省显存")
    # ---- OPD 超参（对齐 verl distillation_loss.*）----
    parser.add_argument('--loss_mode', default='k3', type=str,
                        choices=['k3', 'forward_kl', 'forward_kl_topk'], help="散度类型")
    parser.add_argument('--topk', default=32, type=int, help="forward_kl_topk 的 k")
    parser.add_argument('--distillation_loss_coef', default=1.0, type=float, help="蒸馏损失权重")
    parser.add_argument('--loss_max_clamp', default=None, type=float, help="逐token损失裁剪，None为不裁剪")
    # ---- rollout ----
    parser.add_argument('--max_seq_len', default=512, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=256, help="生成的最大长度")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--rollout_temperature", type=float, default=1.0, help="学生采样温度")
    parser.add_argument("--thinking_ratio", type=float, default=0.5, help="开启thinking的概率")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_opd")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="Prompt数据路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-OPD")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    teacher_config = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers,
                                    max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project,
                   name=f"MiniMind-OPD-{args.loss_mode}-BS-{args.batch_size}-LR-{args.learning_rate}",
                   id=wandb_id, resume='must' if wandb_id else None)

    # ========== 5. 初始化模型和数据 ==========
    # 学生（可训练）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 教师（冻结）
    teacher_model, _ = init_model(teacher_config, args.from_teacher_weight, device=args.device)
    teacher_model = teacher_model.eval().requires_grad_(False)
    # 教师只产 logits 当 KL 目标，不回传梯度，半精度存放足够。
    # 注意教师前向在 no_grad 里但不在 autocast 里，所以 fp32 权重会让整个
    # 前向和 [B, S-1, 6400] 的 logits 都跑在 fp32 上。198M 的 MoE 教师
    # 光权重就 793MB，转 bf16 后 396MB，logits 也减半 —— 这张 8GB 卡上
    # GRPO(actor+ref) 已经用到 6.89GB，省下的正是仅剩的那点余量。
    # 下游 t_logits 会 .float() 回来算散度，数值精度不受影响。
    if args.teacher_dtype != 'float32':
        teacher_model = teacher_model.to(dtype=torch.bfloat16 if args.teacher_dtype == 'bfloat16' else torch.float16)
    Logger(f'教师权重精度: {args.teacher_dtype}')
    Logger(f'教师: hidden={args.teacher_hidden_size} layers={args.teacher_num_layers} '
           f'moe={bool(args.teacher_use_moe)} weight={args.from_teacher_weight}')
    Logger(f'OPD 配置: loss_mode={args.loss_mode} topk={args.topk} '
           f'coef={args.distillation_loss_coef} num_generations={args.num_generations}')
    # Rollout 引擎（学生 on-policy 采样）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine, policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url, sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            opd_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, teacher_model, start_step, wandb)
        else:
            opd_train_epoch(epoch, loader, len(loader), rollout_engine, teacher_model, 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
