"""
Fine-tuning pipeline using LoRA/QLoRA.

Implements Parameter-Efficient Fine-Tuning (PEFT) for all 8 models
on Hindi NLP tasks. Uses LoRA (Low-Rank Adaptation) to fine-tune
large models with minimal memory overhead.

Supports:
- LoRA fine-tuning for auto-regressive models
- Custom fine-tuning loops for diffusion models
- Training curve logging and checkpoint management
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger


def fine_tune_model(
    model_wrapper,
    train_inputs: List[str],
    train_targets: List[str],
    task_config,
    output_dir: str = "results/checkpoints",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    save_steps: int = 100,
    logging_steps: int = 10,
) -> Dict[str, Any]:
    """
    Fine-tune a model using LoRA/QLoRA.

    Args:
        model_wrapper: Model wrapper instance (must be loaded)
        train_inputs: Training input texts
        train_targets: Training target texts
        task_config: Task configuration
        output_dir: Directory to save checkpoints
        ...other LoRA/training hyperparameters

    Returns:
        Dictionary with training metrics and checkpoint path
    """
    from config.model_configs import get_model_config

    model_config = model_wrapper.model_config
    output_path = Path(output_dir) / model_wrapper.model_name.value / task_config.name.value
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting LoRA fine-tuning: {model_wrapper.model_name.value} on {task_config.name.value}")
    logger.info(f"  Training samples: {len(train_inputs)}")
    logger.info(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    logger.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}")

    start_time = time.time()

    if model_config.model_type.value == "autoregressive" and model_config.architecture == "causal_lm":
        result = _fine_tune_causal_lm(
            model_wrapper=model_wrapper,
            train_inputs=train_inputs,
            train_targets=train_targets,
            task_config=task_config,
            output_path=output_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_seq_length=max_seq_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
        )
    elif model_config.architecture == "masked_lm":
        result = _fine_tune_masked_lm(
            model_wrapper=model_wrapper,
            train_inputs=train_inputs,
            train_targets=train_targets,
            task_config=task_config,
            output_path=output_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
    else:
        # Diffusion models: custom fine-tuning loop
        result = _fine_tune_diffusion(
            model_wrapper=model_wrapper,
            train_inputs=train_inputs,
            train_targets=train_targets,
            task_config=task_config,
            output_path=output_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    elapsed = time.time() - start_time
    result["total_training_time_seconds"] = elapsed

    # Save training results
    results_path = output_path / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.success(f"Fine-tuning complete in {elapsed / 60:.1f} minutes")
    logger.info(f"  Final loss: {result.get('final_loss', 'N/A')}")
    logger.info(f"  Checkpoint saved to: {output_path}")

    return result


def _fine_tune_causal_lm(
    model_wrapper,
    train_inputs: List[str],
    train_targets: List[str],
    task_config,
    output_path: Path,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> Dict[str, Any]:
    """Fine-tune causal LM (LLaMA, Gemma, Mistral) with LoRA."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from datasets import Dataset

        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        # Prepare model for LoRA
        if model_wrapper.load_in_4bit or model_wrapper.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=model_wrapper.model_config.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Prepare training data
        def format_sample(input_text, target_text):
            if task_config.task_type == "generation":
                return f"Input: {input_text}\nOutput: {target_text}"
            else:
                return f"Text: {input_text}\nLabel: {target_text}"

        train_texts = [format_sample(inp, tgt) for inp, tgt in zip(train_inputs, train_targets)]

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )

        dataset = Dataset.from_dict({"text": train_texts})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Save LoRA adapters
        model.save_pretrained(str(output_path / "lora_adapters"))
        tokenizer.save_pretrained(str(output_path / "tokenizer"))

        # Update wrapper's model reference
        model_wrapper.model = model

        training_history = trainer.state.log_history

        return {
            "status": "success",
            "method": "LoRA",
            "final_loss": train_result.training_loss,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params / total_params,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "checkpoint_path": str(output_path / "lora_adapters"),
            "training_history": training_history,
        }

    except Exception as e:
        logger.error(f"LoRA fine-tuning failed: {e}")
        return {"status": "failed", "error": str(e), "method": "LoRA"}


def _fine_tune_masked_lm(
    model_wrapper, train_inputs, train_targets, task_config,
    output_path, num_epochs, learning_rate, batch_size,
) -> Dict[str, Any]:
    """Fine-tune masked language model (IndicBERT) for classification."""
    try:
        from transformers import (
            AutoModelForSequenceClassification, TrainingArguments, Trainer
        )
        from datasets import Dataset

        tokenizer = model_wrapper.tokenizer
        num_labels = task_config.num_classes if task_config.task_type == "classification" else 2

        # Load classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_wrapper.model_config.hf_model_id,
            num_labels=num_labels,
        )

        if torch.cuda.is_available():
            model = model.to(model_wrapper.device)

        # Prepare dataset
        label_map = {label: i for i, label in enumerate(task_config.class_labels)}

        def tokenize_and_label(examples):
            tokenized = tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length",
            )
            labels = []
            for t in examples["label"]:
                mapped = label_map.get(t.strip().upper(), 0)
                labels.append(mapped)
            tokenized["labels"] = labels
            return tokenized

        dataset = Dataset.from_dict({"text": train_inputs, "label": train_targets})
        tokenized = dataset.map(tokenize_and_label, batched=True, remove_columns=["text", "label"])

        training_args = TrainingArguments(
            output_dir=str(output_path), num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size, learning_rate=learning_rate,
            logging_steps=10, save_total_limit=2, report_to="none",
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
        train_result = trainer.train()

        model.save_pretrained(str(output_path / "finetuned_model"))
        tokenizer.save_pretrained(str(output_path / "tokenizer"))

        return {
            "status": "success", "method": "full_finetuning",
            "final_loss": train_result.training_loss,
            "num_epochs": num_epochs,
        }
    except Exception as e:
        logger.error(f"Masked LM fine-tuning failed: {e}")
        return {"status": "failed", "error": str(e)}


def _fine_tune_diffusion(
    model_wrapper, train_inputs, train_targets, task_config,
    output_path, num_epochs, learning_rate, batch_size,
) -> Dict[str, Any]:
    """
    Fine-tune diffusion models (SEDD, LLaDA, D3PM, DiffuseLM).

    Uses a custom training loop since diffusion models have different
    training objectives than standard LMs.
    """
    try:
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_inputs) // batch_size)

        model.train()
        training_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(train_inputs), batch_size):
                batch_inputs = train_inputs[i:i + batch_size]
                batch_targets = train_targets[i:i + batch_size]

                # Combine input and target
                batch_texts = [f"{inp} {tgt}" for inp, tgt in zip(batch_inputs, batch_targets)]

                # Tokenize
                encoded = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )

                if torch.cuda.is_available():
                    encoded = {k: v.to(model_wrapper.device) for k, v in encoded.items()}

                input_ids = encoded["input_ids"]

                # Diffusion training: add noise and predict denoising
                t = torch.rand(input_ids.shape[0], 1, device=input_ids.device)

                if hasattr(model_wrapper, '_add_noise'):
                    noised = model_wrapper._add_noise(input_ids, t.mean().item())
                else:
                    noise_mask = torch.rand_like(input_ids.float()) < t
                    noised = torch.where(
                        noise_mask,
                        torch.randint(0, tokenizer.vocab_size, input_ids.shape, device=input_ids.device),
                        input_ids,
                    )

                # Forward pass
                if hasattr(model, 'denoise'):
                    # DiffuseLM style
                    embeddings = model.embedding(noised)
                    predicted = model.denoise(embeddings, t)
                    target_emb = model.embedding(input_ids)
                    loss = torch.nn.functional.mse_loss(predicted, target_emb)
                else:
                    # SEDD/LLaDA/D3PM style
                    logits = model(noised, t)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        input_ids.view(-1),
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            training_history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
            })
            logger.info(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()

        # Save model
        torch.save(model.state_dict(), str(output_path / "diffusion_finetuned.pt"))

        return {
            "status": "success",
            "method": "diffusion_finetuning",
            "final_loss": training_history[-1]["loss"] if training_history else 0,
            "training_history": training_history,
            "num_epochs": num_epochs,
        }

    except Exception as e:
        logger.error(f"Diffusion fine-tuning failed: {e}")
        return {"status": "failed", "error": str(e)}
