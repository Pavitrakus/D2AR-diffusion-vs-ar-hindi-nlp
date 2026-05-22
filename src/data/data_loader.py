"""
Unified data loader for all 4 Hindi NLP tasks.

Handles downloading, caching, preprocessing, and splitting datasets
from HuggingFace for all tasks in the experiment matrix.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def load_dataset_for_task(
    task_name: str,
    split: str = "test",
    num_samples: Optional[int] = 200,
    cache_dir: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load dataset for a specific task.

    Args:
        task_name: One of 'Hindi_translation', 'Hindi_summary',
                   'Hindi_legal_bail', 'Hindi_judge_verdict'
        split: Dataset split ('train', 'validation', 'test')
        num_samples: Maximum number of samples (None for all)
        cache_dir: Directory to cache downloaded datasets

    Returns:
        Tuple of (inputs, targets) as string lists
    """
    from config.base_config import TaskName
    from config.task_configs import get_task_config

    task = TaskName(task_name)
    config = get_task_config(task)

    logger.info(f"Loading dataset: {config.hf_dataset_id} (split={split}, n={num_samples})")

    try:
        from datasets import load_dataset

        ds_kwargs = {"split": split}
        if cache_dir:
            ds_kwargs["cache_dir"] = cache_dir
        if config.hf_dataset_config:
            ds_kwargs["name"] = config.hf_dataset_config

        dataset = load_dataset(config.hf_dataset_id, **ds_kwargs)

        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))

        inputs = []
        targets = []

        for item in dataset:
            # Handle different dataset structures
            if task == TaskName.HINDI_TRANSLATION:
                if "translation" in item:
                    inp = item["translation"].get("en", item["translation"].get("src", ""))
                    tgt = item["translation"].get("hi", item["translation"].get("tgt", ""))
                elif config.input_column in item:
                    inp = item[config.input_column]
                    tgt = item[config.target_column]
                else:
                    continue
                inputs.append(str(inp))
                targets.append(str(tgt))

            elif task == TaskName.HINDI_SUMMARY:
                if config.input_column in item and config.target_column in item:
                    inputs.append(str(item[config.input_column]))
                    targets.append(str(item[config.target_column]))

            elif task in (TaskName.BAIL_PREDICTION, TaskName.JUDGE_VERDICT):
                if config.input_column in item:
                    inputs.append(str(item[config.input_column]))
                    label = item.get(config.label_column, item.get(config.target_column, 0))
                    if isinstance(label, int):
                        if task == TaskName.BAIL_PREDICTION:
                            targets.append("GRANTED" if label == 1 else "REJECTED")
                        else:
                            label_map = {0: "CONVICTED", 1: "ACQUITTED", 2: "REMANDED"}
                            targets.append(label_map.get(label, "CONVICTED"))
                    else:
                        targets.append(str(label))

        logger.success(f"Loaded {len(inputs)} samples for {task_name}")
        return inputs, targets

    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace: {e}. Using sample data.")
        return _get_sample_data(task_name, num_samples or 10)


def _get_sample_data(task_name: str, num_samples: int = 10) -> Tuple[List[str], List[str]]:
    """
    Get sample data for testing when HuggingFace datasets are unavailable.

    Returns realistic Hindi sample data for each task.
    """
    from config.base_config import TaskName
    task = TaskName(task_name)

    if task == TaskName.HINDI_TRANSLATION:
        samples = [
            ("The weather is very pleasant today.", "आज मौसम बहुत सुहावना है।"),
            ("India is a great country.", "भारत एक महान देश है।"),
            ("Education is very important.", "शिक्षा बहुत महत्वपूर्ण है।"),
            ("The Supreme Court gave its verdict.", "सर्वोच्च न्यायालय ने अपना फैसला सुनाया।"),
            ("Technology is changing the world.", "प्रौद्योगिकी दुनिया को बदल रही है।"),
            ("Health is wealth.", "स्वास्थ्य ही धन है।"),
            ("Knowledge is power.", "ज्ञान ही शक्ति है।"),
            ("Unity is strength.", "एकता में बल है।"),
            ("Time is money.", "समय पैसा है।"),
            ("Practice makes perfect.", "अभ्यास से सब कुछ संभव है।"),
            ("The river flows through the city.", "नदी शहर से होकर बहती है।"),
            ("Children are the future of the nation.", "बच्चे देश का भविष्य हैं।"),
        ]
        inputs = [s[0] for s in samples[:num_samples]]
        targets = [s[1] for s in samples[:num_samples]]

    elif task == TaskName.HINDI_SUMMARY:
        samples = [
            (
                "भारत के प्रधानमंत्री ने आज नई शिक्षा नीति की घोषणा की। इस नीति के तहत स्कूली शिक्षा में बड़े बदलाव किए जाएंगे। मातृभाषा में शिक्षा पर जोर दिया जाएगा। कोडिंग और डिजिटल साक्षरता को पाठ्यक्रम में शामिल किया जाएगा। इस नीति का उद्देश्य भारत को शिक्षा के क्षेत्र में विश्व स्तर पर प्रतिस्पर्धी बनाना है।",
                "प्रधानमंत्री ने नई शिक्षा नीति की घोषणा की जिसमें मातृभाषा में शिक्षा और डिजिटल साक्षरता पर जोर दिया गया है।"
            ),
            (
                "वैज्ञानिकों ने एक नई खोज की है जो कैंसर के इलाज में क्रांतिकारी बदलाव ला सकती है। नई दवा ने प्रयोगशाला में कैंसर कोशिकाओं को 90% तक नष्ट करने में सफलता पाई है। हालांकि, मानव परीक्षण अभी बाकी है और इसमें कई साल लग सकते हैं।",
                "वैज्ञानिकों ने एक नई कैंसर दवा खोजी है जो प्रयोगशाला में 90% कैंसर कोशिकाओं को नष्ट कर सकती है।"
            ),
            (
                "भारतीय क्रिकेट टीम ने आज ऑस्ट्रेलिया को फाइनल मैच में हराकर विश्व कप जीत लिया। कप्तान ने शानदार शतक लगाया और गेंदबाजों ने भी अच्छा प्रदर्शन किया। पूरे देश में जश्न का माहौल है।",
                "भारत ने ऑस्ट्रेलिया को हराकर क्रिकेट विश्व कप जीता, कप्तान ने शतक लगाया।"
            ),
        ]
        inputs = [s[0] for s in samples[:num_samples]]
        targets = [s[1] for s in samples[:num_samples]]

    elif task == TaskName.BAIL_PREDICTION:
        samples = [
            ("आरोपी पर धारा 420 के तहत धोखाधड़ी का आरोप है। कोई पूर्व रिकॉर्ड नहीं है।", "GRANTED"),
            ("आरोपी पर धारा 302 के तहत हत्या का गंभीर आरोप है। सबूत मजबूत हैं।", "REJECTED"),
            ("आरोपी पर चोरी का मामला है। सहयोग कर रहा है। पहली बार गिरफ्तार हुआ है।", "GRANTED"),
            ("आरोपी पर अपहरण और फिरौती का आरोप है। भागने का खतरा है।", "REJECTED"),
            ("आरोपी पर साइबर अपराध का मामला है। जांच पूरी हो चुकी है।", "GRANTED"),
        ]
        inputs = [s[0] for s in samples[:num_samples]]
        targets = [s[1] for s in samples[:num_samples]]

    elif task == TaskName.JUDGE_VERDICT:
        samples = [
            ("CCTV में आरोपी स्पष्ट दिख रहा है। गवाहों ने भी पहचान की। अपराध स्वीकार किया।", "CONVICTED"),
            ("कोई प्रत्यक्ष गवाह नहीं। मेडिकल रिपोर्ट अस्पष्ट। एलिबी सत्यापित हुई।", "ACQUITTED"),
            ("सबूत अपर्याप्त हैं। जांच में कमियां पाई गईं। मामला वापस भेजा जाता है।", "REMANDED"),
        ]
        inputs = [s[0] for s in samples[:num_samples]]
        targets = [s[1] for s in samples[:num_samples]]

    else:
        inputs = ["Sample input"] * num_samples
        targets = ["Sample target"] * num_samples

    return inputs, targets


def save_dataset_cache(
    inputs: List[str],
    targets: List[str],
    task_name: str,
    split: str,
    cache_dir: str = "data/cache",
) -> None:
    """Save processed dataset to local cache."""
    cache_path = Path(cache_dir) / f"{task_name}_{split}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"inputs": inputs, "targets": targets}, f, ensure_ascii=False, indent=2)
    logger.info(f"Cached {len(inputs)} samples to {cache_path}")


def load_dataset_cache(
    task_name: str,
    split: str,
    cache_dir: str = "data/cache",
) -> Optional[Tuple[List[str], List[str]]]:
    """Load processed dataset from local cache."""
    cache_path = Path(cache_dir) / f"{task_name}_{split}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["inputs"], data["targets"]
    return None
