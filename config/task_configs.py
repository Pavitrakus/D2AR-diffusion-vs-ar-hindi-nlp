"""
Task-specific configurations for all 4 Hindi NLP tasks.

Contains dataset paths, prompts for zero-shot/few-shot, evaluation metrics,
and task-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from config.base_config import TaskName


@dataclass
class TaskConfig:
    """Configuration for a specific NLP task."""

    # Identification
    name: TaskName
    display_name: str
    description: str
    category: str  # "language" or "legal"
    task_type: str  # "generation" or "classification"

    # Dataset
    hf_dataset_id: str
    hf_dataset_config: Optional[str] = None
    hf_dataset_split_train: str = "train"
    hf_dataset_split_test: str = "test"
    hf_dataset_split_validation: Optional[str] = "validation"
    alternative_datasets: List[str] = field(default_factory=list)

    # Data columns
    input_column: str = "input"
    target_column: str = "target"
    label_column: Optional[str] = None

    # Classification
    num_classes: int = 2
    class_labels: List[str] = field(default_factory=list)

    # Evaluation metrics
    primary_metric: str = "bleu"
    all_metrics: List[str] = field(default_factory=list)

    # Prompts
    system_prompt: str = ""
    zero_shot_template: str = ""
    few_shot_template: str = ""
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)

    # Data processing
    max_input_length: int = 512
    max_target_length: int = 256
    num_train_samples: Optional[int] = 5000
    num_eval_samples: int = 200

    # RAG
    rag_corpus_description: str = ""
    rag_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# Task Configurations
# ============================================================

TASK_CONFIGS: Dict[TaskName, TaskConfig] = {

    # --------------------------------------------------------
    # TASK 1: Hindi Translation (English → Hindi)
    # --------------------------------------------------------
    TaskName.HINDI_TRANSLATION: TaskConfig(
        name=TaskName.HINDI_TRANSLATION,
        display_name="English to Hindi Translation",
        description="Translate English text to Hindi using various language models",
        category="language",
        task_type="generation",

        # Dataset: IIT Bombay English-Hindi Parallel Corpus
        hf_dataset_id="cfilt/iitb-english-hindi",
        hf_dataset_split_train="train",
        hf_dataset_split_test="test",
        hf_dataset_split_validation="validation",
        alternative_datasets=[
            "ai4bharat/samanantar",
            "Helsinki-NLP/opus-100",
        ],

        input_column="en",
        target_column="hi",

        # Metrics
        primary_metric="bleu",
        all_metrics=["bleu", "chrf", "bertscore", "comet", "ter"],

        # Prompts
        system_prompt=(
            "You are an expert English to Hindi translator. "
            "Translate the given English text accurately into Hindi, "
            "preserving the meaning, tone, and cultural context. "
            "Output only the Hindi translation without any explanation."
        ),
        zero_shot_template=(
            "Translate the following English text to Hindi.\n\n"
            "English: {input}\n\n"
            "Hindi:"
        ),
        few_shot_template=(
            "Translate the following English text to Hindi. "
            "Here are some examples:\n\n"
            "{examples}\n\n"
            "Now translate:\n"
            "English: {input}\n\n"
            "Hindi:"
        ),
        few_shot_examples=[
            {
                "input": "The weather is very pleasant today.",
                "output": "आज मौसम बहुत सुहावना है।"
            },
            {
                "input": "India is a country of diverse cultures and traditions.",
                "output": "भारत विविध संस्कृतियों और परंपराओं का देश है।"
            },
            {
                "input": "Education is the most powerful weapon to change the world.",
                "output": "शिक्षा दुनिया को बदलने का सबसे शक्तिशाली हथियार है।"
            },
            {
                "input": "The Supreme Court delivered its verdict yesterday.",
                "output": "सर्वोच्च न्यायालय ने कल अपना फैसला सुनाया।"
            },
            {
                "input": "Technology has transformed the way we communicate.",
                "output": "प्रौद्योगिकी ने हमारे संवाद करने के तरीके को बदल दिया है।"
            },
        ],

        max_input_length=256,
        max_target_length=256,
        num_train_samples=5000,
        num_eval_samples=200,

        rag_corpus_description="Hindi-English parallel sentences from Wikipedia and news articles",
    ),

    # --------------------------------------------------------
    # TASK 2: Hindi Summarization
    # --------------------------------------------------------
    TaskName.HINDI_SUMMARY: TaskConfig(
        name=TaskName.HINDI_SUMMARY,
        display_name="Hindi Text Summarization",
        description="Generate concise summaries of Hindi text passages",
        category="language",
        task_type="generation",

        # Dataset: Hindi text summarization
        hf_dataset_id="csebuetnlp/xlsum",
        hf_dataset_config="hindi",
        hf_dataset_split_train="train",
        hf_dataset_split_test="test",
        hf_dataset_split_validation="validation",
        alternative_datasets=[
            "ai4bharat/IndicSentenceSummarization",
            "rahular/varta",
        ],

        input_column="text",
        target_column="summary",

        # Metrics
        primary_metric="rouge1",
        all_metrics=["rouge1", "rouge2", "rougeL", "bertscore", "bleu"],

        # Prompts
        system_prompt=(
            "You are an expert Hindi text summarizer. "
            "Read the given Hindi passage carefully and generate a concise, "
            "accurate summary in Hindi. The summary should capture the key points "
            "and maintain the factual accuracy of the original text. "
            "Output only the Hindi summary."
        ),
        zero_shot_template=(
            "निम्नलिखित हिंदी पाठ का सारांश लिखें।\n\n"
            "पाठ: {input}\n\n"
            "सारांश:"
        ),
        few_shot_template=(
            "निम्नलिखित हिंदी पाठ का सारांश लिखें। "
            "यहाँ कुछ उदाहरण दिए गए हैं:\n\n"
            "{examples}\n\n"
            "अब इस पाठ का सारांश लिखें:\n"
            "पाठ: {input}\n\n"
            "सारांश:"
        ),
        few_shot_examples=[
            {
                "input": "भारत के प्रधानमंत्री ने आज नई शिक्षा नीति की घोषणा की। इस नीति के तहत स्कूली शिक्षा में बड़े बदलाव किए जाएंगे। मातृभाषा में शिक्षा पर जोर दिया जाएगा। कोडिंग और डिजिटल साक्षरता को पाठ्यक्रम में शामिल किया जाएगा।",
                "output": "प्रधानमंत्री ने नई शिक्षा नीति की घोषणा की जिसमें मातृभाषा में शिक्षा और डिजिटल साक्षरता पर जोर दिया गया है।"
            },
            {
                "input": "वैज्ञानिकों ने एक नई खोज की है जो कैंसर के इलाज में क्रांतिकारी बदलाव ला सकती है। नई दवा ने प्रयोगशाला में कैंसर कोशिकाओं को 90% तक नष्ट करने में सफलता पाई है। हालांकि, मानव परीक्षण अभी बाकी है।",
                "output": "वैज्ञानिकों ने एक नई कैंसर दवा खोजी है जो प्रयोगशाला में 90% कैंसर कोशिकाओं को नष्ट कर सकती है, लेकिन मानव परीक्षण अभी शेष है।"
            },
            {
                "input": "भारतीय अंतरिक्ष अनुसंधान संगठन ने आज चंद्रयान-4 मिशन की सफल लॉन्चिंग की। यह मिशन चंद्रमा की सतह से नमूने लाने का लक्ष्य रखता है। इसरो के वैज्ञानिकों ने इसे भारत की अंतरिक्ष यात्रा में एक ऐतिहासिक कदम बताया है।",
                "output": "इसरो ने चंद्रयान-4 मिशन सफलतापूर्वक लॉन्च किया जो चंद्रमा से नमूने लाने का ऐतिहासिक मिशन है।"
            },
        ],

        max_input_length=1024,
        max_target_length=256,
        num_train_samples=5000,
        num_eval_samples=200,

        rag_corpus_description="Hindi news articles and Wikipedia passages for contextual summarization",
    ),

    # --------------------------------------------------------
    # TASK 3: Bail Prediction (Hindi Legal)
    # --------------------------------------------------------
    TaskName.BAIL_PREDICTION: TaskConfig(
        name=TaskName.BAIL_PREDICTION,
        display_name="Hindi Legal Bail Prediction",
        description="Predict whether bail will be granted or rejected from Hindi legal documents",
        category="legal",
        task_type="classification",

        # Dataset: HLDC (Hindi Legal Documents Corpus)
        hf_dataset_id="Exploration-Lab/HLDC",
        hf_dataset_split_train="train",
        hf_dataset_split_test="test",
        hf_dataset_split_validation="validation",
        alternative_datasets=[
            "SnehaDeshmukh/IndianBailJudgments-1200",
            "Exploration-Lab/IL-TUR",
        ],

        input_column="text",
        target_column="label",
        label_column="label",

        # Classification
        num_classes=2,
        class_labels=["Bail Rejected (जमानत अस्वीकृत)", "Bail Granted (जमानत स्वीकृत)"],

        # Metrics
        primary_metric="f1_macro",
        all_metrics=["accuracy", "f1_macro", "f1_weighted", "precision", "recall", "auc_roc"],

        # Prompts
        system_prompt=(
            "You are an expert Indian legal AI assistant specializing in Hindi bail applications. "
            "Analyze the given Hindi legal document carefully and predict whether the court will "
            "GRANT or REJECT the bail application. Consider the severity of the crime, "
            "evidence strength, flight risk, and legal precedents. "
            "Output ONLY 'GRANTED' or 'REJECTED'."
        ),
        zero_shot_template=(
            "निम्नलिखित हिंदी कानूनी दस्तावेज़ को पढ़ें और बताएं कि क्या जमानत स्वीकृत होगी या अस्वीकृत।\n\n"
            "दस्तावेज़: {input}\n\n"
            "भविष्यवाणी (GRANTED/REJECTED):"
        ),
        few_shot_template=(
            "निम्नलिखित हिंदी कानूनी दस्तावेज़ों को पढ़ें और जमानत की भविष्यवाणी करें।\n"
            "यहाँ कुछ उदाहरण दिए गए हैं:\n\n"
            "{examples}\n\n"
            "अब इस दस्तावेज़ की भविष्यवाणी करें:\n"
            "दस्तावेज़: {input}\n\n"
            "भविष्यवाणी (GRANTED/REJECTED):"
        ),
        few_shot_examples=[
            {
                "input": "आरोपी पर धारा 420 के तहत धोखाधड़ी का आरोप है। आरोपी का कोई पूर्व आपराधिक रिकॉर्ड नहीं है। पुलिस जांच पूरी हो चुकी है। आरोपी ने अदालत में सहयोग किया है।",
                "output": "GRANTED"
            },
            {
                "input": "आरोपी पर धारा 302 के तहत हत्या का गंभीर आरोप है। सबूत मजबूत हैं और गवाहों को धमकी देने का खतरा है। आरोपी का पूर्व आपराधिक रिकॉर्ड है।",
                "output": "REJECTED"
            },
        ],

        max_input_length=1024,
        max_target_length=32,
        num_train_samples=5000,
        num_eval_samples=200,

        rag_corpus_description="Hindi legal bail judgments, IPC sections, and legal precedents",
    ),

    # --------------------------------------------------------
    # TASK 4: Judge Verdict Prediction (Hindi Legal)
    # --------------------------------------------------------
    TaskName.JUDGE_VERDICT: TaskConfig(
        name=TaskName.JUDGE_VERDICT,
        display_name="Hindi Legal Judge Verdict Prediction",
        description="Predict the judge's decision/verdict from Hindi legal case documents",
        category="legal",
        task_type="classification",

        # Dataset: HLDC + supplements
        hf_dataset_id="Exploration-Lab/HLDC",
        hf_dataset_split_train="train",
        hf_dataset_split_test="test",
        hf_dataset_split_validation="validation",
        alternative_datasets=[
            "SnehaDeshmukh/IndianBailJudgments-1200",
            "Exploration-Lab/IL-TUR",
        ],

        input_column="text",
        target_column="label",
        label_column="label",

        # Classification
        num_classes=3,
        class_labels=[
            "Convicted (दोषी करार)",
            "Acquitted (बरी)",
            "Case Remanded (मामला वापस भेजा)",
        ],

        # Metrics
        primary_metric="f1_macro",
        all_metrics=["accuracy", "f1_macro", "f1_weighted", "precision", "recall", "confusion_matrix"],

        # Prompts
        system_prompt=(
            "You are an expert Indian legal AI assistant specializing in analyzing Hindi court documents. "
            "Read the given case document and predict the judge's verdict. "
            "Consider the evidence, legal arguments, witness testimonies, and applicable laws. "
            "Output ONLY one of: 'CONVICTED', 'ACQUITTED', or 'REMANDED'."
        ),
        zero_shot_template=(
            "निम्नलिखित हिंदी कानूनी मामले को पढ़ें और न्यायाधीश के फैसले की भविष्यवाणी करें।\n\n"
            "मामला: {input}\n\n"
            "फैसला (CONVICTED/ACQUITTED/REMANDED):"
        ),
        few_shot_template=(
            "निम्नलिखित हिंदी कानूनी मामलों को पढ़ें और न्यायाधीश के फैसले की भविष्यवाणी करें।\n"
            "यहाँ कुछ उदाहरण दिए गए हैं:\n\n"
            "{examples}\n\n"
            "अब इस मामले की भविष्यवाणी करें:\n"
            "मामला: {input}\n\n"
            "फैसला (CONVICTED/ACQUITTED/REMANDED):"
        ),
        few_shot_examples=[
            {
                "input": "आरोपी ने चोरी की और पकड़ा गया। CCTV फुटेज में आरोपी स्पष्ट दिख रहा है। गवाहों ने भी आरोपी की पहचान की। आरोपी ने अपराध स्वीकार किया।",
                "output": "CONVICTED"
            },
            {
                "input": "आरोपी पर मारपीट का आरोप था लेकिन कोई प्रत्यक्ष गवाह नहीं था। मेडिकल रिपोर्ट अस्पष्ट है। बचाव पक्ष ने एलिबी प्रस्तुत की जो सत्यापित हो गई।",
                "output": "ACQUITTED"
            },
        ],

        max_input_length=1024,
        max_target_length=32,
        num_train_samples=5000,
        num_eval_samples=200,

        rag_corpus_description="Hindi court judgments, IPC sections, and legal precedent databases",
    ),
}


def get_task_config(task_name: TaskName) -> TaskConfig:
    """Get configuration for a specific task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_name]


def get_generation_tasks() -> List[TaskConfig]:
    """Get all generation task configurations."""
    return [cfg for cfg in TASK_CONFIGS.values() if cfg.task_type == "generation"]


def get_classification_tasks() -> List[TaskConfig]:
    """Get all classification task configurations."""
    return [cfg for cfg in TASK_CONFIGS.values() if cfg.task_type == "classification"]
