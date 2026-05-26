from openqa_textual.finetune import (
    DEFAULT_TARGET_MODULES,
    SUPPORTED_QLORA_BASE_MODELS,
    format_sft_example,
    normalize_finetune_config,
    training_arguments_kwargs,
    validate_sft_record,
    validate_sft_records,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return "|".join(f"{message['role']}:{message['content']}" for message in messages)


def test_normalize_finetune_config_applies_defaults_and_overrides() -> None:
    config = normalize_finetune_config(
        {"training": {"num_train_epochs": 2}},
        overrides={
            "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "train_path": "data/processed/train.jsonl",
            "learning_rate": 1e-4,
            "load_in_4bit": False,
        },
    )

    assert config["base_model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config["data"]["train_path"] == "data/processed/train.jsonl"
    assert config["training"]["num_train_epochs"] == 2
    assert config["training"]["learning_rate"] == 1e-4
    assert config["qlora"]["load_in_4bit"] is False
    assert config["lora"]["target_modules"] == list(DEFAULT_TARGET_MODULES)


def test_supported_qlora_models_include_plan_recommendations() -> None:
    assert "Qwen/Qwen2.5-7B-Instruct" in SUPPORTED_QLORA_BASE_MODELS
    assert "Qwen/Qwen2.5-14B-Instruct" in SUPPORTED_QLORA_BASE_MODELS
    assert "mistralai/Mistral-7B-Instruct-v0.3" in SUPPORTED_QLORA_BASE_MODELS


def test_validate_sft_record_accepts_expected_chat_format() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    assert validate_sft_record(record) == []


def test_validate_sft_records_reports_invalid_rows() -> None:
    summary = validate_sft_records([{"messages": []}])

    assert summary["total"] == 1
    assert summary["valid"] == 0
    assert summary["invalid"] == 1


def test_format_sft_example_uses_chat_template_when_available() -> None:
    formatted = format_sft_example(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        FakeTokenizer(),
    )

    assert formatted == "system:sys|user:question|assistant:answer"


def test_training_arguments_kwargs_maps_plan_settings() -> None:
    config = normalize_finetune_config(
        {
            "output_dir": "runs/test",
            "training": {
                "learning_rate": 2e-4,
                "num_train_epochs": 4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
            },
        }
    )
    kwargs = training_arguments_kwargs(config)

    assert kwargs["output_dir"] == "runs/test"
    assert kwargs["learning_rate"] == 2e-4
    assert kwargs["num_train_epochs"] == 4
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["gradient_accumulation_steps"] == 8
