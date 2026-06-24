from gutbrainie.evaluation.official_eval_wrapper import run_official_evaluation


def test_run_official_evaluation_with_explicit_entrypoint(tmp_path):
    official_repo = tmp_path / "official"
    prediction = tmp_path / "prediction.json"
    script = official_repo / "evaluation" / "evaluate.py"
    script.parent.mkdir(parents=True)
    prediction.write_text("{}", encoding="utf-8")
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--prediction')\n"
        "args = parser.parse_args()\n"
        "print(f'evaluated {args.prediction}')\n",
        encoding="utf-8",
    )

    result = run_official_evaluation(
        official_repo=official_repo,
        prediction=prediction,
        entrypoint="evaluation/evaluate.py",
    )

    assert result["returncode"] == 0
    assert "evaluated" in result["stdout"]
