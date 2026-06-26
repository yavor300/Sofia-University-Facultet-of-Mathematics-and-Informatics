# GutBrainIE Streamlit Demo

Run from the project root:

```bash
make app
```

Equivalent command:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

The app loads articles and annotations from:

```text
data/gutbrainie2026/
```

It autodiscovers prediction JSON files under:

```text
outputs/predictions/
```

and metrics reports under:

```text
outputs/reports/
```

The demo supports:

- dataset split exploration;
- T611 entity highlighting and gold/prediction comparison;
- T621 relation cards and tables;
- metrics dashboards from JSON reports;
- error-analysis views;
- a custom-text dictionary NER demo.
