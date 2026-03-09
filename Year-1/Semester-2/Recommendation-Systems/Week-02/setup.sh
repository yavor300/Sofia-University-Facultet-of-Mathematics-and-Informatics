python3 -m venv .venv

source .venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel pandas matplotlib scikit-learn nltk scipy

# register a notebook kernel backed by this .venv
python -m ipykernel install --user --name recsys-week02-venv --display-name "RecSys Week-02 (.venv)"

python -m nltk.downloader stopwords
