import os
from flask import Flask, request, render_template
from search_api import search, more_like_this

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

@app.get("/")
def home():
    q = request.args.get("q", "").strip()
    lang = request.args.get("lang", "ALL").upper()
    fuzzy = request.args.get("fuzzy") == "1"
    exact = request.args.get("exact") == "1"

    results = []
    if q:
        results = search(q=q, lang=lang, size=10, fuzzy=fuzzy, exact=exact)

    return render_template(
        "index.html",
        q=q,
        lang=lang,
        fuzzy=fuzzy,
        exact=exact,
        results=results
    )

@app.get("/similar")
def similar():
    doc_id = request.args.get("id", "").strip()
    lang = request.args.get("lang", "ALL").upper()
    fuzzy = request.args.get("fuzzy") == "1"
    exact = request.args.get("exact") == "1"

    results = []
    if doc_id:
        results = more_like_this(doc_id=doc_id, size=10)

    return render_template(
        "index.html",
        q="",
        lang=lang,
        fuzzy=fuzzy,
        exact=exact,
        results=results
    )

if __name__ == "__main__":
    app.run(port=5000, debug=True)
