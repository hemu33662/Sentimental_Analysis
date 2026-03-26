# Create your views here.
from django.http import HttpResponse
from django.utils.crypto import constant_time_compare
from django.contrib.auth.hashers import check_password, identify_hasher
import logging
from pathlib import Path
from django.views.decorators.http import require_http_methods


# Create your views here.
from django.shortcuts import render

from .SentimentalAnalysis_models import predict_sentiment

logger = logging.getLogger(__name__)

# Repository root (so we can find account.txt reliably on Render).
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ACCOUNT_FILE_PATH = ROOT_DIR / "account.txt"

def read_file(file_name):
    lines_list = []
    with open(file_name, "r", encoding="utf-8") as opened_file:
        for line in opened_file:
            parts = line.split()
            if len(parts) >= 2:
                lines_list.append(parts[:2])
    return lines_list


# Create your views here.
# Create your views here.
def home(request):
    return render(request, "index.html")


@require_http_methods(["POST"])
def input(request):
    name = request.POST.get("name")
    password = request.POST.get("password")
    if not name or not password:
        return HttpResponse("Missing credentials", content_type="text/plain", status=400)

    # account.txt format (current): "<username> <password_or_hash>"
    # For backwards compatibility we support legacy plaintext entries,
    # but we never log credentials and we compare safely.
    if not ACCOUNT_FILE_PATH.exists():
        logger.warning("account.txt not found at %s", ACCOUNT_FILE_PATH)
        return HttpResponse("Server error", content_type="text/plain", status=500)

    account_list = read_file(ACCOUNT_FILE_PATH)

    for username, stored_password in account_list:
        if constant_time_compare(username, name) is False:
            continue

        # If stored_password looks like a Django hash, verify properly.
        try:
            identify_hasher(stored_password)
            if check_password(password, stored_password):
                return render(request, "input.html")
        except Exception:
            # Legacy plaintext fallback (do NOT log); migrate by replacing with Django hashes.
            if constant_time_compare(stored_password, password):
                return render(request, "input.html")

    return HttpResponse("Wrong Password or Name", content_type="text/plain", status=401)


@require_http_methods(["POST"])
def output(request):
    allowed_algos = {"Custom_LLM", "SVM", "SVM_Pipeline", "NaiveBayes", "MultinomialNB"}
    algo = request.POST.get("algo")
    text = request.POST.get("text", "")

    if algo not in allowed_algos:
        return HttpResponse("Invalid algorithm selected", content_type="text/plain", status=400)

    # Prevent abuse / denial-of-service via huge payloads.
    MAX_TEXT_LEN = 5000
    if not text:
        return HttpResponse("Error: No text provided!", content_type="text/plain", status=400)
    if len(text) > MAX_TEXT_LEN:
        return HttpResponse(
            f"Error: Text too long (max {MAX_TEXT_LEN} chars).",
            content_type="text/plain",
            status=400,
        )

    try:
        sentiment_result = predict_sentiment(text, algo)
    except Exception as e:
        # Last-resort safety net to avoid leaking exception details to clients.
        logger.exception("predict_sentiment failed (algo=%s)", algo)
        return HttpResponse(
            "Server error",
            content_type="text/plain",
            status=500,
        )

    # Avoid logging user-provided text.
    logger.info("Predicted sentiment (algo=%s): %s", algo, sentiment_result)

    return render(request, "output.html", {"out": sentiment_result})
