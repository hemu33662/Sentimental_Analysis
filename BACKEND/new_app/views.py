# Create your views here.
from django.http import HttpResponse

# Create your views here.
from django.shortcuts import render

from .SentimentalAnalysis_models import predict_sentiment

print("########### views ##########")


def read_file(file_name):
    opened_file = open(file_name, "r")
    lines_list = []
    for line in opened_file:
        line = line.split()
        lines_list.append(line)
    # print(lines_list)
    return lines_list


# Create your views here.
# Create your views here.
def home(request):
    return render(request, "index.html")


def input(request):
    file_name = "account.txt"
    name = request.POST.get("name")
    password = request.POST.get("password")
    account_list = read_file(file_name)
    print(name)
    print(password)
    for i in account_list:

        if i[0] == name and i[1] == password:
            print(i[0])
            print(i[1])
            return render(request, "input.html")
        else:
            return HttpResponse(
                "Wrong Password or Name", content_type="text/plain"
            )


def output(request):
    algo = request.POST.get("algo")
    text = request.POST.get("text", "")

    if not text:
        return HttpResponse(
            "Error: No text provided!", content_type="text/plain"
        )

    sentiment_result = predict_sentiment(text, algo)

    print(f"🔹 Input Text: {text}")
    print(f"🔹 Predicted Sentiment: {sentiment_result}")

    return render(request, "output.html", {"out": sentiment_result})
