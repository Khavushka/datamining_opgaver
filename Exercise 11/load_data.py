import urllib.request

url = "https://cs.joensuu.fi/sipu/datasets/Aggregation.txt"
filename = "data.txt"

urllib.request.urlretrieve(url, filename)