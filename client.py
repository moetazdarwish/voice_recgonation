import requests


# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
#FILE_PATH = "../test/down.wav"


def main(FILE_PATH):

    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    print(response)
    data = response.json()
    print(data)

    print(f"Predicted keyword: {data['keyword']}")
   # print("Predicted keyword: {}".format(data["keyword"]))


if __name__ == "__main__":
    main()