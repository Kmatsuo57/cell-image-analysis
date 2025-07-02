import requests
import sys

def main():
    url = "https://notify-api.line.me/api/notify"
    token = '3N7g56t8cqBzil7Gl5BlfD38q7HIIBmTXw3Cw9DlEVE'
    headers = {"Authorization" : "Bearer "+ token}

    args = sys.argv
    if args[0] == 1:
        message =  '機械学習終了'
    else :
        message = '異常終了'
    payload = {"message" :  message}
    files = {"imageFile": open("Documents/Data/imaging_data/scripts/end.jpg", "rb")}

    r = requests.post(url ,headers = headers ,params=payload, files=files)

if __name__ == '__main__':
    main()

