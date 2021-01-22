import requests
import json
import base64

""" 你的 APPID AK SK """
APP_ID ="你的APPID"
API_KEY = "你的AK"
SECRET_KEY = "你的SK"

def get_access_token():
    url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'\
        .format(API_KEY, SECRET_KEY)

    response = requests.get(url)

    if response:
        access_token = response.json()["access_token"]
        return access_token

def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as f:
            image_data = f.read()
    else:
        image_data = image

    # print(base64.b64encode(image_data).decode("utf-8"))
    return base64.b64encode(image_data).decode("utf-8")

def similarity(img1, img2):
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"

    params = {
        "access_token": get_access_token()
    }

    header = {
        "Content-Type": "application/json"
    }

    request_data = [
        {
            "image": encode_image(img1),
            "image_type": "BASE64",
            "face_type": "LIVE",
            "quality_control": "LOW",
        },
        {
            "image": encode_image(img2),
            "image_type": "BASE64",
            "face_type": "IDCARD",
            "quality_control": "LOW",
        }
    ]

    data = json.dumps(request_data)

    response = requests.post(request_url, params=params, data=data, headers=header)

    # print(response.text)
    if response and response.json()["error_msg"] == "SUCCESS":
        return response.json()["result"]["score"]
    else:
        return 0


if __name__ == '__main__':
    score = similarity("lena.jpg", "noise.jpg")
    print(score)







