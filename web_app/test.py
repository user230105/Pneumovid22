#from flask import Flask

#app = Flask(__name__)

#@app.route("/")
def hello():
    return "Hello World!"

img = 'mod-xr/image25_popo12.png'
slash_index = img.find('/')
if slash_index == -1:
    pass
else:
    print(slash_index)
    img_split_name = img[slash_index+1:]
    print(img_split_name)