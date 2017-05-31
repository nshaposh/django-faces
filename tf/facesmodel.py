# Copyright 2017 Nikolai Shaposhnikov. 
# ==============================================================================

"""Functions for proceesing images."""

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
#from images.views import logger

face_dict = {
    0:"Me",
    1:"Malish",
    2:"Matvei",
    3:"Mark",
    4:"Gleb"
    }

def my_family_faces(model_file=''):

    """ This is the actual CNN net model  """

    tf.reset_default_graph()
    imgaug = tflearn.ImageAugmentation()
    imgaug.add_random_rotation(max_angle=7.0)
    imgaug.add_random_flip_leftright()
    imgaug.add_random_blur(sigma_max=5.0)
    network = input_data(shape=[None, 120, 120, 1],data_augmentation=imgaug, name='input')

# Random crop of 24x24 into a 32x32 picture => output 24x24

    network = conv_2d(network, 32, 5, activation='relu', regularizer="L2")
    network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
#    network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
#network = fully_connected(network, 256, activation='tanh')
#network = dropout(network, 0.8)
    network = fully_connected(network, 5, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network)

# Provided a valid saved model it will load it.
    if model_file != '': model.load(model_file)

    return model

 
def get_exif_dict(exif):

    """  Given the image EXIF converts it to a dictionary and return."""

    from PIL.ExifTags import TAGS

    dict = {
            "error":0,
            "message":""
            }

    for (k,v) in exif.items():

        try:
            
            dict[TAGS.get(k)] = v

        except Exception as e:
            
            dict["error"] = 1
            dict["message"] = str(e)
    
    return dict


def findfacesonimage(im,logger):

    from PIL import Image
    import PIL
    import cognitive_face as CF
#    import urllib, cStringIO

    import requests
    from io import BytesIO
    logger.info("Call to MS Cognitive Faces...")


    KEY = 'c1fc932a6bc24ca3bbe97fe9b50aba7c'
    CF.Key.set(KEY)




    basewidth = 2800

    try:
        response = requests.get(im)
        img = Image.open(BytesIO(response.content))
#        file = cStringIO.StringIO(urllib.urlopen(im).read())
#        img = Image.open(file)
        img_orig = im
        exif_dict = get_exif_dict(img._getexif())

    except Exception as e:

        logger.error(str(e))
        return str(e),{}


#    print(exif_dict)

    try:
        width = exif_dict["ExifImageWidth"]
    except:
        width = 4000
    try:
        date = exif_dict["DateTime"]
    except:
        date = "3000:12:00 01:01:01"

    if width > basewidth:
        logger.info("Resizing image {}...".format(im))
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img_res = img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        img_res.save('res.jpg') 
        im = 'res.jpg'
        img = Image.open(im)
    
    try:
        result = CF.face.detect(im,landmarks=True,attributes='age,gender,smile')
    except Exception as e:
#        print("Result:",e.args,e,type(e))
        logger.error(str(e))
        estr = str(e)
#        result = {}
        
        if estr.find("RateLimitExceeded") > 0:

            return "rate_limit_exceeded",[]
#        if e.code == return "rate_limit_exceeded"
#    except:
        logger.error(estr)
        return "error",{}

    images = []
    imw,imh = img.size
    
    logger.info("Loading tensorflow model...")
    try:
        tf_model = my_family_faces(model_file='faces-aug.ftlearn')
        logger.error("TF model loaded.")
        
    except Exception as e:
#        print("Result:",e.args,e,type(e))
        logger.error(str(e))
#        estr = str(e)


    for datum in result:

        logger.info("Cropping face images...")

        image = {"info":datum}
        image["image_width"] = imw
        image["image_height"] = imh
        fr = datum['faceRectangle']
        x = fr['left']
        y = fr['top']
        w = fr['width']
        h = fr['height']
        age = datum['faceAttributes']['age']
        smile = datum['faceAttributes']['smile']
        gender = datum['faceAttributes']['gender']

#        rect = patches.Rectangle((x,y),w,h,linewidth=3,edgecolor='blue',facecolor='none')
#        ax.text(x-1, y+125, age, fontsize=22, color="red")
#        ax.add_patch(rect)
#        print(imgf.size,x,y,w,h)

        region = img.crop((x,y,x+w,y+h))
#        face_img_file = 'faces/{0}_{1}.jpg'.format(img_cnt,face_cnt)
        reg_res = region.resize((120,120), PIL.Image.ANTIALIAS).convert('L')
#        reg_res.convert('L').save(face_img_file,'JPEG')
        image["data"] = reg_res
        pred = np.zeros((1,120,120,1),dtype=int)
        pred[0,:,:,0] = np.array(reg_res)
        face_pred = tf_model.predict(pred)
        print(face_pred)
#        face_img_arr.append(face_img_file)
        image["face_id"] = datum['faceId']
        image["predict"] = face_pred[0]
        image["orig"] = img_orig
        image["date"]  = date
        image["age"] = age
        image["smile"] = smile
        image["gender"] = gender

#        id_arr.append(face_cnt)

        images.append(image)

#        face_cnt += 1
#    img_cnt += 1  


    logger.info("Done findfacesonimage...")
    return "success", images



def img_faces_bok(im,logger):

    from bokeh.models import ColumnDataSource, Range1d, Plot, LinearAxis, Grid
    from bokeh.models.glyphs import Image
    from bokeh.plotting import figure,show
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    from bokeh.models import ColumnDataSource, Range1d, Label


    logger.info("Calling findfacesonimage...")
    try:
        faces = findfacesonimage(im,logger)
        logger.info("Done...")
    except Exception as e:
        logger.info(str(e))
        
#    im_info = faces[1][0]["info"]
    try:
        w = faces[1][0]["image_width"]
        h = faces[1][0]["image_height"]
    except:
        w = 1000
        h = 640

#    print(im_info)
     
    src = ColumnDataSource(dict(url = [im]))

    scale = 1000/w

#    top = int(r["top"]*scale)
#    left = int(r["left"]*scale)
#    fw = int(r["width"]*scale)
#    fh = int(r["height"]*scale)

    logger.info("Bokeh plotting")
    p = figure(plot_width = 1000, plot_height = int(1000*h/w), title="")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.x_range=Range1d(start=0, end=w)
    p.y_range=Range1d(start=0, end=h)
    p.xaxis.visible = None
    p.yaxis.visible = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p.image_url(url='url', x=0, y = h, h=h, w=w, source=src)

    for f in faces[1]:

        face = f["info"] 
        r = face["faceRectangle"]

        fw =    r["width"]
        fh =    r["height"]
        top =   h-r["top"]
        left =  r["left"]

        p.patch([left, left, left+fw, left+fw], [top, top-fh, top-fh,top], alpha=0.5, line_width=2)
        
        person = face_dict[np.argmax(f["predict"])]

        person_label = Label(x=left+fw, y=top, text=person,text_color = 'white',text_font_size='20pt')

        p.add_layout(person_label)

    p.outline_line_alpha = 0 

    
#    plot = figure(x_range=(0,1), y_range=(0,1))
#    image1 = Image(image=im)
#    plot.image_url(im,x=0,y=1,w=1,h=1)
    html = file_html(p, CDN, "my plot")

    logger.info("Faces Bokeh Done!")
#    faces = findfacesonimage(im)
    
    return html
    

    