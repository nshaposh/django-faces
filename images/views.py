from django.shortcuts import render,redirect,render_to_response
from images.models import FeaturedImage
from images.forms import ImageForm
from django.conf import settings
from tf.facesmodel import *
from django.http import HttpResponse
from django.views.generic.list import ListView
from django.contrib.auth.decorators import login_required


import logging
import sys
logname = 'django-faces'

logger = logging.getLogger(logname)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger.setLevel(logging.INFO)

hdlr = logging.FileHandler('/var/tmp/' + logname + '.log')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 

#import os

def home(request):
    logger.info("home")
    image = FeaturedImage.objects.latest('uploaded') 
#    print(settings.MEDIA_URL)
#    res = findfacesonimage(image.img.url))
    return render_to_response('images/home.html',
                              { 'image' : image})


@login_required(login_url='/login/')
def image_with_faces(request):
    logger.info("Faces!!!")
    image = FeaturedImage.objects.latest('uploaded') 
#    print(settings.MEDIA_URL)
#    res = findfacesonimage(image.img.url)
#    print(res)
    bok_plot = ''
    try:
        
        logger.info("Calling img_faces_bok")
        bok_plot = img_faces_bok(image.img.url,logger)

#        face = res[1][0]['info']
#        face_rect = face['faceRectangle']
        print(bok_plot)
        logger.info("Bokeh plot successfully generated.")

    except Exception as e:

        bok_plot = "<html><body>{}</body></html>".format(str(e));
        logger.error(str(e))
#        pass


#    return bok_plot
#    return render_to_response('images/image_faces.html',
#                              { 'bok_plot' : bok_plot , 
#                                })
    return HttpResponse(bok_plot)

@login_required(login_url='/login/')
def image_upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
#        print(request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = ImageForm()
    return render(request, 'images/image_upload.html', {
        'form': form
    })


#@login_required(login_url='/login/')
class ImageListView(ListView):

    model = FeaturedImage

    def get_context_data(self, **kwargs):
        context = super(ImageListView, self).get_context_data(**kwargs)
#        context['now'] = timezone.now()
        return context

