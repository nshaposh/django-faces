from django.shortcuts import render,redirect,render_to_response
from images.models import FeaturedImage
from images.forms import ImageForm
from django.conf import settings
from tf.facesmodel import *
from django.http import HttpResponse

#import os

def home(request):
    image = FeaturedImage.objects.latest('uploaded') 
#    print(settings.MEDIA_URL)
#    res = findfacesonimage(image.img.url))
    return render_to_response('images/home.html',
                              { 'image' : image})


def image_with_faces(request):

    image = FeaturedImage.objects.latest('uploaded') 
#    print(settings.MEDIA_URL)
#    res = findfacesonimage(image.img.url)
#    print(res)
    bok_plot = ''
    try:
        
        bok_plot = img_faces_bok(image.img.url)

#        face = res[1][0]['info']
#        face_rect = face['faceRectangle']
        print(bok_plot)

    except Exception as e:

        bok_plot = str(e);
#        pass


#    return bok_plot
#    return render_to_response('images/image_faces.html',
#                              { 'bok_plot' : bok_plot , 
#                                })
    return HttpResponse(bok_plot)


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
