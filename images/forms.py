from django import forms
from images.models import FeaturedImage

class ImageForm(forms.ModelForm):
    class Meta:
        model = FeaturedImage
        fields = ('name', 'tagline','img' )
