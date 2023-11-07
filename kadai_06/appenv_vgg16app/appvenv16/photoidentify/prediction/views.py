from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.models import save_model
from io import BytesIO
import os
import numpy as np


def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)               
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
                   

            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            result = decode_predictions(result,top=5)[0]

            prediction = result[0][1]
            prediction2 = '{:.10f}'.format(result[0][2]) + '%'

            prediction = result[0][1]
            prediction2 = '{:.10f}'.format(result[0][2]) + '%'

            prediction3 = result[1][1]
            prediction4 = '{:.10f}'.format(result[1][2]) + '%'

            prediction5 = result[2][1]
            prediction6 = '{:.10f}'.format(result[2][2]) + '%'

            prediction7 = result[3][1]
            prediction8 = '{:.10f}'.format(result[3][2]) + '%'

            prediction9 = result[4][1]
            prediction10 = '{:.10f}'.format(result[4][2]) + '%'            

            img_data = request.POST.get('img_data')  
            
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 
                                                                'prediction2': prediction2,
                                                                'prediction3': prediction3,
                                                                'prediction4': prediction4,
                                                                'prediction5': prediction5,
                                                                'prediction6': prediction6,
                                                                'prediction7': prediction7,
                                                                'prediction8': prediction8,
                                                                'prediction9': prediction9,
                                                                'prediction10': prediction10,
                                                                'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})        

    # else:
