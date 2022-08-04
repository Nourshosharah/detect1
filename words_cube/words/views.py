from rest_framework.decorators import api_view

from words_cube.words.controllers.prediction_model_controller import predict_model


@api_view(['GET','POST'])
def predicting_model(request):
    return predict_model(request)
