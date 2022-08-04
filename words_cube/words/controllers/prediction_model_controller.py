from rest_framework.response import Response

# from words_cube.words.dto.requests.predict_model_request import PredictModelRequest
from words_cube.words.services.prediction_model_services import predict


def predict_model(request):
    # predict_request = PredictModelRequest(request.data)
    predict_response = predict(request)
    return Response(predict_response)