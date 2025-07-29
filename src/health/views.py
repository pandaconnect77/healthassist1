from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate
from .models import UserProfile, HealthRecord, SkinDisease
from .serializers import userProfileSerializer, HealthRecordSerializer, SkinDiseaseSerializer
from .utils import predict_image
import requests
import os
import logging
from dotenv import load_dotenv
from django.http import JsonResponse


load_dotenv()
logger = logging.getLogger(__name__)

class UserRegister(APIView):
    def post(self, request):
        try:
            data = request.data.copy()
            data['password'] = make_password(data.get('password'))
            serializer = userProfileSerializer(data=data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("Registration failed.")
            return Response({"error": "Server error during registration."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def server_status(request):
    return JsonResponse({"status": "Server started", "host": "https://healthassist-38x5.onrender.com"})
    
class UserLogin(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        try:
            user = UserProfile.objects.get(username=username)
            if user.check_password(password):
                request.session['user_id'] = user.id
                serializer = userProfileSerializer(user)
                return Response({'user': serializer.data}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Invalid password"}, status=status.HTTP_401_UNAUTHORIZED)
        except UserProfile.DoesNotExist:
            return Response({"error": "Invalid username"}, status=status.HTTP_401_UNAUTHORIZED)

class UserInfo(APIView):
    def get(self, request):
        user_id = request.session.get('user_id')
        if not user_id:
            return Response({'error': 'User not logged in'}, status=status.HTTP_401_UNAUTHORIZED)
        try:
            user = UserProfile.objects.get(id=user_id)
            serializer = userProfileSerializer(user)
            return Response({'user': serializer.data}, status=status.HTTP_200_OK)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = userProfileSerializer

class HealthRecordView(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        if not user_id:
            return Response({"error": "Missing user ID"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = UserProfile.objects.get(id=user_id)
        except UserProfile.DoesNotExist:
            return Response({"error": "Invalid user ID"}, status=status.HTTP_404_NOT_FOUND)

        serializer = HealthRecordSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_message = serializer.validated_data['message']
        city_name = user.address or "Ongole"

        groq_api_key = os.getenv("GROQ_API_KEY")
        geoapify_api_key = os.getenv("GEOAPIFY_API_KEY")

        # GROQ AI Request
        try:
            groq_payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI health assistant."},
                    {"role": "user", "content": f"""
                        Analyze the user's symptoms and provide:
                        1. Summary
                        2. OTC Medications
                        3. Natural Remedies
                        4. Medical Advice

                        Symptoms: \"{user_message}\" 
                    """}
                ]
            }
            groq_res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=groq_payload,
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
            )
            groq_res.raise_for_status()
            bot_reply = groq_res.json()['choices'][0]['message']['content'].strip()
        except requests.RequestException as e:
            return Response({"error": f"GROQ API failed: {str(e)}"}, status=500)

        # Geoapify: Get nearby hospitals
        hospitals = []
        try:
            geo_res = requests.get(
                "https://api.geoapify.com/v1/geocode/search",
                params={"text": city_name, "apiKey": geoapify_api_key}
            )
            geo_res.raise_for_status()
            features = geo_res.json().get("features", [])

            if features:
                coords = features[0]["geometry"]["coordinates"]
                lat, lon = coords[1], coords[0]

                places_res = requests.get(
                    "https://api.geoapify.com/v2/places",
                    params={
                        "categories": "healthcare.hospital",
                        "bias": f"proximity:{lon},{lat}",
                        "limit": 5,
                        "apiKey": geoapify_api_key
                    }
                )
                places_res.raise_for_status()
                place_data = places_res.json().get("features", [])

                hospitals = [
                    {
                        "name": p["properties"].get("name", "Unnamed"),
                        "address": p["properties"].get("formatted", "No address"),
                        "category": p["properties"].get("sub_category", "N/A"),
                        "lat": p["properties"].get("lat"),
                        "lon": p["properties"].get("lon"),
                        "map_link": f"https://www.google.com/maps/search/?api=1&query={p['properties'].get('lat')},{p['properties'].get('lon')}"
                    }
                    for p in place_data
                ]
            else:
                hospitals = [{"error": "City not found"}]
        except requests.RequestException as e:
            hospitals = [{"error": f"Geoapify error: {str(e)}"}]

        record = HealthRecord.objects.create(
            user=user,
            message=user_message,
            bot_response=bot_reply
        )

        return Response({
            "record": HealthRecordSerializer(record).data,
            "suggested_hospitals": hospitals
        }, status=status.HTTP_201_CREATED)

class SkinDiseaseView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = SkinDiseaseSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()

            try:
                image_path = instance.image.path
                prediction = predict_image(image_path)
                instance.message = f"Predicted class: {prediction}"
                instance.save()
            except Exception as e:
                logger.exception("Prediction error")
                return Response({"error": "Prediction failed"}, status=500)

            return Response({
                "prediction": prediction,
                "image_url": instance.image.url,
                "message": instance.message
            }, status=200)

        return Response(serializer.errors, status=400)
