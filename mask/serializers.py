
# food_api/foods/serializers.py

from .models import Masking
from rest_framework import serializers


class MaskingSerializer(serializers.ModelSerializer):
	class Meta:
		model =Masking
		fields ='__all__'

