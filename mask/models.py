from django.db import models
from django.conf import settings

class Masking(models.Model):
	name = models.CharField(max_length=20)
	GI= models.IntegerField(null=True)
	GI_A =models.CharField(max_length=10, default = '')

	class Meta:
		ordering = ['-GI']

# Create your models here.
