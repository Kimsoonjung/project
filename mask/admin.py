from django.contrib import admin
from .models import Masking

class MaskingAdmin(admin.ModelAdmin):
	list_display = ['name', 'GI','GI_A']

admin.site.register(Masking, MaskingAdmin)

# Register your models here.
