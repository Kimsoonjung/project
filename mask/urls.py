# # -*- coding: utf-8-*-

from django.urls import path
from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns
from .views import *
from .views import kkk

urlpatterns = [
    path('mask/', MaskingList.as_view()),
    path('mask/<int:pk>/', MaskingDetail.as_view()),
]
if kkk == 1:
    urlpatterns = [
        path('mask/', MaskingList.as_view()),
        path('mask1/', GIList1.as_view()),
    ]
if kkk == 2:
    urlpatterns = [
        path('mask/', MaskingList.as_view()),
        path('mask1/', GIList1.as_view()),
        path('mask2/', GIList2.as_view()),
    ]

if kkk == 3:
    urlpatterns = [
        path('mask/', MaskingList.as_view()),
        path('mask1/', GIList1.as_view()),
        path('mask2/', GIList2.as_view()),
        path('mask3/', GIList3.as_view()),
    ]
if kkk == 4:
    urlpatterns = [
        path('mask/', MaskingList.as_view()),
        path('mask1/', GIList1.as_view()),
        path('mask2/', GIList2.as_view()),
        path('mask3/', GIList3.as_view()),
        path('mask4/', GIList4.as_view()),
    ]