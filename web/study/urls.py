from django.urls import path

from . import views

urlpatterns = [
    path("start/", views.start, name="start"),
    path("play/", views.play, name="play"),
    path("play/round/", views.play_round, name="play_round"),
    path("play/feedback/", views.play_feedback, name="play_feedback"),
]
