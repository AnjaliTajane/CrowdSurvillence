from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class FireState(models.Model):
    fire_detected = models.BooleanField(default=False)





class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    message = models.CharField(max_length=255)  # Notification message
    created_at = models.DateTimeField(default=timezone.now)  # Timestamp of creation
    
    def __str__(self):
        return f'{self.user.first_name} {self.user.last_name} - {self.message}'
