# Generated by Django 5.1.4 on 2025-01-06 09:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_rename_created_at_notification_timestamp'),
    ]

    operations = [
        migrations.RenameField(
            model_name='notification',
            old_name='timestamp',
            new_name='created_at',
        ),
    ]
