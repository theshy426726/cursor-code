# Generated by Django 5.1.7 on 2025-03-21 08:40

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("ship_management", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Route",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("start_point", models.CharField(max_length=100, verbose_name="起点")),
                ("end_point", models.CharField(max_length=100, verbose_name="终点")),
                (
                    "algorithm",
                    models.CharField(
                        choices=[("astar", "A*算法"), ("ddpg", "DDPG算法")],
                        default="astar",
                        max_length=10,
                        verbose_name="算法",
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("planning", "规划中"),
                            ("completed", "已完成"),
                            ("cancelled", "已取消"),
                        ],
                        default="planning",
                        max_length=10,
                        verbose_name="状态",
                    ),
                ),
                (
                    "distance",
                    models.FloatField(blank=True, null=True, verbose_name="距离(海里)"),
                ),
                (
                    "estimated_time",
                    models.FloatField(
                        blank=True, null=True, verbose_name="预计时间(小时)"
                    ),
                ),
                (
                    "path_data",
                    models.JSONField(blank=True, null=True, verbose_name="路径数据"),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, verbose_name="创建时间"),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, verbose_name="更新时间"),
                ),
                (
                    "ship",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="routes",
                        to="ship_management.ship",
                        verbose_name="船舶",
                    ),
                ),
            ],
            options={
                "verbose_name": "航线",
                "verbose_name_plural": "航线",
                "ordering": ["-created_at"],
            },
        ),
    ]
