from django.db import models
from django.conf import settings
from ship_management.models import Ship

class Route(models.Model):
    ALGORITHM_CHOICES = (
        ('astar', 'A*算法'),
        ('ddpg', 'DDPG算法'),
    )
    
    STATUS_CHOICES = (
        ('planning', '规划中'),
        ('completed', '已完成'),
        ('cancelled', '已取消'),
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='routes', verbose_name='用户')
    ship = models.ForeignKey(Ship, on_delete=models.CASCADE, related_name='routes', verbose_name='船舶')
    start_point = models.CharField(max_length=100, verbose_name='起点')
    end_point = models.CharField(max_length=100, verbose_name='终点')
    algorithm = models.CharField(max_length=10, choices=ALGORITHM_CHOICES, default='astar', verbose_name='算法')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='planning', verbose_name='状态')
    distance = models.FloatField(null=True, blank=True, verbose_name='距离(海里)')
    estimated_time = models.FloatField(null=True, blank=True, verbose_name='预计时间(小时)')
    path_data = models.JSONField(null=True, blank=True, verbose_name='路径数据')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '航线'
        verbose_name_plural = '航线'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.ship.name}: {self.start_point} 到 {self.end_point}"
