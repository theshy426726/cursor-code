from django.db import models
from django.conf import settings

class Ship(models.Model):
    STATUS_CHOICES = (
        ('sailing', '航行中'),
        ('docked', '停泊中'),
        ('maintenance', '维修中'),
    )
    
    SHIP_TYPE_CHOICES = (
        ('cargo', '货船'),
        ('passenger', '客船'),
        ('tanker', '油轮'),
        ('fishing', '渔船'),
        ('other', '其他'),
    )
    
    ship_id = models.CharField(max_length=20, unique=True, verbose_name='船舶编号')
    name = models.CharField(max_length=100, verbose_name='船舶名称')
    ship_type = models.CharField(max_length=20, choices=SHIP_TYPE_CHOICES, verbose_name='船舶类型')
    production_year = models.IntegerField(verbose_name='生产年份')
    length = models.FloatField(verbose_name='船长(米)')
    tonnage = models.FloatField(verbose_name='吨位(吨)')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='docked', verbose_name='状态')
    description = models.TextField(blank=True, null=True, verbose_name='描述')
    image = models.ImageField(upload_to='ships/', blank=True, null=True, verbose_name='船舶图片')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '船舶'
        verbose_name_plural = '船舶'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.name} ({self.ship_id})"
