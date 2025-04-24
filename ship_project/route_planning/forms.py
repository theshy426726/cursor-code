from django import forms
from .models import Route

class RouteForm(forms.ModelForm):
    class Meta:
        model = Route
        fields = ['ship', 'start_point', 'end_point']
        widgets = {
            'ship': forms.Select(attrs={'class': 'form-control'}),
            'start_point': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入起点位置'}),
            'end_point': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入终点位置'}),
        } 