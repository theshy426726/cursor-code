from django import forms
from .models import Ship

class ShipForm(forms.ModelForm):
    class Meta:
        model = Ship
        fields = ['ship_id', 'name', 'ship_type', 'production_year', 'length', 'tonnage', 'status', 'description', 'image']
        widgets = {
            'ship_id': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入船舶编号'}),
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入船舶名称'}),
            'ship_type': forms.Select(attrs={'class': 'form-control'}),
            'production_year': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '请输入生产年份'}),
            'length': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '请输入船长(米)'}),
            'tonnage': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '请输入吨位(吨)'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': '请输入船舶描述(可选)'}),
        }
        
    def clean_ship_id(self):
        ship_id = self.cleaned_data.get('ship_id')
        if Ship.objects.filter(ship_id=ship_id).exists() and not self.instance.pk:
            raise forms.ValidationError('该船舶编号已存在')
        return ship_id
        
    def clean_production_year(self):
        year = self.cleaned_data.get('production_year')
        if year < 1900 or year > 2100:
            raise forms.ValidationError('请输入有效的生产年份(1900-2100)')
        return year 