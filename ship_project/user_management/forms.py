from django import forms
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm as DjangoPasswordChangeForm
from .models import User

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, label='邮箱', widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '请输入邮箱'}))
    username = forms.CharField(label='用户名', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'}))
    password1 = forms.CharField(label='密码', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入密码'}))
    password2 = forms.CharField(label='确认密码', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请确认密码'}))
    
    class Meta:
        model = User
        fields = ('email', 'username', 'password1', 'password2')
        
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('该邮箱已被注册')
        return email

class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField(required=True, label='邮箱', widget=forms.EmailInput(attrs={'class': 'form-control'}))
    username = forms.CharField(label='用户名', widget=forms.TextInput(attrs={'class': 'form-control'}))
    
    class Meta:
        model = User
        fields = ('email', 'username')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'].disabled = True  # 邮箱不允许修改

class PasswordChangeForm(DjangoPasswordChangeForm):
    old_password = forms.CharField(label='旧密码', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入旧密码'}))
    new_password1 = forms.CharField(label='新密码', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入新密码'}))
    new_password2 = forms.CharField(label='确认新密码', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请确认新密码'}))
    
    class Meta:
        model = User