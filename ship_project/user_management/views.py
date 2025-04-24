from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.core.paginator import Paginator
from .models import User
from .forms import UserRegistrationForm, UserUpdateForm, PasswordChangeForm

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "注册成功！欢迎使用船舶管理系统。")
            return redirect('ship_list')
    else:
        form = UserRegistrationForm()
    return render(request, 'user_management/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        # 修改表单标签为中文
        form.fields['username'].label = '邮箱'
        form.fields['password'].label = '密码'
        if form.is_valid():
            email = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=email, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"欢迎回来，{user.username}！")
                return redirect('ship_list')
        else:
            messages.error(request, "登录失败，请检查邮箱和密码。")
    else:
        form = AuthenticationForm()
        # 修改表单标签为中文
        form.fields['username'].label = '邮箱'
        form.fields['password'].label = '密码'
    return render(request, 'user_management/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "您已成功退出登录。")
    return redirect('login')

@login_required
def profile_view(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "个人信息更新成功！")
            return redirect('profile')
    else:
        form = UserUpdateForm(instance=request.user)
    
    return render(request, 'user_management/profile.html', {'form': form})

@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "密码修改成功！请重新登录。")
            return redirect('login')
    else:
        form = PasswordChangeForm(request.user)
    
    return render(request, 'user_management/change_password.html', {'form': form})

# 管理员视图
@login_required
def user_list_view(request):
    if not request.user.is_admin:
        messages.error(request, "您没有权限访问此页面。")
        return redirect('ship_list')
    
    users = User.objects.all().order_by('-date_joined')
    
    # 分页
    paginator = Paginator(users, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'user_management/user_list.html', {'page_obj': page_obj})

@login_required
def user_detail_view(request, pk):
    if not request.user.is_admin:
        messages.error(request, "您没有权限访问此页面。")
        return redirect('ship_list')
    
    user = get_object_or_404(User, pk=pk)
    return render(request, 'user_management/user_detail.html', {'user': user})

@login_required
def user_create_view(request):
    if not request.user.is_admin:
        messages.error(request, "您没有权限访问此页面。")
        return redirect('ship_list')
    
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "用户创建成功！")
            return redirect('user_list')
    else:
        form = UserRegistrationForm()
    
    # 这里不传递edit_user或传递为None
    return render(request, 'user_management/user_form.html', {'form': form})

@login_required
def user_update_view(request, pk):
    if not request.user.is_admin:
        messages.error(request, "您没有权限访问此页面。")
        return redirect('ship_list')
    
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "用户信息更新成功！")
            return redirect('user_detail', pk=user.pk)
    else:
        form = UserUpdateForm(instance=user)
    
    # 使用edit_user变量而不是user
    return render(request, 'user_management/user_form.html', {'form': form, 'edit_user': user})

@login_required
def user_delete_view(request, pk):
    if not request.user.is_admin:
        messages.error(request, "您没有权限访问此页面。")
        return redirect('ship_list')
    
    user = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        user.delete()
        messages.success(request, "用户删除成功！")
        return redirect('user_list')
    
    return render(request, 'user_management/user_confirm_delete.html', {'user': user})
