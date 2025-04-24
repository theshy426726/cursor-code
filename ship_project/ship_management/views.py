from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from .models import Ship
from .forms import ShipForm

@login_required
def ship_list_view(request):
    ships = Ship.objects.all()
    
    # 搜索功能
    search_query = request.GET.get('search', '')
    if search_query:
        ships = ships.filter(name__icontains=search_query) | ships.filter(ship_id__icontains=search_query)
    
    # 过滤功能
    status_filter = request.GET.get('status', '')
    
    if status_filter and status_filter in dict(Ship.STATUS_CHOICES):
        ships = ships.filter(status=status_filter)
        
    ship_type_filter = request.GET.get('ship_type', '')
    if ship_type_filter and ship_type_filter in dict(Ship.SHIP_TYPE_CHOICES):
        ships = ships.filter(ship_type=ship_type_filter)
    
    # 分页
    paginator = Paginator(ships, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'status_filter': status_filter,
        'ship_type_filter': ship_type_filter,
    }
    
    return render(request, 'ship_management/ship_list.html', context)

@login_required
def ship_detail_view(request, pk):
    ship = get_object_or_404(Ship, pk=pk)
    return render(request, 'ship_management/ship_detail.html', {'ship': ship})

@login_required
def ship_create_view(request):
    if not request.user.is_admin:
        messages.error(request, "您没有权限执行此操作。")
        return redirect('ship_list')
    
    if request.method == 'POST':
        form = ShipForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "船舶创建成功！")
            return redirect('ship_list')
    else:
        form = ShipForm()
    
    return render(request, 'ship_management/ship_form.html', {'form': form})

@login_required
def ship_update_view(request, pk):
    if not request.user.is_admin:
        messages.error(request, "您没有权限执行此操作。")
        return redirect('ship_list')
    
    ship = get_object_or_404(Ship, pk=pk)
    
    if request.method == 'POST':
        form = ShipForm(request.POST, request.FILES, instance=ship)
        if form.is_valid():
            form.save()
            messages.success(request, "船舶信息更新成功！")
            return redirect('ship_detail', pk=ship.pk)
    else:
        form = ShipForm(instance=ship)
    
    return render(request, 'ship_management/ship_form.html', {'form': form, 'ship': ship})

@login_required
def ship_delete_view(request, pk):
    if not request.user.is_admin:
        messages.error(request, "您没有权限执行此操作。")
        return redirect('ship_list')
    
    ship = get_object_or_404(Ship, pk=pk)
    
    if request.method == 'POST':
        ship.delete()
        messages.success(request, "船舶删除成功！")
        return redirect('ship_list')
    
    return render(request, 'ship_management/ship_confirm_delete.html', {'ship': ship})
