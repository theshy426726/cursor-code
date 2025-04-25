from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse
from .models import Route
from ship_management.models import Ship
from .forms import RouteForm
from .route_algorithms import astar_algorithm, ddpg_algorithm

@login_required
def route_list_view(request):
    # 管理员可以看到所有路径，普通用户只能看到自己的路径
    if request.user.is_admin:
        routes = Route.objects.select_related('ship', 'user').all()
    else:
        routes = Route.objects.select_related('ship', 'user').filter(user=request.user)
    
    # 搜索功能
    search_query = request.GET.get('search', '')
    if search_query:
        routes = routes.filter(
            ship__name__icontains=search_query
        ) | routes.filter(
            start_point__icontains=search_query
        ) | routes.filter(
            end_point__icontains=search_query
        )
    
    # 分页
    paginator = Paginator(routes, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'route_planning/route_list.html', {
        'page_obj': page_obj,
        'search_query': search_query
    })

@login_required
def route_detail_view(request, pk):
    route = get_object_or_404(Route, pk=pk)
    
    # 确保普通用户只能查看自己的路径规划
    if not request.user.is_admin and route.user != request.user:
        messages.error(request, "您没有权限查看此路径规划。")
        return redirect('route_list')
    
    return render(request, 'route_planning/route_detail.html', {'route': route})

@login_required
def route_create_view(request):
    if request.method == 'POST':
        form = RouteForm(request.POST)
        if form.is_valid():
            route = form.save(commit=False)
            route.user = request.user
            route.save()
            
            # 重定向到规划路径页面
            return redirect('route_plan', pk=route.pk)
    else:
        # 只选择状态为"航行中"或"停泊中"的船
        available_ships = Ship.objects.filter(status__in=['sailing', 'docked'])
        
        # 普通用户无法选择其他用户的路径
        initial = {}
        form = RouteForm(initial=initial)
        form.fields['ship'].queryset = available_ships
    
    return render(request, 'route_planning/route_form.html', {'form': form})

@login_required
def route_plan_view(request, pk):
    route = get_object_or_404(Route, pk=pk)
    
    # 确保普通用户只能规划自己的路径
    if not request.user.is_admin and route.user != request.user:
        messages.error(request, "您没有权限规划此路径。")
        return redirect('route_list')
    
    if request.method == 'POST':
        # 执行路径规划算法
        algorithm = request.POST.get('algorithm')
        
        if algorithm == 'astar':
            # 使用A*算法进行路径规划
            path_data, distance, estimated_time = astar_algorithm(
                route.start_point, route.end_point
            )
            route.algorithm = 'astar'
        else:
            # 使用DDPG算法进行路径规划
            path_data, distance, estimated_time = ddpg_algorithm(
                route.start_point, route.end_point
            )
            route.algorithm = 'ddpg'
        
        # 更新路径信息
        route.path_data = path_data
        route.distance = distance
        route.estimated_time = estimated_time
        route.status = 'completed'
        route.save()
        
        messages.success(request, "路径规划完成！")
        return redirect('route_detail', pk=route.pk)
    
    return render(request, 'route_planning/route_plan.html', {'route': route})

@login_required
def route_delete_view(request, pk):
    route = get_object_or_404(Route, pk=pk)
    
    # 确保只有管理员和创建者可以删除路径
    if not request.user.is_admin and route.user != request.user:
        messages.error(request, "您没有权限删除此路径规划。")
        return redirect('route_list')
    
    if request.method == 'POST':
        route.delete()
        messages.success(request, "路径规划删除成功！")
        return redirect('route_list')
    
    return render(request, 'route_planning/route_confirm_delete.html', {'route': route})
