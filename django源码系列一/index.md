# Django源码系列：文件变化后server自动重启机制


## 初试 - 文件变化后 `server` 自动重启

本源码系列是基于 Django4.0 的源码，可以自行到[django 官方](https://github.com/django/django.git)下载。

> 在此之前，不妨先了解下 `django` 是如何做到自动重启的

### 开始

`django` 使用 `runserver` 命令的时候，会启动俩个进程。

`runserver` 主要调用了 `django/utils/autoreload.py` 下 `main` 方法。  
_至于为何到这里的，我们这里不作详细的赘述，后面篇章会进行说明。_

主线程通过 `os.stat` 方法获取文件最后的修改时间进行比较，继而重新启动 `django` 服务（也就是子进程）。

大概每秒监控一次。

```python
# django/utils/autoreload.py 的 reloader_thread 方法

def reloader_thread():
    ...
    # 监听文件变化
    # -- Start
    # 这里主要使用了 `pyinotify` 模块，因为目前可能暂时导入不成功，使用 else 块代码
    # USE_INOTIFY 该值为 False
    if USE_INOTIFY:
        fn = inotify_code_changed
    else:
        fn = code_changed
    # -- End
    while RUN_RELOADER:
        change = fn()
        if change == FILE_MODIFIED:
            sys.exit(3)  # force reload
        elif change == I18N_MODIFIED:
            reset_translations()
        time.sleep(1)
```

`code_changed` 根据每个文件的最后修改时间是否发生变更，则返回 `True` 达到重启的目的。

### 父子进程&多线程

关于重启的代码在 `python_reloader` 函数内

```python

# django/utils/autoreload.py

def restart_with_reloader():
    # 在这里开始设置环境变量为true
    new_environ = {**os.environ, DJANGO_AUTORELOAD_ENV: "true"}
    args = get_child_arguments() #获取执行的命令参数
    # 重启命令在这里开始生效
    while True:
        p = subprocess.run(args, env=new_environ, close_fds=False)
        if p.returncode != 3:
            return p.returncode


def run_with_reloader(main_func, *args, **kwargs):
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    # 刚开始 DJANGO_AUTORELOAD_ENV是没有被设置为true的所以这里会进入到else里。
    try:
        if os.environ.get(DJANGO_AUTORELOAD_ENV) == "true":
            reloader = get_reloader()
            logger.info(
                "Watching for file changes with %s", reloader.__class__.__name__
            )
            start_django(reloader, main_func, *args, **kwargs) # 开启django服务线程
        else:
            exit_code = restart_with_reloader()
            sys.exit(exit_code) # 0为正常退出，其他的会抛出相关的错误
    except KeyboardInterrupt:
        pass
```

程序启动，因为没有 `RUN_MAIN` 变量，所以走的 else 语句块。

颇为有趣的是，`restart_with_reloader` 函数中使用 `subprocess.run` 方法执行了启动程序的命令（ e.g：python3 manage.py runserver ），此刻 `RUN_MAIN` 的值为 `True` ，接着执行 `_thread.start_new_thread(main_func, args, kwargs)` 开启新线程，意味着启动了 `django` 服务。

如果子进程不退出，则停留在 `run` 方法这里（进行请求处理），如果子进程退出，退出码不是 3，while 则被终结。反之就继续循环，重新创建子进程。

### 总结

以上就是 `django` 检测文件修改而达到重启服务的实现流程。

结合 `subprocess.run` 和 环境变量 创建俩个进程。主进程负责监控子进程和重启子进程。
子进程下通过开启一个新线程（也就是 `django` 服务）。主线程监控文件变化，如果变化则通过 `sys.exit(3)` 来退出子进程，父进程获取到退出码不是 3 则继续循环创建子进程，反之则退出整个程序。

好，到这里。我们勇敢的迈出了第一步，我们继续下一个环节！！！ ヾ(◍°∇°◍)ﾉﾞ

