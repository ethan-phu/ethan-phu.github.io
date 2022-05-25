# ssh自动断开修复方法


工作中常需要连接着服务器，比如在深度学习训练模型的过程中，需要长时间连接着服务器，在一段时间没有操作后，ssh会自动断开。
为了解决这个问题，在网上找到一种配置方法，亲测很久都不会再断开了，在此记录：

众所周知，ssh是用于与远程服务器建立加密通信通道的，因此配置涉及到服务器和客户端：

- 服务端 /etc/ssh/sshd_config

```shell
+ClientAliveInterval 60 # 每60秒发送一个KeepAlive请求
+ClientAliveCountMax 15 # 总时间为：15*60， 15分钟没有操作，终端断开。
# 以下命令重启 sshd 服务
service sshd reload
service sshd restart
```

- 客户端 ~/.ssh/config

```shell
# 修改～/.ssh/config配置，对当前用户生效
# 这样配置通配所有服务端
Host *
	ServerAliveInterval 60
```

## 参考文献

1. [SSH长时间不使用自动断开解决方案](https://blog.csdn.net/xiaojingfirst/article/details/81744689)

