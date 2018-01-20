## 目录结构
```
├─benchmarks
│  └─imp
├─imp_extract
│  └── 11510xxx
		├─ISE.py
│       └─IMP.py
└─imp_result
	├─ISE_result
    └─IMP_result
```
## 如何测试

建议在Linux环境下测试

1. 在 `imp_extract` 目录下创建以**纯学号**命名的文件夹，放入程序代码
2. 在 `IMP_test` 文件夹下打开终端，运行命令 `python ISE_test.py`或'python IMP_test.py'
3. 等待，可另开终端输入`top`或者`ps -ef|grep python`，看python进程是否运行完成
4. 查看`IMP_result`和'ISE_result'中是否有运行结果，并查看imp_result中的两个xls文件中的结果

## 测试结果解释


## 可能导致错误的原因
1. 参数长度出错，运行命令含有重定向，可能造成误判定。 如果将超长参数判定为异常，则没有输出。
2.IMP的输出格式为每行一个节点编号，请按照格式输出，不要自定义格式。自定义的格式认为格式错误，不参与排名。

