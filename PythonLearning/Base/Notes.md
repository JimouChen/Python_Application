##**PythonLearning Notes**  
####正则表达式  
python正则表达式中，如果有个匹配模式想要一直使用，  
或者在一个循环里面查找多次，那可以使用：
>>**re.compile()**  
```
import re

p = re.compile(r'[A-Z]')
print(p.search('Hello World'))
print(p.findall('Hello World'))

output:
<re.Match object; span=(0, 1), match='H'>
['H', 'W']

```
不用compile的情况
``` 
print(re.search(r'[A-Z]', 'Hello World'))
print(re.findall(r'[A-Z]', 'Hello World'))

output:
<re.Match object; span=(0, 1), match='H'>
['H', 'W']
```  
###使用group  
也就是说如果使用search，打印出来的个对象，还需要使用group()把内容显示出来。
``` 
obj = re.search(r' (\w+) (\w+)', 'I am HaHa.')
print(obj.group())  

output:
 am HaHa
```  
也可以指定输出指定位置的查找结果：
``` 
obj = re.search(r' (\w+) (\w+)', 'I am HaHa.')
obj.group()
Out[4]: ' am HaHa'
obj.group(1)
Out[5]: 'am'
obj.group(2)
Out[6]: 'HaHa'
```  
####在pip install 时，如果C盘的user用户名是中文，可能会在cmd报转码的错误，  
####解决方法如下：  
在Python/Lib/site-packages里面加个py文件叫：sitecumtomize.py,  
里面内容如下：  
``` 
import sys
sys.setdefaultencoding('gb2312')
```

