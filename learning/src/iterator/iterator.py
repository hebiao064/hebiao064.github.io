#!/usr/bin/env python3
"""
Iterator Examples - 迭代器示例
包含多种iterator实现方式和应用场景
"""

# 1. 基础Iterator类 - 实现迭代器协议
class NumberIterator:
    """数字迭代器：从start到end（不包含）"""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.end:
            num = self.current
            self.current += 1
            return num
        else:
            raise StopIteration


# 2. 生成器函数 - 更简洁的iterator实现
def fibonacci_generator(n):
    """生成前n个斐波那契数"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def even_numbers(start, end):
    """生成指定范围内的偶数"""
    for num in range(start, end):
        if num % 2 == 0:
            yield num


# 3. 文件读取迭代器
class FileLineIterator:
    """逐行读取文件的迭代器"""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __iter__(self):
        self.file = open(self.filename, 'r', encoding='utf-8')
        return self
    
    def __next__(self):
        if self.file is None:
            raise StopIteration
        
        line = self.file.readline()
        if line:
            return line.rstrip('\n')
        else:
            self.file.close()
            raise StopIteration


# 4. 批量处理迭代器
class BatchIterator:
    """将数据分批处理的迭代器"""
    
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch


# 5. 链式迭代器
class ChainIterator:
    """链接多个可迭代对象的迭代器"""
    
    def __init__(self, *iterables):
        self.iterables = iterables
        self.current_iterable = 0
        self.current_iterator = iter(self.iterables[0]) if self.iterables else None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iterator is None:
            raise StopIteration
        
        try:
            return next(self.current_iterator)
        except StopIteration:
            self.current_iterable += 1
            if self.current_iterable < len(self.iterables):
                self.current_iterator = iter(self.iterables[self.current_iterable])
                return next(self.current_iterator)
            else:
                raise StopIteration


# 6. 无限迭代器
def infinite_counter(start=0, step=1):
    """无限计数器生成器"""
    current = start
    while True:
        yield current
        current += step


# 7. 过滤迭代器
class FilterIterator:
    """根据条件过滤数据的迭代器"""
    
    def __init__(self, data, predicate):
        self.data = data
        self.predicate = predicate
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while self.index < len(self.data):
            item = self.data[self.index]
            self.index += 1
            if self.predicate(item):
                return item
        raise StopIteration


# 8. 树遍历迭代器
class TreeNode:
    """简单的树节点"""
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []


class TreeIterator:
    """深度优先遍历树的迭代器"""
    
    def __init__(self, root):
        self.stack = [root] if root else []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        
        node = self.stack.pop()
        # 逆序添加子节点，确保左子树先被访问
        self.stack.extend(reversed(node.children))
        return node.value


# 使用示例和测试
def main():
    print("=== Iterator Examples ===\n")
    
    # 1. 基础Iterator
    print("1. 基础Iterator:")
    for num in NumberIterator(1, 5):
        print(f"  {num}")
    
    # 2. 生成器
    print("\n2. 斐波那契生成器:")
    for fib in fibonacci_generator(8):
        print(f"  {fib}")
    
    print("\n3. 偶数生成器:")
    for even in even_numbers(1, 10):
        print(f"  {even}")
    
    # 4. 批量处理
    print("\n4. 批量处理迭代器:")
    data = list(range(10))
    for batch in BatchIterator(data, 3):
        print(f"  批次: {batch}")
    
    # 5. 链式迭代器
    print("\n5. 链式迭代器:")
    for item in ChainIterator([1, 2, 3], "abc", (4, 5)):
        print(f"  {item}")
    
    # 6. 无限迭代器（只取前5个）
    print("\n6. 无限计数器（前5个）:")
    counter = infinite_counter(10, 2)
    for i, value in enumerate(counter):
        if i >= 5:
            break
        print(f"  {value}")
    
    # 7. 过滤迭代器
    print("\n7. 过滤迭代器（只要偶数）:")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for even in FilterIterator(numbers, lambda x: x % 2 == 0):
        print(f"  {even}")
    
    # 8. 树遍历迭代器
    print("\n8. 树遍历迭代器:")
    # 构建一个简单的树
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    root = TreeNode(1, [
        TreeNode(2, [TreeNode(4), TreeNode(5)]),
        TreeNode(3)
    ])
    
    print("  深度优先遍历:")
    for value in TreeIterator(root):
        print(f"    {value}")
    
    # 9. 使用内置函数配合迭代器
    print("\n9. 与内置函数配合使用:")
    # 使用map
    squared = map(lambda x: x**2, NumberIterator(1, 6))
    print(f"  平方数: {list(squared)}")
    
    # 使用filter
    filtered = filter(lambda x: x > 3, NumberIterator(1, 8))
    print(f"  大于3的数: {list(filtered)}")
    
    # 使用enumerate
    print("  带索引的斐波那契数:")
    for i, fib in enumerate(fibonacci_generator(5)):
        print(f"    第{i}个: {fib}")


class NaiveIterator:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        return_value = self.current
        self.current += 1
        return return_value

def gen_test():
    for i in range(1, 5):
        yield i

if __name__ == "__main__":
    main() 
     # 0. 我写的Iterator
    print("0. 我写的Iterator:")
    for num in NaiveIterator(1, 5):
        print(f"  {num}")

    generator = gen_test()
    print(generator.__next__())
    print(generator.__next__())
    