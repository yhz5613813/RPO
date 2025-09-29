import hashlib

class GRPOCache:
    def __init__(self, data):
        """
        初始化方法，接收一个字典列表，每个字典包含instruction和answer键
        
        参数:
            dict_list (list): 包含字典的列表，每个字典必须有instruction和answer键
        """
        self.data = data
        if "cache" not in self.data.column_names:  # 如果cache没有被初始化那么需要对cache初始化一下
            def add_new_column(example):
                example["cache"] = None
                return example
            self.data = self.data.map(add_new_column)
            print(f"grpo cache dataset has been initialed")
        self.cache = {item["uuid"]: item["cache"] for item in self.data}
        
    
    def _generate_hash(self, text):
        """
        生成字符串的哈希值
        
        参数:
            text (str): 要哈希化的文本
            
        返回:
            str: 文本的SHA256哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def update(self, hash_value, new_prompt):
        """
        根据哈希值更新对应项的prompt字段
        
        参数:
            hash_value (str): 要更新的项的哈希值
            new_prompt (str): 新的prompt内容
            
        返回:
            bool: 如果找到并更新了对应项返回True，否则返回False
        """
        if hash_value in self.hash_map:
            index = self.hash_map[hash_value]
            self.data[index]['prompt'] = new_prompt
            return True
        return False
    
    def __getitem__(self, index):
        """
        支持索引访问
        
        参数:
            index (int): 要访问的索引
            
        返回:
            dict: 对应索引的数据项
        """
        return self.data[index]
    
    def __len__(self):
        """
        返回数据项的数量
        
        返回:
            int: 数据项的数量
        """
        return len(self.data)
    
    def __iter__(self):
        """
        使类可迭代
        
        返回:
            iterator: 数据列表的迭代器
        """
        return iter(self.data)
    
    def find_by_hash(self, hash_value):
        """
        根据哈希值查找数据项
        
        参数:
            hash_value (str): 要查找的哈希值
            
        返回:
            dict or None: 找到的数据项，如果没找到返回None
        """
        if hash_value in self.hash_map:
            return self.data[self.hash_map[hash_value]]
        return None


# # 示例用法
# if __name__ == "__main__":
#     # 示例数据
#     sample_data = [
#         {'instruction': '打开文件', 'answer': '使用open()函数'},
#         {'instruction': '关闭文件', 'answer': '使用close()方法'},
#         {'instruction': '读取文件内容', 'answer': '使用read()方法'}
#     ]
    
#     # 初始化类
#     dataset = InstructionDataset(sample_data)
    
#     # 打印初始数据
#     print("初始数据:")
#     for item in dataset:
#         print(item)
    
#     # 更新一个项的prompt
#     first_hash = dataset[0]['hash']
#     dataset.update(first_hash, "这是一个关于文件操作的提示")
    
#     # 打印更新后的数据
#     print("\n更新后的数据:")
#     for item in dataset:
#         print(item)
    
#     # 查找特定哈希的项
#     found_item = dataset.find_by_hash(first_hash)
#     print("\n找到的项:", found_item)
    
#     # 测试性能
#     import time
#     start = time.time()
#     for _ in range(10000):
#         dataset.find_by_hash(first_hash)
#     end = time.time()
#     print(f"\n查找性能测试: 10000次查找耗时 {end-start:.6f} 秒")