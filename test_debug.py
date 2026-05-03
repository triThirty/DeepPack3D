#!/usr/bin/env python
"""
多线程调试测试脚本
"""
import sys
sys.path.insert(0, '/Users/maoshanshi/Workspace/code/3D-packing/DeepPack3D')

from gp.evaluate import evaluate


class DummyAgent:
    """模拟 agent 对象"""
    def __init__(self):
        self.name = "test_agent"


class DummyIndividual:
    """模拟 individual 对象"""
    def __init__(self, idx):
        self.idx = idx
    
    def __repr__(self):
        return f"Individual({self.idx})"


if __name__ == "__main__":
    agent = DummyAgent()
    print(f"Main Agent ID: {id(agent)}")
    
    # 创建 population
    population = [DummyIndividual(i) for i in range(4)]
    
    # 调用 evaluate，会在多线程中运行
    print("\n开始调试，设置断点后启动...\n")
    results = evaluate(population, agent)
    
    print(f"\n结果: {results}")
