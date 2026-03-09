#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行所有 Redis 集成测试
"""
import sys
import os
import subprocess
from pathlib import Path

# 测试文件列表
TESTS = [
    ("Redis 客户端基础测试", "test_redis_client.py"),
    ("任务状态管理测试", "test_task_status.py"),
    ("完整集成测试", "test_redis_integration.py"),
    ("综合测试（推荐）", "test_redis_comprehensive.py"),
    ("性能基准测试", "test_redis_benchmark.py"),
]


def run_test(test_name, test_file):
    """运行单个测试"""
    print("\n" + "=" * 70)
    print(f"运行测试: {test_name}")
    print("=" * 70)
    
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"\n✅ {test_name} 通过")
            return True
        else:
            print(f"\n❌ {test_name} 失败 (退出码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n❌ {test_name} 执行异常: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("BiliNote Redis 集成测试套件")
    print("=" * 70)
    print(f"\n共 {len(TESTS)} 个测试文件")
    
    results = []
    
    for test_name, test_file in TESTS:
        success = run_test(test_name, test_file)
        results.append((test_name, success))
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    
    if failed > 0:
        print("\n失败的测试:")
        for test_name, success in results:
            if not success:
                print(f"  - {test_name}")
    
    print("\n" + "=" * 70)
    
    if failed == 0:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误详情")
        return 1


if __name__ == "__main__":
    sys.exit(main())
