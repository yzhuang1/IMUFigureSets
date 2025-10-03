#!/bin/bash
set -euo pipefail   # 严格模式

for i in {1..5}
do
    echo "===== 第 $i 次运行开始 ====="
    
    # 即使 main.py 报错，也继续下一次
    if python3 main.py; then
        echo "第 $i 次运行成功"
    else
        echo "第 $i 次运行失败，继续下一次"
    fi

    echo "===== 第 $i 次运行结束 ====="
    echo
done

echo "所有 10 次运行已完成！（不论报错与否）"
