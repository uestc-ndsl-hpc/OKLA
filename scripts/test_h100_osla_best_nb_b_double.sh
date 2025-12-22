#!/bin/bash

EXE="./build/okla-test"
M=32768
NRHS=32768
LOG_FILE="./logs/benchmark_nb_b_double.csv"

mkdir -p ./logs
echo "NB,b,Cusolver_TFLOPS,OSLA_TFLOPS" > $LOG_FILE

NB_LIST="4096 8192 16384"
B_LIST="128 256 512 1024 2048"

echo "开始 NB & b 双重扫描测试 (H100)..."

for nb in $NB_LIST; do
    for b in $B_LIST; do
        if [ $b -ge $nb ]; then continue; fi

        echo -n "Testing NB=$nb, b=$b ... "

        # 执行并将标准输出和错误输出都捕获，同时去掉颜色代码
        raw_output=$($EXE --double --test-cusolver-trsm --test-osla-trsm --m=$M --nrhs=$NRHS --nb=$nb --b=$b -v 2>&1 | sed 's/\x1b\[[0-9;]*m//g')

        # 提取数值：匹配 "[TFLOPS] Cusolver TRSM: <num> TFLOPS"
        CUSOLVER_T=$(echo "$raw_output" | awk '/\[TFLOPS\].*Cusolver TRSM:/ {print $(NF-1)}' | tail -n 1)
        # 提取数值：匹配 "[TFLOPS] OSLA TRSM: <num> TFLOPS"
        OSLA_T=$(echo "$raw_output" | awk '/\[TFLOPS\].*OSLA TRSM:/ {print $(NF-1)}' | tail -n 1)

        # 调试：如果还是抓不到，打印这一行看看
        if [ -z "$OSLA_T" ]; then
            echo "Error! Raw line: $(echo "$raw_output" | grep "OSLA" | grep "TFLOPS" | tail -n 1)"
            OSLA_T=0
        fi
        
        CUSOLVER_T=${CUSOLVER_T:-0}
        OSLA_T=${OSLA_T:-0}

        echo "$nb,$b,$CUSOLVER_T,$OSLA_T" >> $LOG_FILE
        echo "OSLA: $OSLA_T TFLOPS"
    done
done

echo "-----------------------------------------------"
sort -t',' -k4 -nr $LOG_FILE | head -n 3
