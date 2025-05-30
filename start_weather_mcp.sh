#!/bin/bash

# 天气查询MCP服务启动脚本

echo "🌤️ 启动天气查询MCP服务..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python 3.7或更高版本，当前版本: $python_version"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo "📥 安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动服务器
echo "🚀 启动MCP服务器..."
python3 weather_mcp_server.py