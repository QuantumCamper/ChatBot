# 天气查询MCP服务

一个基于Python实现的天气查询MCP（Model Context Protocol）服务，支持实时天气信息查询和天气预报功能。

## 🌟 功能特性

- **实时天气查询**: 获取指定城市的当前天气信息
- **天气预报**: 查询未来1-5天的天气预报
- **多语言支持**: 支持中文和英文城市名称
- **模拟数据**: 当API不可用时提供模拟天气数据
- **MCP协议兼容**: 完全符合MCP 2024-11-05协议规范

## 📋 系统要求

- Python 3.7+
- aiohttp库
- asyncio支持

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥（可选）

编辑 `config.json` 文件，添加您的OpenWeatherMap API密钥：

```json
{
  "weather_api": {
    "api_key": "your_openweathermap_api_key_here"
  }
}
```

> 注意：如果没有API密钥，服务将使用模拟数据进行演示。

### 3. 启动MCP服务器

```bash
python weather_mcp_server.py
```

### 4. 运行测试客户端

**交互式模式：**
```bash
python test_weather_mcp.py
```

**自动演示模式：**
```bash
python test_weather_mcp.py --auto
```

## 🔧 MCP工具说明

### get_weather

获取指定城市的当前天气信息。

**参数：**
- `city` (必需): 城市名称（中文或英文）
- `country_code` (可选): 国家代码（如CN、US等）

**示例：**
```json
{
  "name": "get_weather",
  "arguments": {
    "city": "北京",
    "country_code": "CN"
  }
}
```

### get_weather_forecast

获取指定城市的天气预报。

**参数：**
- `city` (必需): 城市名称
- `days` (可选): 预报天数（1-5天，默认3天）

**示例：**
```json
{
  "name": "get_weather_forecast",
  "arguments": {
    "city": "上海",
    "days": 5
  }
}
```

## 📊 返回数据格式

### 天气信息包含：

- 🌡️ **温度**: 当前温度和体感温度
- ☁️ **天气描述**: 天气状况描述
- 💧 **湿度**: 相对湿度百分比
- 💨 **风速**: 风速（米/秒）
- 🔽 **气压**: 大气压强（百帕）
- 🕐 **更新时间**: 数据更新时间戳

### 示例输出：

```
🌤️ 北京 当前天气

🌡️ 温度: 15°C (体感温度: 13°C)
☁️ 天气: 晴朗
💧 湿度: 65%
💨 风速: 3.2 m/s
🔽 气压: 1013 hPa
🕐 更新时间: 2024-05-30 14:30:25
```

## 🏗️ 项目结构

```
weather-mcp/
├── weather_mcp_server.py    # MCP服务器主程序
├── test_weather_mcp.py      # 测试客户端
├── config.json              # 配置文件
├── requirements.txt         # Python依赖
└── README_Weather_MCP.md    # 项目文档
```

## 🔌 MCP协议集成

### 初始化连接

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "weather-client",
      "version": "1.0.0"
    }
  }
}
```

### 获取工具列表

```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "tools/list"
}
```

### 调用工具

```json
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "北京"
    }
  }
}
```

## 🛠️ 自定义配置

### 修改模拟数据

编辑 `config.json` 中的 `mock_data` 部分来自定义模拟天气数据：

```json
{
  "mock_data": {
    "cities": {
      "你的城市": {"temp": 20, "desc": "晴朗"}
    }
  }
}
```

### 添加新的天气数据源

在 `WeatherMCPServer` 类中添加新的数据源支持：

```python
async def get_weather_from_custom_api(self, city: str):
    # 实现自定义API调用
    pass
```

## 🐛 故障排除

### 常见问题

1. **API密钥无效**
   - 检查 `config.json` 中的API密钥是否正确
   - 确认API密钥有效且未过期

2. **网络连接问题**
   - 检查网络连接
   - 确认防火墙设置允许HTTP请求

3. **依赖安装失败**
   - 使用 `pip install --upgrade pip` 更新pip
   - 尝试使用虚拟环境安装依赖

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenWeatherMap API](https://openweathermap.org/api) - 天气数据提供
- [MCP Protocol](https://modelcontextprotocol.io/) - 协议规范
- [aiohttp](https://aiohttp.readthedocs.io/) - 异步HTTP客户端

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 [Issue](https://github.com/your-repo/weather-mcp/issues)
- 发送邮件至：your-email@example.com

---

**享受使用天气查询MCP服务！** 🌤️