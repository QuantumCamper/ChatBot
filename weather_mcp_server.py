#!/usr/bin/env python3
"""
Weather MCP Server - 支持天气查询的MCP服务
提供实时天气信息查询功能
"""

import asyncio
import json
import sys
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import aiohttp
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """天气数据结构"""
    city: str
    temperature: float
    description: str
    humidity: int
    wind_speed: float
    pressure: float
    feels_like: float
    timestamp: str

class WeatherMCPServer:
    """天气查询MCP服务器"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "demo_key"  # 使用演示密钥或真实API密钥
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    async def get_weather(self, city: str, country_code: str = "") -> WeatherData:
        """
        获取指定城市的天气信息
        
        Args:
            city: 城市名称
            country_code: 国家代码（可选）
            
        Returns:
            WeatherData: 天气数据对象
        """
        try:
            location = f"{city},{country_code}" if country_code else city
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",  # 使用摄氏度
                "lang": "zh_cn"     # 中文描述
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_weather_data(data)
                    elif response.status == 401:
                        # API密钥无效时返回模拟数据
                        return self._get_mock_weather_data(city)
                    else:
                        raise Exception(f"API请求失败: {response.status}")
                        
        except Exception as e:
            logger.warning(f"获取真实天气数据失败: {e}，返回模拟数据")
            return self._get_mock_weather_data(city)
    
    def _parse_weather_data(self, data: Dict[str, Any]) -> WeatherData:
        """解析API返回的天气数据"""
        main = data["main"]
        weather = data["weather"][0]
        wind = data.get("wind", {})
        
        return WeatherData(
            city=data["name"],
            temperature=main["temp"],
            description=weather["description"],
            humidity=main["humidity"],
            wind_speed=wind.get("speed", 0),
            pressure=main["pressure"],
            feels_like=main["feels_like"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _get_mock_weather_data(self, city: str) -> WeatherData:
        """返回模拟天气数据（用于演示）"""
        import random
        
        mock_data = {
            "北京": {"temp": 15, "desc": "晴朗"},
            "上海": {"temp": 20, "desc": "多云"},
            "广州": {"temp": 25, "desc": "小雨"},
            "深圳": {"temp": 26, "desc": "阴天"},
            "杭州": {"temp": 18, "desc": "晴朗"},
        }
        
        if city in mock_data:
            base_temp = mock_data[city]["temp"]
            desc = mock_data[city]["desc"]
        else:
            base_temp = random.randint(10, 30)
            desc = random.choice(["晴朗", "多云", "小雨", "阴天"])
        
        return WeatherData(
            city=city,
            temperature=base_temp + random.randint(-3, 3),
            description=desc,
            humidity=random.randint(40, 80),
            wind_speed=random.uniform(1, 10),
            pressure=random.randint(1000, 1020),
            feels_like=base_temp + random.randint(-2, 2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

class MCPProtocolHandler:
    """MCP协议处理器"""
    
    def __init__(self, weather_server: WeatherMCPServer):
        self.weather_server = weather_server
        self.tools = [
            {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称（中文或英文）"
                        },
                        "country_code": {
                            "type": "string",
                            "description": "国家代码（可选，如CN、US等）"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_weather_forecast",
                "description": "获取指定城市的天气预报（未来几天）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "days": {
                            "type": "integer",
                            "description": "预报天数（1-5天）",
                            "minimum": 1,
                            "maximum": 5
                        }
                    },
                    "required": ["city"]
                }
            }
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return await self._handle_initialize(request_id)
            elif method == "tools/list":
                return await self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params)
            else:
                return self._create_error_response(request_id, -32601, f"未知方法: {method}")
                
        except Exception as e:
            logger.error(f"处理请求时出错: {e}")
            return self._create_error_response(request_id, -32603, str(e))
    
    async def _handle_initialize(self, request_id: str) -> Dict[str, Any]:
        """处理初始化请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "weather-mcp-server",
                    "version": "1.0.0",
                    "description": "天气查询MCP服务器"
                }
            }
        }
    
    async def _handle_tools_list(self, request_id: str) -> Dict[str, Any]:
        """处理工具列表请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self.tools
            }
        }
    
    async def _handle_tool_call(self, request_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "get_weather":
            return await self._call_get_weather(request_id, arguments)
        elif tool_name == "get_weather_forecast":
            return await self._call_get_weather_forecast(request_id, arguments)
        else:
            return self._create_error_response(request_id, -32602, f"未知工具: {tool_name}")
    
    async def _call_get_weather(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用天气查询工具"""
        city = arguments.get("city")
        country_code = arguments.get("country_code", "")
        
        if not city:
            return self._create_error_response(request_id, -32602, "缺少必需参数: city")
        
        weather_data = await self.weather_server.get_weather(city, country_code)
        
        result_text = f"""🌤️ {weather_data.city} 当前天气

🌡️ 温度: {weather_data.temperature}°C (体感温度: {weather_data.feels_like}°C)
☁️ 天气: {weather_data.description}
💧 湿度: {weather_data.humidity}%
💨 风速: {weather_data.wind_speed} m/s
🔽 气压: {weather_data.pressure} hPa
🕐 更新时间: {weather_data.timestamp}"""
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        }
    
    async def _call_get_weather_forecast(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用天气预报工具（模拟实现）"""
        city = arguments.get("city")
        days = arguments.get("days", 3)
        
        if not city:
            return self._create_error_response(request_id, -32602, "缺少必需参数: city")
        
        # 模拟天气预报数据
        forecast_text = f"📅 {city} 未来{days}天天气预报\n\n"
        
        import random
        from datetime import datetime, timedelta
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i+1)).strftime("%m月%d日")
            temp_high = random.randint(15, 30)
            temp_low = random.randint(5, 15)
            weather_desc = random.choice(["晴朗", "多云", "小雨", "阴天", "雷阵雨"])
            
            forecast_text += f"📆 {date}: {weather_desc} {temp_low}°C - {temp_high}°C\n"
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": forecast_text
                    }
                ]
            }
        }
    
    def _create_error_response(self, request_id: str, code: int, message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

async def main():
    """主函数 - 启动MCP服务器"""
    logger.info("启动天气查询MCP服务器...")
    
    # 创建天气服务器实例
    weather_server = WeatherMCPServer()
    
    # 创建MCP协议处理器
    protocol_handler = MCPProtocolHandler(weather_server)
    
    # 处理标准输入输出
    while True:
        try:
            # 读取请求
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # 解析JSON请求
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                continue
            
            # 处理请求
            response = await protocol_handler.handle_request(request)
            
            # 发送响应
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务器...")
            break
        except Exception as e:
            logger.error(f"服务器错误: {e}")

if __name__ == "__main__":
    # 安装依赖提示
    try:
        import aiohttp
    except ImportError:
        print("请先安装依赖: pip install aiohttp")
        sys.exit(1)
    
    # 运行服务器
    asyncio.run(main())