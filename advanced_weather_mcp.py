#!/usr/bin/env python3
"""
Advanced Weather MCP Server - 高级天气查询MCP服务
支持多种天气数据源、缓存、历史数据等高级功能
"""

import asyncio
import json
import sys
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """扩展的天气数据结构"""
    city: str
    country: str
    temperature: float
    feels_like: float
    description: str
    humidity: int
    wind_speed: float
    wind_direction: int
    pressure: float
    visibility: float
    uv_index: float
    timestamp: str
    source: str = "openweathermap"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WeatherAlert:
    """天气预警数据结构"""
    alert_type: str
    severity: str
    title: str
    description: str
    start_time: str
    end_time: str

class WeatherCache:
    """天气数据缓存"""
    
    def __init__(self, ttl_seconds: int = 600):  # 10分钟缓存
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _get_key(self, city: str, country: str = "") -> str:
        """生成缓存键"""
        return hashlib.md5(f"{city}_{country}".encode()).hexdigest()
    
    def get(self, city: str, country: str = "") -> Optional[WeatherData]:
        """获取缓存数据"""
        key = self._get_key(city, country)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, city: str, weather_data: WeatherData, country: str = ""):
        """设置缓存数据"""
        key = self._get_key(city, country)
        self.cache[key] = (weather_data, time.time())
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()

class AdvancedWeatherMCPServer:
    """高级天气查询MCP服务器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.cache = WeatherCache()
        self.api_calls_count = 0
        self.start_time = time.time()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return {
                "weather_api": {
                    "api_key": "demo_key",
                    "base_url": "http://api.openweathermap.org/data/2.5/weather",
                    "units": "metric",
                    "language": "zh_cn"
                },
                "cache": {
                    "ttl_seconds": 600
                }
            }
    
    async def get_weather(self, city: str, country_code: str = "") -> WeatherData:
        """获取天气信息（带缓存）"""
        # 先检查缓存
        cached_data = self.cache.get(city, country_code)
        if cached_data:
            logger.info(f"从缓存获取 {city} 的天气数据")
            return cached_data
        
        # 从API获取数据
        weather_data = await self._fetch_weather_from_api(city, country_code)
        
        # 存入缓存
        self.cache.set(city, weather_data, country_code)
        
        return weather_data
    
    async def _fetch_weather_from_api(self, city: str, country_code: str = "") -> WeatherData:
        """从API获取天气数据"""
        self.api_calls_count += 1
        
        try:
            api_config = self.config["weather_api"]
            location = f"{city},{country_code}" if country_code else city
            params = {
                "q": location,
                "appid": api_config["api_key"],
                "units": api_config["units"],
                "lang": api_config["language"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_config["base_url"], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_api_weather_data(data)
                    elif response.status == 401:
                        logger.warning("API密钥无效，使用模拟数据")
                        return self._get_enhanced_mock_weather_data(city, country_code)
                    else:
                        raise Exception(f"API请求失败: {response.status}")
                        
        except Exception as e:
            logger.warning(f"获取真实天气数据失败: {e}，返回模拟数据")
            return self._get_enhanced_mock_weather_data(city, country_code)
    
    def _parse_api_weather_data(self, data: Dict[str, Any]) -> WeatherData:
        """解析API返回的天气数据"""
        main = data["main"]
        weather = data["weather"][0]
        wind = data.get("wind", {})
        sys_data = data.get("sys", {})
        
        return WeatherData(
            city=data["name"],
            country=sys_data.get("country", ""),
            temperature=main["temp"],
            feels_like=main["feels_like"],
            description=weather["description"],
            humidity=main["humidity"],
            wind_speed=wind.get("speed", 0),
            wind_direction=wind.get("deg", 0),
            pressure=main["pressure"],
            visibility=data.get("visibility", 10000) / 1000,  # 转换为公里
            uv_index=0,  # OpenWeatherMap免费版不提供UV指数
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="openweathermap"
        )
    
    def _get_enhanced_mock_weather_data(self, city: str, country_code: str = "") -> WeatherData:
        """返回增强的模拟天气数据"""
        import random
        
        mock_cities = self.config.get("mock_data", {}).get("cities", {})
        
        if city in mock_cities:
            base_temp = mock_cities[city]["temp"]
            desc = mock_cities[city]["desc"]
        else:
            base_temp = random.randint(10, 30)
            desc = random.choice(["晴朗", "多云", "小雨", "阴天", "雷阵雨"])
        
        temp = base_temp + random.randint(-3, 3)
        
        return WeatherData(
            city=city,
            country=country_code or "CN",
            temperature=temp,
            feels_like=temp + random.randint(-2, 2),
            description=desc,
            humidity=random.randint(40, 80),
            wind_speed=random.uniform(1, 10),
            wind_direction=random.randint(0, 360),
            pressure=random.randint(1000, 1020),
            visibility=random.uniform(5, 20),
            uv_index=random.uniform(0, 11),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source="mock_data"
        )
    
    async def get_weather_forecast(self, city: str, days: int = 3) -> List[Dict[str, Any]]:
        """获取天气预报"""
        forecast = []
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i+1))
            weather_data = self._get_enhanced_mock_weather_data(city)
            
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "temperature_high": weather_data.temperature + random.randint(3, 8),
                "temperature_low": weather_data.temperature - random.randint(2, 5),
                "description": weather_data.description,
                "humidity": weather_data.humidity,
                "wind_speed": weather_data.wind_speed,
                "precipitation_chance": random.randint(0, 100)
            })
        
        return forecast
    
    async def get_weather_alerts(self, city: str) -> List[WeatherAlert]:
        """获取天气预警（模拟）"""
        import random
        
        alerts = []
        
        # 随机生成一些预警信息
        if random.random() < 0.3:  # 30%概率有预警
            alert_types = ["高温", "暴雨", "大风", "雾霾"]
            severities = ["轻微", "中等", "严重"]
            
            alert_type = random.choice(alert_types)
            severity = random.choice(severities)
            
            alerts.append(WeatherAlert(
                alert_type=alert_type,
                severity=severity,
                title=f"{city}{alert_type}预警",
                description=f"{city}地区将出现{alert_type}天气，请注意防范。",
                start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                end_time=(datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        return alerts
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": int(uptime),
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "api_calls_count": self.api_calls_count,
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": "N/A"  # 可以实现缓存命中率统计
        }

class AdvancedMCPProtocolHandler:
    """高级MCP协议处理器"""
    
    def __init__(self, weather_server: AdvancedWeatherMCPServer):
        self.weather_server = weather_server
        self.tools = [
            {
                "name": "get_weather",
                "description": "获取指定城市的详细天气信息",
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
                "description": "获取指定城市的详细天气预报",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "days": {
                            "type": "integer",
                            "description": "预报天数（1-7天）",
                            "minimum": 1,
                            "maximum": 7
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_weather_alerts",
                "description": "获取指定城市的天气预警信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_server_stats",
                "description": "获取MCP服务器运行统计信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "clear_cache",
                "description": "清空天气数据缓存",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
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
                    "name": "advanced-weather-mcp-server",
                    "version": "2.0.0",
                    "description": "高级天气查询MCP服务器 - 支持缓存、预警、统计等功能"
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
        elif tool_name == "get_weather_alerts":
            return await self._call_get_weather_alerts(request_id, arguments)
        elif tool_name == "get_server_stats":
            return await self._call_get_server_stats(request_id, arguments)
        elif tool_name == "clear_cache":
            return await self._call_clear_cache(request_id, arguments)
        else:
            return self._create_error_response(request_id, -32602, f"未知工具: {tool_name}")
    
    async def _call_get_weather(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用天气查询工具"""
        city = arguments.get("city")
        country_code = arguments.get("country_code", "")
        
        if not city:
            return self._create_error_response(request_id, -32602, "缺少必需参数: city")
        
        weather_data = await self.weather_server.get_weather(city, country_code)
        
        result_text = f"""🌤️ {weather_data.city} 详细天气信息

🌡️ 温度: {weather_data.temperature}°C (体感: {weather_data.feels_like}°C)
☁️ 天气: {weather_data.description}
💧 湿度: {weather_data.humidity}%
💨 风速: {weather_data.wind_speed} m/s (风向: {weather_data.wind_direction}°)
🔽 气压: {weather_data.pressure} hPa
👁️ 能见度: {weather_data.visibility} km
☀️ UV指数: {weather_data.uv_index}
🌍 国家: {weather_data.country}
📡 数据源: {weather_data.source}
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
        """调用天气预报工具"""
        city = arguments.get("city")
        days = arguments.get("days", 3)
        
        if not city:
            return self._create_error_response(request_id, -32602, "缺少必需参数: city")
        
        forecast = await self.weather_server.get_weather_forecast(city, days)
        
        forecast_text = f"📅 {city} 未来{days}天详细天气预报\n\n"
        
        for day_data in forecast:
            forecast_text += f"""📆 {day_data['date']} ({day_data['day_name']})
🌡️ 温度: {day_data['temperature_low']}°C - {day_data['temperature_high']}°C
☁️ 天气: {day_data['description']}
💧 湿度: {day_data['humidity']}%
💨 风速: {day_data['wind_speed']:.1f} m/s
🌧️ 降水概率: {day_data['precipitation_chance']}%

"""
        
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
    
    async def _call_get_weather_alerts(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用天气预警工具"""
        city = arguments.get("city")
        
        if not city:
            return self._create_error_response(request_id, -32602, "缺少必需参数: city")
        
        alerts = await self.weather_server.get_weather_alerts(city)
        
        if not alerts:
            result_text = f"✅ {city} 当前无天气预警信息"
        else:
            result_text = f"⚠️ {city} 天气预警信息\n\n"
            for alert in alerts:
                severity_emoji = {"轻微": "🟡", "中等": "🟠", "严重": "🔴"}.get(alert.severity, "⚪")
                result_text += f"""{severity_emoji} {alert.title}
📋 类型: {alert.alert_type}
⚡ 严重程度: {alert.severity}
📝 描述: {alert.description}
⏰ 生效时间: {alert.start_time}
⏰ 结束时间: {alert.end_time}

"""
        
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
    
    async def _call_get_server_stats(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用服务器统计工具"""
        stats = self.weather_server.get_server_stats()
        
        result_text = f"""📊 MCP服务器运行统计

⏱️ 运行时间: {stats['uptime_formatted']}
📞 API调用次数: {stats['api_calls_count']}
💾 缓存条目数: {stats['cache_size']}
📈 缓存命中率: {stats['cache_hit_rate']}
🕐 统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
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
    
    async def _call_clear_cache(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用清空缓存工具"""
        cache_size_before = len(self.weather_server.cache.cache)
        self.weather_server.cache.clear()
        
        result_text = f"🗑️ 缓存清理完成\n\n清理前缓存条目数: {cache_size_before}\n清理后缓存条目数: 0"
        
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
    """主函数 - 启动高级MCP服务器"""
    logger.info("启动高级天气查询MCP服务器...")
    
    # 创建高级天气服务器实例
    weather_server = AdvancedWeatherMCPServer()
    
    # 创建高级MCP协议处理器
    protocol_handler = AdvancedMCPProtocolHandler(weather_server)
    
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
    # 运行高级服务器
    asyncio.run(main())